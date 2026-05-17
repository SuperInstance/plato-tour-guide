# CUDA Kernel Style vs Fortran: A Philosophical Comparison

*Or: How to think in CUDA without forgetting everything you learned in Fortran.*

---

## Overview

CUDA and Fortran share a fundamental philosophy: **the machine does what you tell it, explicitly**. No virtual dispatch, no runtime polymorphism, no garbage collection. This document maps Fortran mental models to CUDA constructs so you can write kernels the way a Fortran programmer would — without reaching for OOP patterns that don't belong on the GPU.

> **Core thesis:** CUDA thread = Fortran job. Global memory = COMMON block. Register pressure = register allocation. Warp = vector unit. If this makes sense, you're ready to write CUDA the Fortran way.

---

## 1. CUDA Thread = Fortran Job

### Fortran
```fortran
! Each Fortran "job" processes one row independently
DO i = 1, n_agents
  CALL process_row(embeddings(i,:), dist_matrix(i,:))
END DO
```

### CUDA
```cuda
// Each CUDA thread processes one row (grid dimension)
__global__ void cosine_distance_kernel(float* dist_matrix, 
                                       const float* embeddings, 
                                       int n_agents, int dim)
{
    int i = blockIdx.x;  // Fortran job index
    const float* emb_i = embeddings + i * dim;
    
    // ... compute row i
}
```

**Why it maps:**
- Fortran DO loops are implicitly sequential across "jobs" (MPI ranks, threads, or batch items)
- CUDA grid lets you launch thousands of "jobs" (threads) in parallel
- The `blockIdx.x` is your Fortran DO index, except every iteration runs *simultaneously*
- No mutexes needed if each thread writes to its own row (no read-after-write hazard)

**Fortran got right:** Data parallelism is about the same operation on different data. The DO loop makes this explicit. CUDA thread index is the same idea, but hardware-parallel instead of loop-overhead-parallel.

---

## 2. Register Pressure = Fortran Register Allocation

### Fortran
```fortran
! Only 4 registers in use at once — compiler manages this
DO i = 1, n
  a = x(i); b = y(i); c = z(i)
  r = a*b + c
END DO
```

### CUDA
```cuda
// Register allocation is YOUR job in CUDA
// Variables here = CUDA "registers" (limited per thread)
// Requesting too many = register spilling to slow local memory

__global__ void example(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // These are REGISTER variables — CUDA allocates them
    // If you have >64 registers live, CUDA spills to local memory (SLOW)
    float a = in[i];
    float b = in[i + n];
    float c = in[i + 2*n];
    float r = a*b + c;  // result in register, no heap allocation
    
    out[i] = r;
}
```

**Why it maps:**
- Fortran compilers manage register allocation for you (with `#regs` directives in Fortran 90+)
- CUDA exposes register pressure directly: each thread has a fixed register file
- RTX 4050 (sm_75): 64K registers per SM, ~255 registers per thread limit
- **Fortran rule:** Don't allocate arrays inside a DO loop. **CUDA rule:** Don't have more than ~64 local variables alive at once.

**The analogy:**
```
Fortran:  !OCL INTEGER #REGS=4
CUDA:     int a, b, c, r;  // 4 registers live in this scope
```

---

## 3. Warp = Fortran Vector Unit

### Fortran (with vector unit)
```fortran
! Vector Fortran: operations on 4-element vectors at once
! (or 8 for AVX-256, 16 for AVX-512)
REAL :: a(4), b(4), c(4)
c = a * b  ! Vector unit fires 4 FMAs in one cycle
```

### CUDA
```cuda
// A warp is 32 threads executing the same instruction simultaneously
// This is a SIMT (Single Instruction, Multiple Thread) vector unit
// All threads in a warp execute the same arithmetic on different data

// Warp-level reduction via shuffle (no shared memory needed!)
unsigned int mask = 0xFFFFFFFFu;
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(mask, val, offset);  // Horizontal add across warp
}
// After loop: val[0] contains the sum of all 32 lanes
```

**Why it maps:**
- Fortran vector units operate on N elements per cycle (N = vector length)
- CUDA warps operate on 32 elements per "cycle" (actually per clock per SM)
- Fortran vector operations map naturally to warp-level operations
- `__shfl_xor_sync` is like Fortran's horizontal add, but across 32 lanes instead of 4

**Key insight:** The shuffle operations (`__shfl_xor_sync`, `__shfl_up_sync`, etc.) are the CUDA equivalent of Fortran's horizontal reduction intrinsics (`SUM`, `MAX`, etc.) — but you have to write the reduction explicitly because CUDA exposes the SIMD nature.

---

## 4. Global Memory = COMMON Block

### Fortran
```fortran
! COMMON block: shared data accessible by multiple program units
! Lives in memory, not registers. Survives between subroutine calls.
REAL :: embeddings(1000, 512)
REAL :: dist_matrix(1000, 1000)
COMMON /CONSENSUS/ embeddings, dist_matrix

SUBROUTINE compute()
  ! All threads/rank access the same COMMON block
  DO i = 1, n_agents
    dist_matrix(i, j) = 1.0 - cosine_sim(embeddings(i,:), embeddings(j,:))
  END DO
END SUBROUTINE
```

### CUDA
```cuda
// Global memory: lives in device DRAM, accessible by all threads
// Persists across kernel launches. Equivalent to a module-level COMMON.
// Layout: [n_agents * dim] row-major (like Fortran with leading dimension)

// dist_matrix: [n_agents * n_agents] — same storage as Fortran COMMON
// embeddings: [n_agents * dim]

__global__ void cosine_distance_kernel(
    float* __restrict__ dist_matrix,    // COMMON /CONSENSUS/dist_matrix
    const float* __restrict__ embeddings, // COMMON /CONSENSUS/embeddings
    int n_agents,
    int dim)
{
    int i = blockIdx.x;
    // Thread i reads row i of embeddings from global memory
    const float* emb_i = embeddings + i * dim;
    
    // Writes to its row of dist_matrix — all threads writing different rows
    dist_matrix[i * n_agents + j] = distance;
}
```

**Why it maps:**
- Fortran COMMON lives in memory, not registers — global memory is the same
- All Fortran program units see the same addresses in COMMON — all threads see the same global memory addresses
- COMMON is allocated at link time — global memory is allocated at `cudaMalloc`
- The difference: Fortran COMMON has static lifetime, CUDA global memory is explicit (`cudaMalloc`/`cudaFree`)

**Performance note:** Global memory access is slow (400-800 cycles). Fortran has the same problem with COMMON blocks — you wouldn't read from a COMMON block inside a tight inner loop without caching. Use `__restrict__` and access patterns that are coalesced (like Fortran's leading dimension convention).

---

## 5. Shared Memory = Fortran LOCAL Arrays (with sync)

### Fortran
```fortran
SUBROUTINE compute_batch(embeddings, dist_matrix, n_agents, dim)
  REAL, INTENT(IN)  :: embeddings(dim, n_agents)
  REAL, INTENT(OUT) :: dist_matrix(n_agents, n_agents)
  
  ! LOCAL array: private to this call, but fast
  REAL :: row_norms(n_agents)  ! Computed once, reused many times
  
  DO i = 1, n_agents
    row_norms(i) = SQRT(DOT_PRODUCT(embeddings(:,i), embeddings(:,i)))
  END DO
  
  ! Now use row_norms for all pairwise computations
END SUBROUTINE
```

### CUDA
```cuda
// Shared memory: thread-block-private, survives between __syncthreads() calls
// Equivalent to a LOCAL array that all threads in the block can access
// Lifetime: duration of kernel invocation

__global__ void cosine_distance_with_caching(
    float* dist_matrix, const float* embeddings, int n_agents, int dim)
{
    extern __shared__ float shared_norms[];  // dynamic shared memory
    
    int i = blockIdx.x;
    int tid = threadIdx.x;
    
    // ---- Phase 1: compute all norms, store in shared memory ----
    float my_norm = 0.0f;
    for (int k = tid; k < dim; k += blockDim.x) {
        float x = embeddings[i * dim + k];
        my_norm += x * x;
    }
    
    // Reduction in shared memory (warp-level then block-level)
    // Store final norm in shared memory
    if (tid == 0) {
        // Reduction complete, norm stored at index i
    }
    __syncthreads();  // Wait for all threads before reading shared_norms[i]
    
    // ---- Phase 2: use cached norm for pairwise computations ----
    float norm_i = shared_norms[i];
    // ... compute distances using cached norms
}
```

**Why it maps:**
- Fortran LOCAL arrays are fast (registers or stack, not heap) — shared memory is fast (on-chip SRAM)
- Both survive across subroutine/kernel calls within the same scope
- Fortran's `!$OMP BARRIER` is analogous to `__syncthreads()` — explicit synchronization before consuming shared data
- The difference: Fortran LOCAL is per-subroutine-call, CUDA shared is per-thread-block

**Performance note:** Shared memory latency is ~1000x lower than global memory. If you're reading the same data multiple times in a kernel, cache it in shared memory like a LOCAL array.

---

## 6. Fused Multiply-Add = Fortran FMAD

### Fortran
```fortran
! FMAD: r = a*b + c (single rounding, fused operation)
! Hardware FMAD: one multiply + one add, one rounding at end
r = a * b + c  ! Compiler emits FMAD instruction
```

### CUDA
```cuda
// fma(): CUDA's FMAD equivalent
// r = a*b + c, with single rounding at the end
// Semantics: rn (round-to-nearest)

float norm_i = 0.0f;
for (int k = 0; k < dim; k++) {
    float x = emb_i[k];
    norm_i = fma(x, x, norm_i);  // norm_i += x*x  (FMA, single rounding)
}
norm_i = sqrtf(norm_i);

// Dot product via FMA
float dot = 0.0f;
for (int k = 0; k < dim; k++) {
    dot = fma(emb_i[k], emb_j[k], dot);  // dot += vi[k] * vj[k]
}
```

**Why it maps:**
- Fortran FMAD has been in hardware since the 1990s (MIPS R10000, HP PA-8000)
- CUDA's `-use_fast_math` flag enables FMAD for floating-point operations
- FMAD: one instruction instead of two (MUL + ADD), one rounding step instead of two
- **Numerical difference:** `a*b + c` can round after MUL, then again after ADD. `fma(a,b,c)` rounds only once. This matters for accumulation over long loops (like dot products).
- Fortran compilers emit FMAD when optimization is on. CUDA exposes it explicitly via `fma()`.

---

## 7. No Heap Allocation = Fortran's Automatic Arrays Are Forbidden in Hot Loops

### Fortran (WRONG way)
```fortran
! DON'T DO THIS: ALLOCATABLE inside a hot loop
DO i = 1, n_agents
  ALLOCATE(tmp(dim))  ! Heap allocation per iteration = slow
  tmp = embeddings(i,:)
  ! ...
  DEALLOCATE(tmp)
END DO
```

### Fortran (RIGHT way)
```fortran
! Pre-allocate once, reuse across iterations
REAL :: tmp(MAX_DIM)  ! Static or local with known size
DO i = 1, n_agents
  tmp(1:dim) = embeddings(i,:)  ! No allocation in loop
END DO
```

### CUDA
```cuda
// WRONG: malloc in kernel (global memory allocation, 400+ cycles)
__global__ void wrong_kernel(...) {
    float* tmp = (float*)malloc(dim * sizeof(float));  // NEVER do this
    // ...
    free(tmp);
}

// RIGHT: pre-allocated buffers, passed via function arguments
__global__ void right_kernel(
    float* dist_matrix,    // Pre-allocated [n*n]
    const float* embeddings, // Pre-allocated [n*dim]
    int n_agents, int dim)
{
    // All memory pre-allocated. No malloc in kernel.
    // Registers for temporary values only.
    float row_sum = 0.0f;  // Register variable
    // ...
}
```

**Why it maps:**
- Fortran's rule "never ALLOCATE inside a hot loop" is CUDA's rule "never malloc in a kernel"
- Both are heap allocations: slow, can fail, introduce non-determinism
- CUDA heap allocation (`malloc`/`free` in kernel) is even worse than Fortran's because it runs on the GPU's memory allocator (sequential, high latency)
- Solution for both: allocate once at the top level, pass buffers as arguments

**Fortran philosophy made explicit:**
```
Fortran:  LOCAL arrays are stack-allocated, fast.
          ALLOCATABLE arrays are heap-allocated, slower.
          Don't allocate inside tight loops.

CUDA:     Registers are fast. Global memory is slow.
          Don't malloc inside kernels.
          Pre-allocate, reuse.
```

---

## 8. Atomic Operations = Fortran Critical Sections

### Fortran
```fortran
! Critical section: only one thread/process can execute at a time
!$OMP CRITICAL(max_update)
  IF (new_max > current_max) THEN
    current_max = new_max
  END IF
!$OMP END CRITICAL
```

### CUDA
```cuda
// Atomic operation: only one warp can execute the atomic at a time
// Equivalent to a critical section, but hardware-supported (no explicit lock)

__device__ __forceinline__ float atomicMaxFloat(float* addr, float val) {
    unsigned int* base = (unsigned int*)addr;
    unsigned int old = *base, assumed;
    do {
        assumed = old;
        old = atomicCAS(base, assumed,
            __float_as_uint(fmaxf(val, __uint_as_float(assumed))));
    } while (assumed != old);
    return __uint_as_float(old);
}

// Usage: track maximum distance across all threads
atomicMaxFloat(&global_max, my_dist);
```

**Why it maps:**
- Fortran CRITICAL ensures serialized access to shared variables
- CUDA atomics serialize access to a memory location (but without software locks — hardware CAS)
- CRITICAL has overhead (OS lock, context switch). CUDA atomics are hardware operations.
- Both are "escape hatches" — use only when necessary because they serialize part of your parallelism.

---

## 9. Thread Synchronization = Fortran Barrier Points

### Fortran (OpenMP)
```fortran
!$OMP PARALLEL DO
DO i = 1, n
  local_result(i) = compute_something(i)
END DO
!$OMP BARRIER  ! All threads must reach here before any proceed
!$OMP PARALLEL DO
DO i = 1, n
  final_result(i) = combine(local_result(i))
END DO
```

### CUDA
```cuda
__global__ void two_phase_kernel(...) {
    int i = blockIdx.x;
    
    // ---- Phase 1: per-thread computation ----
    float local = compute_phase1(i);
    shared_local[threadIdx.x] = local;
    __syncthreads();  // CUDA BARRIER equivalent
    
    // ---- Phase 2: block-level reduction ----
    // Only proceeds after ALL threads in block reach here
    if (threadIdx.x == 0) {
        float block_total = 0.0f;
        for (int t = 0; t < blockDim.x; t++) {
            block_total += shared_local[t];
        }
        // Write block result
    }
}
```

**Why it maps:**
- `!$OMP BARRIER` synchronizes all threads in a team
- `__syncthreads()` synchronizes all threads in a block
- Fortran threads in a team are cooperative (like CUDA threads in a block)
- The difference: CUDA block barrier is per-block (hardware-scoped), Fortran OMP barrier is per-team (software-scoped)

**Fortran got right:** Explicit barrier points prevent data races. CUDA makes you insert `__syncthreads()` explicitly — same principle, more explicit.

---

## 10. No Virtual Dispatch = Fortran's No Type Dispatch Problem

### Fortran (no OOP)
```fortran
! Pure procedural Fortran: all routines resolved at compile time
! No runtime type resolution, no vtable lookup
SUBROUTINE compute_cosine_distance(embeddings, dist_matrix, n, dim)
  REAL :: embeddings(dim, n)
  REAL :: dist_matrix(n, n)
  ! Fully determined at compile time. No branch misprediction from vtables.
END SUBROUTINE
```

### CUDA
```cuda
// CUDA has NO classes, NO virtual functions in device code
// (You can technically use C++ virtual in device code but it's terrible for performance)
// No vtable, no dynamic dispatch, no runtime polymorphism

// Instead, explicit conditional logic (like Fortran IF statements)
if (n_agents < 32) {
    // Small-path optimization
    for (int j = 0; j < n_agents; j++) { ... }
} else {
    // Large-path (vectorized across warps)
    for (int j = 0; j < n_agents; j += 32) { ... }
}
```

**Why it maps:**
- Fortran's strength: compile-time resolution, no runtime overhead
- CUDA's constraint: device code can't use virtual functions (vtable lookup = divergent warps = slow)
- Both require explicit branching for different code paths (no polymorphic hiding)
- The fix: Fortran uses `SELECT CASE` or `IF` blocks. CUDA uses `if` statements evaluated per-thread.

**Performance implication:** Virtual dispatch in CUDA causes warp divergence — some threads take one branch, others take another. The hardware then executes both paths sequentially, halving effective parallelism. Fortran subroutines avoid this by being statically resolved.

---

## Summary Table

| Fortran Concept | CUDA Equivalent | Fortran Philosophy Mapping |
|----------------|-----------------|---------------------------|
| DO loop (per-item) | CUDA thread (grid) | Each "job" = one thread, all run simultaneously |
| DO loop (vectorized) | CUDA warp (32 threads) | 32 items processed in lockstep SIMD fashion |
| COMMON block | Global memory (`cudaMalloc`) | Shared data across all workers, static lifetime |
| LOCAL array | Shared memory (`__shared__`) | Fast, per-block, survives barriers |
| Local scalar register | CUDA register variable | Zero-overhead storage for temporaries |
| FMAD instruction | `fma()` | One rounding, not two (better accuracy + speed) |
| `!$OMP BARRIER` | `__syncthreads()` | Explicit synchronization point |
| `!$OMP CRITICAL` | `atomic*()` | Serialize access to shared variable |
| ALLOCATABLE (in loop) | `malloc()` (in kernel) | NEVER DO THIS — heap allocation is slow |
| Subroutine (static dispatch) | Plain function | No virtual, no vtable, no runtime overhead |
| IF inside hot loop | `if` statement in kernel | Explicit branching (watch warp divergence) |

---

## The Deeper Insight

Fortran and CUDA share the same worldview: **the programmer knows more than the compiler about the data and the algorithm.** Fortran was designed for scientists who would hand-tune inner loops. CUDA was designed by hardware engineers who understood that giving programmers explicit control over memory hierarchy and thread coordination produces faster code than a black-box compiler.

The Fortran philosophy on the GPU means:
1. **Pre-allocate everything.** No heap in hot paths.
2. **Use registers for temporaries.** Shared memory for reusable data across threads.
3. **FMA for accumulation.** Better accuracy, better throughput.
4. **Explicit sync points.** Don't let the compiler insert barriers it doesn't know about.
5. **No virtual dispatch.** Static calls, explicit conditionals.
6. **Think in terms of "jobs" (threads) not "workers" (threads).** Each thread does one row, not one dot product.

Fortran programmers already think this way. The only new piece is the SIMD width (32 vs 4/8/16) and the memory hierarchy (registers → shared → global). Once you internalize those, CUDA looks like Fortran with vector extensions and explicit NUMA awareness.

---

*End of comparison. For the kernels themselves, see the source files in this directory.*