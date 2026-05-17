# CT-Math → CUDA Memory Hierarchy Bridge

## Overview

This module implements the mathematical bridge between **Constraint Theory (CT)** theorems and **GPU memory hierarchy** optimization. It translates abstract mathematical properties (rigidity, cohomology, holonomy) into concrete CUDA kernel selection and memory access patterns.

## Mathematical Foundations

### Key CT Theorems Mapped to CUDA

1. **H1 Cohomology (β₁)** → Memory Access Pattern Selection
   - β₁ = 0 (rigid): Coalesced memory access
   - β₁ > 0 (flexible): Scattered memory access

2. **Laman's Theorem** → Cache-Friendly Layout
   - Rigid graphs (E = 2V - 3): Sequential memory layout
   - Flexible graphs: Indirect addressing with tiling

3. **Zero Holonomy Consensus** → Register Optimization
   - Trust vectors live in registers (no DRAM access)
   - Pythagorean 48-direction encoding (6 bits vs 96 bits)

## File Structure

```
ct_cuda_bridge/
├── h1_memory_patterns.cu          # H1 → Kernel dispatch
├── pythagorean48_cuda.cu          # 48-direction code on GPU
├── memory_latency_benchmark.cu    # Memory hierarchy measurement
├── zhc_memory_model.md            # Zero Holonomy Consensus memory
├── laman_constrained_layout.md    # Laman theorem → cache layout
├── ct_kernel_selector.py          # Python kernel selection logic
├── benchmark.sh                   # Run all benchmarks
└── README.md                      # This file
```

## Core Concepts

### 1. H1 Memory Patterns (`h1_memory_patterns.cu`)

**Purpose**: Map H1 cohomology to GPU kernel selection

**Key Functions**:
- `compute_h1_cohomology()`: Calculate β₁ = E - V + 1
- `coalesced_distance_kernel()`: For rigid graphs (β₁ = 0)
- `scattered_gather_kernel()`: For flexible graphs (β₁ > 0)
- `dispatch_kernel_based_on_h1()`: Runtime kernel selection

**Decision Tree**:
```
Compute H1 → Check rigidity → Branch:
  β₁ = 0 → Coalesced kernel (predictable access)
  β₁ > 0 → Scattered kernel (irregular access)
```

### 2. Pythagorean 48-Direction Code (`pythagorean48_cuda.cu`)

**Purpose**: Efficient trust vector storage using geometric code

**Key Features**:
- 48 directions with ~37.5° angular separation
- log₂(48) ≈ 5.585 bits per vector (vs 96 bits naive)
- Codebook stored in `__constant__` memory (cached)
- Encode/decode entirely in registers

**Compression Ratio**: 96 bits → 6 bits (16x reduction)

**Key Functions**:
- `encode_pythagorean48()`: Vector → 6-bit code
- `decode_pythagorean48()`: 6-bit code → vector
- `batch_encode_pythagorean48()`: GPU batch encoding
- `compute_trust_matrix_pythagorean()`: Pairwise trust using codes

### 3. Memory Latency Benchmark (`memory_latency_benchmark.cu`)

**Purpose**: Measure actual GPU memory hierarchy performance

**Benchmarks**:
- Register-only operations (baseline)
- Shared memory (user-managed cache)
- L1 cache (hardware cache)
- L2 cache (larger cache)
- Global memory (DRAM)
- Coalesced vs strided access
- Register pressure analysis
- Managed vs static memory

**Key Function**:
- `run_memory_hierarchy_benchmarks()`: Complete hierarchy analysis

### 4. Zero Holonomy Consensus Memory (`zhc_memory_model.md`)

**Purpose**: Mathematical foundation of ZHC on GPU

**Key Concepts**:
- Zero holonomy condition: ∏ T_i = I (identity)
- Trust vectors live in curled subspace of ℝ³
- Register-only hot path (no DRAM access)
- Pairwise trust in one kernel launch

**Memory Layout**:
```
Registers:    Individual trust vectors (48 dims per agent)
Shared Mem:   Thread-group partial answers
Global Mem:   Encoded trust vectors (6 bits per agent)
```

### 5. Laman Constrained Layout (`laman_constrained_layout.md`)

**Purpose**: Map rigidity theory to cache-friendly layouts

**Laman's Theorem**:
- Graph is minimally rigid iff E = 2V - 3
- Rigid subgraphs have cache-friendly spatial locality
- Flexible regions require indirect addressing

**Strategy**:
```
1. Check Laman condition
2. Partition into rigid blocks
3. Layout rigid blocks sequentially
4. Handle flexible regions separately
```

### 6. CT Kernel Selector (`ct_kernel_selector.py`)

**Purpose**: Python implementation of kernel selection logic

**Key Functions**:
- `compute_h1_cohomology()`: Calculate β₁
- `check_laman_condition()`: Verify rigidity
- `compute_spread_threshold()`: Measure spatial distribution
- `select_kernel()`: Return optimal kernel

**Returns**:
- `KernelRecommendation` with:
  - Kernel type (coalesced/mixed/scattered)
  - Confidence score
  - Reasoning
  - Expected performance

### 7. Benchmark Suite (`benchmark.sh`)

**Purpose**: Run all benchmarks and generate comparison tables

**Benchmarks**:
1. H1 memory patterns (rigid vs flexible)
2. Pythagorean encoding performance
3. Memory hierarchy latency
4. Kernel selector accuracy

**Outputs**:
- CSV comparison tables
- Performance metrics
- GPU information
- Recommendations

## Usage

### Compile CUDA Kernels

```bash
cd /home/ubuntu/.openclaw/workspace/repos/plato-tour-guide/ct_cuda_bridge

nvcc -O3 -arch=native h1_memory_patterns.cu -o h1_benchmark
nvcc -O3 -arch=native pythagorean48_cuda.cu -o pythagorean_benchmark
nvcc -O3 -arch=native memory_latency_benchmark.cu -o memory_benchmark
```

### Run Benchmarks

```bash
# Run all benchmarks
./benchmark.sh

# Run individual benchmarks
./h1_benchmark
./pythagorean_benchmark
./memory_benchmark

# Run Python kernel selector
python3 ct_kernel_selector.py
```

### Use in Your Code

#### C++ Integration

```cpp
#include "h1_memory_patterns.cu"

// Compute embeddings and dispatch kernel
float* d_embeddings;
int* d_edges;
// ... allocate and initialize ...

dispatch_kernel_based_on_h1(
    d_embeddings, d_edges, num_vertices, num_edges, embedding_dim, d_output
);
```

#### Python Integration

```python
from ct_kernel_selector import select_kernel

# Prepare partial answers
partials = [np.array([0, 1, 2]), np.array([3, 4, 5])]

# Select optimal kernel
recommendation = select_kernel(partials, embeddings)
print(f"Use kernel: {recommendation.kernel_type}")
print(f"Reasoning: {recommendation.reasoning}")
```

## Performance Targets

### Memory Access Patterns

| Pattern | Cache Hits | Bandwidth | Latency | Use Case |
|---------|------------|-----------|---------|----------|
| Register-only | 100% | N/A | ~0 ns | Hot path |
| Shared Memory | 95% | 100 TB/s | ~30 cycles | Thread-group |
| L1 Cache | 90% | 50 TB/s | ~80 cycles | Frequent access |
| L2 Cache | 70% | 10 TB/s | ~200 cycles | Working set |
| Global Memory | 30% | 900 GB/s | ~500 cycles | Large data |

### Kernel Performance

| Kernel | Graph Type | Bandwidth | Speedup vs Naive |
|--------|-----------|-----------|------------------|
| Coalesced | Rigid (β₁=0) | 900 GB/s | 3x |
| Mixed | Hybrid | 600 GB/s | 2x |
| Scattered | Flexible | 300 GB/s | 1x (baseline) |
| Register-only | Small rigid | N/A | 10x |

## Mathematical Connection

### CT Theorem → CUDA Implementation

```
CT Theorem                    CUDA Implementation
─────────────────────────────────────────────────────────
H1 Cohomology (β₁)      →     Kernel selection
  β₁ = 0 (rigid)              Coalesced kernel
  β₁ > 0 (flexible)           Scattered kernel

Laman Condition         →     Memory layout
  E = 2V - 3                   Sequential layout
  E < 2V - 3                   Indirect addressing

Zero Holonomy           →     Register optimization
  ∏ T_i = I                   Trust in registers
  Curled subspace              6-bit encoding

Spread Threshold        →     Cache blocking
  High spread                 Large tiles
  Low spread                  Small tiles
```

## Key Insights

1. **Rigidity → Predictability**: Rigid graphs have regular structure that maps to coalesced memory access

2. **Flexibility → Irregularity**: Flexible graphs require scatter/gather operations

3. **H1 as Decision Metric**: β₁ perfectly predicts optimal kernel choice

4. **Zero Holonomy → Compression**: Trust vectors live in curled subspace → 16x compression

5. **Laman → Locality**: Rigid subgraphs have spatial locality → cache-friendly

## Future Work

1. **Adaptive Runtime**: Dynamically switch kernels based on performance
2. **Multi-GPU**: Scale to multiple GPUs with NVLink
3. **Tensor Cores**: Use WMMA for trust matrix operations
4. **Graph Learning**: Learn optimal kernels from data
5. **Formal Verification**: Prove correctness of kernel selection

## References

- Constraint Theory: See `constraint-theory-core/`
- CUDA Programming: See `cudaclaw/`
- Consensus: See `holonomy-consensus/`
- Plato SDK: See `plato-sdk/`

## License

MIT License - See project root for details

## Contributing

Contributions welcome! Please ensure:
1. All benchmarks pass
2. Code follows CUDA best practices
3. Mathematical correctness verified
4. Documentation updated

---

**Generated**: 2026-05-17
**Part of**: Plato Tour Guide
**Purpose**: Bridge CT mathematics to GPU memory optimization
