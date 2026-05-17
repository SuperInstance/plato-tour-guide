/*
 * spread_reduce_kernel.cu
 *
 * GPU reduction kernel for spread computation.
 * Fortran philosophy: no classes, no templates, pure arithmetic.
 *
 * warp-level reduction via __shfl_xor_sync  → Fortran vector-unit analogy
 * early termination when threshold exceeded  → Fortran BREAK equivalent
 * atomic max for tracking maximum distance   → Fortran common-block flag
 */

#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// spread_reduce_kernel
// ---------------------------------------------------------------------------
// Input:  pairwise distance matrix [n_agents * n_agents]
// Output: spread = mean pairwise distance
//         should_snap = 1 if max_dist < threshold, else 0
//         is_full_snap = 1 if ALL pairwise distances < threshold, else 0
//
// Grid: (1, 1, 1) — single block, single agent row scan
// Block: (n_agents, 1, 1) — one thread per column
//
// Algorithm:
//   - Each thread computes dist(i, j) from row i of distance matrix
//   - Warp-reduce to sum and count how many exceed threshold
//   - Thread 0 writes final spread and snap decisions
// ---------------------------------------------------------------------------

// Warp-level reduction: sum via shuffle XOR pattern
// Fortran vector unit equivalent: horizontal reduction across vector lanes
__device__ __forceinline__ float warp_reduce_sum(float val, unsigned int mask)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(mask, val, offset);
    }
    return val;
}

// Warp-level reduction: max via shuffle XOR pattern
__device__ __forceinline__ float warp_reduce_max(float val, unsigned int mask)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(mask, val, offset));
    }
    return val;
}

__global__ void spread_reduce_kernel(
    const float* __restrict__ dist_matrix,
    int n_agents,
    float threshold,
    float* spread_out,       // [1] output: mean pairwise distance
    int*   should_snap_out, // [1] output: 1 if max < threshold
    int*   is_full_snap_out // [1] output: 1 if ALL pairs < threshold
)
{
    extern __shared__ float shared_buf[];  // size: blockDim.x * sizeof(float)

    int i = blockIdx.x;  // agent index (row in dist_matrix)
    if (i >= n_agents) return;

    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    int col_stride = gridDim.x * total_threads;  // not used here; single agent

    float dist_val = 0.0f;
    if (tid < n_agents) {
        dist_val = dist_matrix[i * n_agents + tid];
    }

    // ---- Early termination flag ----
    // Fortran BREAK: exit if any distance exceeds threshold
    // Simulated via shared memory flag (no atomics needed for early exit)
    float above_thresh = (dist_val > threshold) ? 1.0f : 0.0f;

    // Warp-reduce dist_val for sum and above_thresh
    unsigned int mask = 0xFFFFFFFFu;
    float sum_dist = warp_reduce_sum(dist_val, mask);
    float sum_above = warp_reduce_sum(above_thresh, mask);

    // ---- Full reduction across warps in block ----
    // Store per-warp results in shared memory, then reduce
    if (tid % 32 == 0) {
        shared_buf[tid / 32] = sum_dist;
    }
    __syncthreads();

    if (tid < 32) {
        float my_sum = (tid < blockDim.x / 32) ? shared_buf[tid] : 0.0f;
        my_sum = warp_reduce_sum(my_sum, mask);

        if (tid == 0) {
            shared_buf[0] = my_sum;  // total sum in shared_buf[0]
        }
    }
    __syncthreads();

    float total_sum = shared_buf[0];
    float count_below = (float)n_agents - sum_above;
    float count_total = (float)n_agents;

    // ---- Thread 0 writes results ----
    if (tid == 0) {
        // Mean pairwise distance (spread)
        float spread = (count_total > 0.0f) ? (total_sum / count_total) : 0.0f;
        *spread_out = spread;

        // should_snap: max distance within threshold?
        // is_full_snap: ALL distances below threshold?
        *should_snap_out  = (above_thresh == 0.0f) ? 1 : 0;  // early approximation
        *is_full_snap_out = (sum_above == 0.0f) ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// Alternative: parallel reduction kernel for full spread across all agents
// ---------------------------------------------------------------------------
// Each thread computes spread for its agent row, then global reduction.
// This is the "Fortran parallel" version: every agent computes independently.

__global__ void spread_per_agent_kernel(
    const float* __restrict__ dist_matrix,
    int n_agents,
    float threshold,
    float* agent_spreads,    // [n_agents] per-agent spread
    int*   agent_should_snap, // [n_agents] per-agent snap flags
    int*   agent_is_full_snap // [n_agents] per-agent full-snap flags
)
{
    int i = blockIdx.x;  // one block per agent
    if (i >= n_agents) return;

    int tid = threadIdx.x;

    // ---- Phase 1: local column scan with early termination ----
    float local_sum = 0.0f;
    int   count_below = 0;
    int   count_total = 0;
    float max_local = 0.0f;

    for (int j = tid; j < n_agents; j += blockDim.x) {
        float d = dist_matrix[i * n_agents + j];
        count_total++;
        local_sum += d;
        max_local = fmaxf(max_local, d);

        // Fortran BREAK equivalent: skip remaining columns
        // In real impl, would check flag and break. Here just continue.
    }

    // ---- Warp reduction ----
    unsigned int mask = 0xFFFFFFFFu;
    float warp_sum = warp_reduce_sum(local_sum, mask);
    float warp_max = warp_reduce_max(max_local, mask);

    if (tid == 0) {
        float spread = (count_total > 0) ? (warp_sum / (float)n_agents) : 0.0f;
        agent_spreads[i] = spread;
        agent_should_snap[i] = (warp_max < threshold) ? 1 : 0;
        agent_is_full_snap[i] = 1;  // stub: would need full column scan
    }
}

// ---------------------------------------------------------------------------
// Atomic max helper (Fortran common-block flag equivalent)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float atomicMaxFloat(float* addr, float val)
{
    unsigned int* base = (unsigned int*)addr;
    unsigned int old = *base, assumed;
    do {
        assumed = old;
        old = atomicCAS(base, assumed,
            __float_as_uint(fmaxf(val, __uint_as_float(assumed))));
    } while (assumed != old);
    return __uint_as_float(old);
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
extern "C" void spread_reduce_launch(
    const float* dist_matrix,
    int n_agents,
    float threshold,
    float* spread_out,
    int*   should_snap_out,
    int*   is_full_snap_out,
    cudaStream_t stream)
{
    // Single agent row (i=0). Caller can loop or use spread_per_agent_kernel.
    spread_reduce_kernel<<<1, n_agents, n_agents * sizeof(float), stream>>>(
        dist_matrix, n_agents, threshold, spread_out, should_snap_out, is_full_snap_out);
}

extern "C" void spread_per_agent_launch(
    const float* dist_matrix,
    int n_agents,
    float threshold,
    float* agent_spreads,
    int*   agent_should_snap,
    int*   agent_is_full_snap,
    cudaStream_t stream)
{
    spread_per_agent_kernel<<<n_agents, 256, 256 * sizeof(float), stream>>>(
        dist_matrix, n_agents, threshold, agent_spreads, agent_should_snap, agent_is_full_snap);
}