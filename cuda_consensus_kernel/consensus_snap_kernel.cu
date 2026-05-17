/*
 * consensus_snap_kernel.cu
 *
 * Final snap decision on GPU: find the medoid (agent closest to all others).
 * Fortran philosophy: no classes, no templates, pure arithmetic.
 *
 * Shared memory for clique search      → Fortran shared/COMMON memory
 * Threshold-based pruning             → Fortran WHERE mask
 * Write to global memory on exit      → Fortran output through COMMON block
 */

#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Warp-level min-finding via shuffle
// Returns (value, lane_id) of the minimum across the warp
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_min(float val, int lane, unsigned int mask)
{
    // Iteratively compare with neighbor lanes (XOR pattern)
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, val, offset);
        int   other_lane = __shfl_xor_sync(mask, lane, offset);
        if (other < val || (other == val && other_lane < lane)) {
            val = other;
            lane = other_lane;
        }
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block-level reduction for min-finding across warps
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_min(float val, int lane, unsigned int* min_lane)
{
    __shared__ float shared[32];
    __shared__ int   shared_lane[32];

    // Warp 0 result
    unsigned int mask = 0xFFFFFFFFu;
    float warp_min = warp_reduce_min(val, lane, mask);

    if (lane == 0) {
        shared[0] = warp_min;
        shared_lane[0] = 0;  // relative to block start
    }
    __syncthreads();

    // Final reduction in warp 0
    if (lane == 0) {
        float my_val = shared[0];
        int my_lane = 0;
        for (int i = 1; i < blockDim.x / 32; i++) {
            if (shared[i] < my_val) {
                my_val = shared[i];
                my_lane = shared_lane[i];
            }
        }
        *min_lane = blockIdx.x * blockDim.x + my_lane;  // global lane index
        shared[0] = my_val;
    }
    __syncthreads();

    return shared[0];
}

// ---------------------------------------------------------------------------
// medoid_find_kernel
// ---------------------------------------------------------------------------
// Finds medoid = agent i that minimizes sum_j dist(i, j)
// Uses shared memory for clique search (Fortran COMMON block equivalent).
//
// Grid: (1, 1, 1) — single block processes all agents
// Block: (256, 1, 1) — threads cooperatively scan candidate agents
//
// Algorithm:
//   - Each warp (32 threads) evaluates one candidate agent
//   - Threads in warp cooperatively sum that agent's row
//   - Warp-reduce to find best candidate
//   - Block-reduce to find global medoid
// ---------------------------------------------------------------------------
__global__ void medoid_find_kernel(
    const float* __restrict__ dist_matrix,
    int n_agents,
    float threshold,
    int*   medoid_out,        // [1] output: index of medoid agent
    float* medoid_score_out,  // [1] output: sum of distances for medoid
    int*   consensus_val_out  // [1] output: final consensus value (snap decision)
)
{
    extern __shared__ float shared_min_score[];
    extern __shared__ int   shared_min_lane[];

    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp_id = tid / 32;
    int warps_per_block = blockDim.x / 32;

    // ---- Each warp evaluates one candidate medoid ----
    int candidate = warp_id;  // one candidate per warp
    float candidate_score = 0.0f;

    if (candidate < n_agents) {
        // Sum row 'candidate' of distance matrix
        float row_sum = 0.0f;
        for (int j = lane; j < n_agents; j += 32) {
            float d = dist_matrix[candidate * n_agents + j];
            row_sum += d;
        }

        // Warp reduction to sum this row
        unsigned int mask = 0xFFFFFFFFu;
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_sum += __shfl_xor_sync(mask, row_sum, offset);
        }

        if (lane == 0) {
            candidate_score = row_sum;
        }
    }

    // ---- Block-level reduction: find warp with minimum score ----
    __shared__ float s_min_score[32];
    __shared__ int   s_min_warp[32];

    if (lane == 0) {
        s_min_score[warp_id] = candidate_score;
        s_min_warp[warp_id] = candidate;
    }
    __syncthreads();

    if (tid == 0) {
        float best_score = 3.40282347e+38f;  // FLT_MAX
        int   best_candidate = 0;

        for (int w = 0; w < warps_per_block; w++) {
            if (s_min_score[w] < best_score) {
                best_score = s_min_score[w];
                best_candidate = s_min_warp[w];
            }
        }

        // ---- Threshold-based pruning (Fortran WHERE equivalent) ----
        int consensus_val = 0;
        if (best_score < threshold * (float)n_agents) {
            consensus_val = 1;  // snap to medoid
        }

        *medoid_out = best_candidate;
        *medoid_score_out = best_score;
        *consensus_val_out = consensus_val;
    }
}

// ---------------------------------------------------------------------------
// clique_kernel
// ---------------------------------------------------------------------------
// Finds the maximal clique of agents where ALL pairwise distances < threshold.
// Uses shared memory adjacency matrix (Fortran COMMON block).
// Bron–Kerbosch algorithm (simplified for GPU).
//
// Grid: (1, 1, 1)
// Block: (n_agents, 1, 1)
//
// Algorithm (Fortran-style):
//   - Build adjacency matrix in shared memory: A[i][j] = 1 if dist < threshold
//   - Find maximal clique using recursive expansion
//   - Return clique size and representative (median of clique members)
// ---------------------------------------------------------------------------
__global__ void clique_kernel(
    const float* __restrict__ dist_matrix,
    int n_agents,
    float threshold,
    int*   clique_size_out,   // [1] size of found clique
    int*   clique_rep_out,    // [1] representative (median member)
    float* clique_spread_out // [1] spread of clique
)
{
    extern __shared__ int adj_matrix[];  // [n_agents * n_agents] int

    int tid = threadIdx.x;
    int i = tid;

    // ---- Phase 1: Build adjacency matrix (shared memory = Fortran COMMON) ----
    if (i < n_agents) {
        for (int j = 0; j < n_agents; j++) {
            float d = dist_matrix[i * n_agents + j];
            adj_matrix[i * n_agents + j] = (d < threshold && d != 0.0f) ? 1 : 0;
        }
    }
    __syncthreads();

    // ---- Phase 2: Naive maximal clique search ----
    // Fortran style: explicit nested loops, no recursion (GPU recursion bad)
    // Simplified: find largest all-close subgraph via greedy expansion
    //
    // Real Bron–Kerbosch would be: BK(R, P, X):
    //   if P is empty and X is empty: report R as maximal clique
    //   for each v in P: BK(R ∪ {v}, P ∩ N(v), X ∩ N(v)); P := P \ {v}
    // GPU impl: unroll to fixed depth, use shared stack

    __shared__ int clique_members[64];  // max clique size 64
    __shared__ int current_clique_size;

    if (tid == 0) {
        current_clique_size = 0;
    }
    __syncthreads();

    // Greedy clique expansion (each thread proposes one clique, pick max)
    // For real impl, would use atomic compare-and-swap for clique building
    // Here: simple greedy — start with agent 0, add neighbors that are close to all
    if (tid == 0) {
        int size = 1;
        clique_members[0] = 0;

        for (int candidate = 1; candidate < n_agents && size < 64; candidate++) {
            int can_join = 1;
            for (int m = 0; m < size; m++) {
                int member = clique_members[m];
                if (adj_matrix[candidate * n_agents + member] == 0) {
                    can_join = 0;
                    break;
                }
            }
            if (can_join) {
                clique_members[size] = candidate;
                size++;
            }
        }

        current_clique_size = size;
        *clique_size_out = size;

        // Representative = median member (by index)
        int rep = clique_members[size / 2];
        *clique_rep_out = rep;

        // Compute clique spread
        float spread_sum = 0.0f;
        int pair_count = 0;
        for (int a = 0; a < size; a++) {
            for (int b = a + 1; b < size; b++) {
                int i_a = clique_members[a];
                int i_b = clique_members[b];
                spread_sum += dist_matrix[i_a * n_agents + i_b];
                pair_count++;
            }
        }
        *clique_spread_out = (pair_count > 0) ? (spread_sum / (float)pair_count) : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------
extern "C" void consensus_snap_launch(
    const float* dist_matrix,
    int n_agents,
    float threshold,
    int*   medoid_out,
    float* medoid_score_out,
    int*   consensus_val_out,
    cudaStream_t stream)
{
    medoid_find_kernel<<<1, 256, 256 * sizeof(float), stream>>>(
        dist_matrix, n_agents, threshold, medoid_out, medoid_score_out, consensus_val_out);
}

extern "C" void clique_search_launch(
    const float* dist_matrix,
    int n_agents,
    float threshold,
    int*   clique_size_out,
    int*   clique_rep_out,
    float* clique_spread_out,
    cudaStream_t stream)
{
    int shared_size = n_agents * n_agents * sizeof(int);
    clique_kernel<<<1, n_agents, shared_size, stream>>>(
        dist_matrix, n_agents, threshold, clique_size_out, clique_rep_out, clique_spread_out);
}