/*
 * cosine_distance_kernel.cu
 * Pure CUDA C kernel — Fortran philosophy: no classes, no templates, pure arithmetic.
 *
 * Each thread computes one row of the pairwise cosine distance matrix.
 * CUDA thread j = Fortran job j (one row).
 * Register pressure managed explicitly — no dynamic dispatch.
 */

#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Device constants — set at kernel launch via cudaLaunchKernel
// ---------------------------------------------------------------------------
__device__ int g_n_agents = 0;
__device__ int g_dim = 0;
__device__ float g_threshold = 0.0f;

// ---------------------------------------------------------------------------
// cosine_distance_kernel
// ---------------------------------------------------------------------------
// Grid: (n_agents, 1, 1)
// Block: (256, 1, 1) — 256 threads per row (over-provisioned for small dim)
// Every thread in the block cooperates on one row.
//
// Registers per thread (Fortran-style allocation):
//   norm_i, dot_prod, cosine_sim, distance
//
// Fused multiply-add via __fma_rn() — Fortran FMA equivalent.
// Pre-allocated dist_matrix[n_agents][n_agents] written directly.
// No malloc in kernel. No shared memory (register-only compute).
// ---------------------------------------------------------------------------
__global__ void cosine_distance_kernel(
    float* __restrict__ dist_matrix,   // [n_agents * n_agents] output
    const float* __restrict__ embeddings, // [n_agents * dim] input
    int n_agents,
    int dim)
{
    // Fortran equivalent: implicitly n_agents and dim via COMMON block
    // Here we load them from globals (set by cudaLaunchKernel config)
    (void)g_n_agents;  // silence unused warning — available for extension
    (void)g_dim;

    // Global row index — one thread block per row
    int i = blockIdx.x;

    if (i >= n_agents) return;

    const float* emb_i = embeddings + i * dim;

    // ---- Phase 1: compute ||v_i|| ----
    float norm_i = 0.0f;
    for (int k = 0; k < dim; k++) {
        float x = emb_i[k];
        norm_i = fma(x, x, norm_i);   // norm_i += x*x  (FMA)
    }
    norm_i = sqrtf(norm_i);

    // Guard against zero vector
    if (norm_i == 0.0f) {
        for (int j = 0; j < n_agents; j++) {
            dist_matrix[i * n_agents + j] = 1.0f;  // max distance
        }
        return;
    }

    // ---- Phase 2: for each j, compute 1 - cosine_similarity(i,j) ----
    // All threads in block cooperatively compute one row.
    // Stride = blockDim.x so each thread handles a subset of columns.
    float local_dot  = 0.0f;
    float local_norm = 0.0f;
    int   col_start  = threadIdx.x;
    int   col_stride = blockDim.x;

    for (int k = col_start; k < dim; k += col_stride) {
        float x = emb_i[k];
        local_dot = fma(x, x, local_dot);  // accumulate for norm check (same as norm_i)
    }

    // Warp-level reduction to sum local_dot contributions
    // Uses __shfl_xor_sync for warp shuffle reduction (Fortran vector-unit analogy)
    unsigned int mask = 0xFFFFFFFFu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dot = fma(__shfl_xor_sync(mask, local_dot, offset), 1.0f, local_dot);
    }
    // __shfl result only valid in lane 0 — broadcast
    if (threadIdx.x == 0) {
        local_norm = local_dot;
    }
    local_norm = __shfl_sync(mask, local_norm, 0);

    // Actual norm_i already computed in Phase 1, but we recompute per-warp
    // for cache efficiency. Real impl would store norm_i in shared.
    // Here we use the full norm_i computed above.

    // ---- Main column loop ----
    for (int j = col_start; j < n_agents; j += col_stride) {
        if (j == i) {
            dist_matrix[i * n_agents + j] = 0.0f;
            continue;
        }

        // Compute dot product v_i · v_j
        const float* emb_j = embeddings + j * dim;
        float dot_prod = 0.0f;

        for (int k = 0; k < dim; k++) {
            dot_prod = fma(emb_i[k], emb_j[k], dot_prod);  // FMA accumulate
        }

        // Compute ||v_j|| (could be precomputed and passed in — Fortran style would cache)
        float norm_j = 0.0f;
        for (int k = 0; k < dim; k++) {
            float x = emb_j[k];
            norm_j = fma(x, x, norm_j);
        }
        norm_j = sqrtf(norm_j);

        if (norm_j == 0.0f) {
            dist_matrix[i * n_agents + j] = 1.0f;
            continue;
        }

        // Cosine similarity = dot / (||vi|| * ||vj||)
        float cos_sim = fmaxf(-1.0f, fminf(1.0f, dot_prod / (norm_i * norm_j)));
        float distance = 1.0f - cos_sim;  // cosine distance

        dist_matrix[i * n_agents + j] = distance;
    }
}

// ---------------------------------------------------------------------------
// Host wrapper (Fortran-style: explicit interface, no classes)
// ---------------------------------------------------------------------------
extern "C" void cosine_distance_launch(
    float* dist_matrix,
    const float* embeddings,
    int n_agents,
    int dim,
    cudaStream_t stream)
{
    int block_threads = 256;
    dim3 block(block_threads, 1, 1);
    dim3 grid(n_agents, 1, 1);

    cosine_distance_kernel<<<grid, block, 0, stream>>>(
        dist_matrix, embeddings, n_agents, dim);
}

// ---------------------------------------------------------------------------
// CPU fallback — Fortran philosophy: same signature, pure arithmetic
// ---------------------------------------------------------------------------
extern "C" void cosine_distance_cpu(
    float* dist_matrix,
    const float* embeddings,
    int n_agents,
    int dim)
{
    for (int i = 0; i < n_agents; i++) {
        const float* emb_i = embeddings + i * dim;

        // ||vi||
        float norm_i = 0.0f;
        for (int k = 0; k < dim; k++) {
            float x = emb_i[k];
            norm_i += x * x;
        }
        norm_i = sqrtf(norm_i);
        if (norm_i == 0.0f) norm_i = 1.0f;

        for (int j = 0; j < n_agents; j++) {
            if (j == i) {
                dist_matrix[i * n_agents + j] = 0.0f;
                continue;
            }
            const float* emb_j = embeddings + j * dim;

            // v_i · v_j
            float dot = 0.0f;
            for (int k = 0; k < dim; k++) {
                dot += emb_i[k] * emb_j[k];
            }

            // ||v_j||
            float norm_j = 0.0f;
            for (int k = 0; k < dim; k++) {
                float x = emb_j[k];
                norm_j += x * x;
            }
            norm_j = sqrtf(norm_j);
            if (norm_j == 0.0f) norm_j = 1.0f;

            float cos_sim = dot / (norm_i * norm_j);
            // Clamp for numerical safety
            if (cos_sim >  1.0f) cos_sim =  1.0f;
            if (cos_sim < -1.0f) cos_sim = -1.0f;

            dist_matrix[i * n_agents + j] = 1.0f - cos_sim;
        }
    }
}