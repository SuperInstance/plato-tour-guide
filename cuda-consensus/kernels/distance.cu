/**
 * distance.cu — CUDA kernel for batched cosine distance computation.
 *
 * Computes pairwise cosine distances across an embedding matrix:
 *   d(i,j) = 1 - dot(vi, vj) / (|vi| * |vj|)
 *
 * Each thread handles one (i,j) pair — high parallelism for large swarms.
 */

/**
 * @brief Cosine distance kernel — 2D grid, one (i,j) pair per thread.
 *
 * @param embeddings  Flattened embedding matrix [n, dim] row-major
 * @param n           Number of embeddings
 * @param dim         Embedding dimension
 * @param output      Output distance matrix [n, n] row-major
 */
extern "C" __global__ void cosine_distance_kernel(
    const float* __restrict__ embeddings,
    int n,
    int dim,
    float* __restrict__ output
) {
    // 2D indexing
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;

    // Compute dot product + squared norms in a single pass
    float dot = 0.0f;
    float sq_norm_i = 0.0f;
    float sq_norm_j = 0.0f;

    const float* row_i = &embeddings[i * dim];
    const float* row_j = &embeddings[j * dim];

#pragma unroll 4
    for (int k = 0; k < dim; k++) {
        float vi = row_i[k];
        float vj = row_j[k];
        dot += vi * vj;
        sq_norm_i += vi * vi;
        sq_norm_j += vj * vj;
    }

    float norm_i = sqrtf(sq_norm_i);
    float norm_j = sqrtf(sq_norm_j);
    float denom = max(norm_i * norm_j, 1e-8f);

    output[i * n + j] = 1.0f - dot / denom;
}

/**
 * @brief Fused cosine distance + norm precompute kernel.
 *
 * Pre-computes norms in shared memory for reuse across all j for a given i.
 * Reduces redundant sqrt operations when computing many pairs with the same row.
 *
 * @param embeddings  Flattened embedding matrix [n, dim] row-major
 * @param n           Number of embeddings
 * @param dim         Embedding dimension
 * @param norms       Output pre-computed norms [n]
 * @param output      Output distance matrix [n, n] row-major
 */
extern "C" __global__ void cosine_distance_fused_kernel(
    const float* __restrict__ embeddings,
    int n,
    int dim,
    float* __restrict__ norms,
    float* __restrict__ output
) {
    // Shared memory for row norms — up to 256 agents
    extern __shared__ float shared_norms[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;

    // Compute norm for row i and cache in shared memory
    float sq_norm_i = 0.0f;
    const float* row_i = &embeddings[i * dim];
    for (int k = 0; k < dim; k++) {
        float vi = row_i[k];
        sq_norm_i += vi * vi;
    }
    float norm_i = sqrtf(sq_norm_i + 1e-12f);
    shared_norms[threadIdx.x] = norm_i;
    __syncthreads();

    // Compute norm for row j and distance
    float sq_norm_j = 0.0f;
    float dot = 0.0f;
    const float* row_j = &embeddings[j * dim];
    for (int k = 0; k < dim; k++) {
        float vi = row_i[k];
        float vj = row_j[k];
        dot += vi * vj;
        sq_norm_j += vj * vj;
    }
    float norm_j = sqrtf(sq_norm_j + 1e-12f);

    float denom = max(shared_norms[threadIdx.x] * norm_j, 1e-8f);
    output[i * n + j] = 1.0f - dot / denom;
}

/**
 * @brief Batch distance update kernel — efficient for incremental swarms.
 *
 * Given new embeddings appended to an existing matrix, compute distances
 * between new rows and ALL rows (including other new ones).
 *
 * @param all_embeddings  Full embedding matrix [total, dim]
 * @param total           Total embeddings (old + new)
 * @param old_count       Number of existing embeddings
 * @param dim             Embedding dimension
 * @param output          Output distance matrix [total, total]
 */
extern "C" __global__ void batch_update_kernel(
    const float* __restrict__ all_embeddings,
    int total,
    int old_count,
    int dim,
    float* __restrict__ output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Only compute for new rows (i >= old_count) or when j is new
    if (i >= total || j >= total) return;
    if (i < old_count && j < old_count) return; // already computed

    float dot = 0.0f, sq_i = 0.0f, sq_j = 0.0f;
    const float* row_i = &all_embeddings[i * dim];
    const float* row_j = &all_embeddings[j * dim];

    for (int k = 0; k < dim; k++) {
        float vi = row_i[k];
        float vj = row_j[k];
        dot += vi * vj;
        sq_i += vi * vi;
        sq_j += vj * vj;
    }

    float denom = max(sqrtf(sq_i) * sqrtf(sq_j), 1e-8f);
    output[i * total + j] = 1.0f - dot / denom;
}
