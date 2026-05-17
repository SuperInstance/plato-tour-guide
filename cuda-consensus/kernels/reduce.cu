/**
 * reduce.cu — CUDA kernels for spread / max reduction.
 *
 * Computes the maximum (spread) across a distance matrix using a
 * two-level tree reduction (shared memory + warp shuffle).
 */

/**
 * @brief Maximum reduction — single block tree reduction.
 *
 * @param input  Input array
 * @param n      Number of elements
 * @param output Scalar output — the maximum value
 */
extern "C" __global__ void max_reduce_kernel(
    const float* __restrict__ input,
    int n,
    float* __restrict__ output
) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    shared[tid] = (idx < n) ? input[idx] : -1.0f;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float a = shared[tid];
            float b = shared[tid + s];
            shared[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

/**
 * @brief Warp-shuffle max reduction — faster for large arrays.
 *
 * Uses __shfl_xor_sync for warp-level reduction before shared memory.
 * Requires compute capability 3.0+.
 *
 * @param input  Input array
 * @param n      Number of elements
 * @param output Partial results (one per block)
 */
extern "C" __global__ void warp_max_reduce_kernel(
    const float* __restrict__ input,
    int n,
    float* __restrict__ output
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load input (or sentinel -inf)
    float val = (gid < n) ? input[gid] : -INFINITY;

    // Warp-level shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, offset, 32);
        val = (val > other) ? val : other;
    }

    // Warp leaders write to shared memory
    if ((tid & 31) == 0) {
        shared[tid / 32] = val;
    }
    __syncthreads();

    // Final reduction in shared memory (first warp only)
    if (tid < blockDim.x / 32) {
        float s_val = shared[tid];
        for (int offset = (blockDim.x / 64); offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFF, s_val, offset, 32);
            s_val = (s_val > other) ? s_val : other;
        }
        if (tid == 0) {
            shared[0] = s_val;
        }
    }
    __syncthreads();

    // Output
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

/**
 * @brief 2D max reduction — directly reduces distance matrix without flattening.
 *
 * Each block processes a tile of the matrix and reduces to max.
 * More efficient than flatten + reduce for the distance matrix use case.
 *
 * @param matrix  2D distance matrix [rows, cols] row-major
 * @param rows    Number of rows
 * @param cols    Number of columns
 * @param output  Partial max values (one per block-row stripe)
 */
extern "C" __global__ void matrix_max_reduce_kernel(
    const float* __restrict__ matrix,
    int rows,
    int cols,
    float* __restrict__ output
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int row = blockIdx.x;

    if (row >= rows) return;

    // Each thread loads one column element
    float val = -INFINITY;
    if (tid < cols) {
        val = matrix[row * cols + tid];
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, offset, 32);
        val = (val > other) ? val : other;
    }

    // Warp leaders to shared memory
    if ((tid & 31) == 0) {
        shared[tid / 32] = val;
    }
    __syncthreads();

    // Final shared memory reduction
    int warps_per_block = blockDim.x / 32;
    if (tid < warps_per_block) {
        float s_val = shared[tid];
        for (int offset = warps_per_block / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                float a = shared[tid];
                float b = shared[tid + offset];
                shared[tid] = (a > b) ? a : b;
            }
            __syncthreads();
            s_val = shared[tid];
        }
        if (tid == 0) {
            output[row] = shared[0];
        }
    }
}

/**
 * @brief Early-termination spread kernel.
 *
 * Stops early if a pair with distance > threshold is found. Useful when
 * the consensus check only needs to know IF spread exceeds threshold,
 * not the exact maximum.
 *
 * @param matrix     Distance matrix [n, n]
 * @param n          Matrix size
 * @param threshold  Spread threshold
 * @param output     Output: 1 if spread > threshold, 0 otherwise
 */
extern "C" __global__ void early_termination_spread_kernel(
    const float* __restrict__ matrix,
    int n,
    float threshold,
    int* __restrict__ output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;
    if (i == j) return;

    if (matrix[i * n + j] > threshold) {
        // Atomic store — any thread finding an exceedance signals the result
        atomicExch(output, 1);
    }
}
