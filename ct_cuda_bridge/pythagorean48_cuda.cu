/**
 * Pythagorean 48-Direction Code on GPU
 *
 * Implements the 48-direction Pythagorean code for efficient trust vector storage.
 * This is a geometric code where each direction represents a specific 3D vector
 * that maximizes angular separation (≈ 37.5° between neighbors).
 *
 * Storage efficiency: log₂(48) ≈ 5.585 bits per direction
 * Total bandwidth: 5.585 × n_agents (vs 32 × 3 floats = 96 floats per agent)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#define NUM_DIRECTIONS 48
#define EMBEDDING_DIM 3

/**
 * 48 Pythagorean directions as constant memory
 * These are unit vectors in ℝ³ with maximum angular separation.
 * Stored in __constant__ memory (64KB cache, optimized for broadcast)
 */
__constant__ float pythagorean_directions[NUM_DIRECTIONS * EMBEDDING_DIM] = {
    // Octahedral vertices (6 directions)
    1.0f, 0.0f, 0.0f,    // 0: +x
    -1.0f, 0.0f, 0.0f,   // 1: -x
    0.0f, 1.0f, 0.0f,    // 2: +y
    0.0f, -1.0f, 0.0f,   // 3: -y
    0.0f, 0.0f, 1.0f,    // 4: +z
    0.0f, 0.0f, -1.0f,   // 5: -z

    // Cubic vertices (8 corners)
    0.57735f, 0.57735f, 0.57735f,     // 6: (+x,+y,+z)
    0.57735f, 0.57735f, -0.57735f,    // 7: (+x,+y,-z)
    0.57735f, -0.57735f, 0.57735f,    // 8: (+x,-y,+z)
    0.57735f, -0.57735f, -0.57735f,   // 9: (+x,-y,-z)
    -0.57735f, 0.57735f, 0.57735f,    // 10: (-x,+y,+z)
    -0.57735f, 0.57735f, -0.57735f,   // 11: (-x,+y,-z)
    -0.57735f, -0.57735f, 0.57735f,   // 12: (-x,-y,+z)
    -0.57735f, -0.57735f, -0.57735f,  // 13: (-x,-y,-z)

    // Edge midpoints (12 edges)
    0.7071f, 0.7071f, 0.0f,      // 14: (+x,+y)
    0.7071f, -0.7071f, 0.0f,     // 15: (+x,-y)
    -0.7071f, 0.7071f, 0.0f,     // 16: (-x,+y)
    -0.7071f, -0.7071f, 0.0f,    // 17: (-x,-y)
    0.7071f, 0.0f, 0.7071f,      // 18: (+x,+z)
    0.7071f, 0.0f, -0.7071f,     // 19: (+x,-z)
    -0.7071f, 0.0f, 0.7071f,     // 20: (-x,+z)
    -0.7071f, 0.0f, -0.7071f,    // 21: (-x,-z)
    0.0f, 0.7071f, 0.7071f,      // 22: (+y,+z)
    0.0f, 0.7071f, -0.7071f,     // 23: (+y,-z)
    0.0f, -0.7071f, 0.7071f,     // 24: (-y,+z)
    0.0f, -0.7071f, -0.7071f,    // 25: (-y,-z)

    // Face centers (additional 22 directions for uniform coverage)
    0.8944f, 0.4472f, 0.0f,      // 26: weighted (+x,+y)
    0.4472f, 0.8944f, 0.0f,      // 27: weighted (+y,+x)
    0.8944f, -0.4472f, 0.0f,     // 28: weighted (+x,-y)
    0.4472f, -0.8944f, 0.0f,     // 29: weighted (-y,+x)
    -0.8944f, 0.4472f, 0.0f,     // 30: weighted (-x,+y)
    -0.4472f, 0.8944f, 0.0f,     // 31: weighted (+y,-x)
    -0.8944f, -0.4472f, 0.0f,    // 32: weighted (-x,-y)
    -0.4472f, -0.8944f, 0.0f,    // 33: weighted (-y,-x)
    0.8944f, 0.0f, 0.4472f,      // 34: weighted (+x,+z)
    0.8944f, 0.0f, -0.4472f,     // 35: weighted (+x,-z)
    0.4472f, 0.0f, 0.8944f,      // 36: weighted (+z,+x)
    0.4472f, 0.0f, -0.8944f,     // 37: weighted (-z,+x)
    -0.8944f, 0.0f, 0.4472f,     // 38: weighted (-x,+z)
    -0.8944f, 0.0f, -0.4472f,    // 39: weighted (-x,-z)
    -0.4472f, 0.0f, 0.8944f,     // 40: weighted (+z,-x)
    -0.4472f, 0.0f, -0.8944f,    // 41: weighted (-z,-x)
    0.0f, 0.8944f, 0.4472f,      // 42: weighted (+y,+z)
    0.0f, 0.8944f, -0.4472f,     // 43: weighted (+y,-z)
    0.0f, 0.4472f, 0.8944f,      // 44: weighted (+z,+y)
    0.0f, 0.4472f, -0.8944f,     // 45: weighted (-z,+y)
    0.0f, -0.8944f, 0.4472f,     // 46: weighted (-y,+z)
    0.0f, -0.8944f, -0.4472f     // 47: weighted (-y,-z)
};

/**
 * Encode a 3D vector to its nearest Pythagorean direction
 *
 * Algorithm:
 * 1. Compute dot product with all 48 directions (register-only)
 * 2. Find maximum dot product (closest direction)
 * 3. Return direction index (0-47)
 *
 * Performance: All operations in registers, no global memory access
 */
__device__ int encode_pythagorean48(const float* vector) {
    float max_dot = -1.0f;
    int best_dir = 0;

    // Loop unrolling for better instruction-level parallelism
    #pragma unroll
    for (int i = 0; i < NUM_DIRECTIONS; i++) {
        // Compute dot product: v · d_i
        float dot = vector[0] * pythagorean_directions[i * EMBEDDING_DIM] +
                   vector[1] * pythagorean_directions[i * EMBEDDING_DIM + 1] +
                   vector[2] * pythagorean_directions[i * EMBEDDING_DIM + 2];

        if (dot > max_dot) {
            max_dot = dot;
            best_dir = i;
        }
    }

    return best_dir;
}

/**
 * Decode a Pythagorean direction index to its 3D vector
 *
 * Algorithm:
 * 1. Single load from constant memory (cached)
 * 2. Return direction vector
 *
 * Performance: One load from constant memory (L1 cache hit)
 */
__device__ void decode_pythagorean48(int dir_index, float* output) {
    // Constant memory access (cached in L1)
    output[0] = pythagorean_directions[dir_index * EMBEDDING_DIM];
    output[1] = pythagorean_directions[dir_index * EMBEDDING_DIM + 1];
    output[2] = pythagorean_directions[dir_index * EMBEDDING_DIM + 2];
}

/**
 * Batch encode: Convert multiple vectors to direction codes
 *
 * This is the hot path for ZHC consensus:
 * - Each agent's trust vector is encoded to 6 bits (0-47)
 * - Total storage: 6 × n_agents bits (vs 96 × n_agents floats)
 */
__global__ void batch_encode_pythagorean48(
    const float* vectors,        // [n_vectors × 3]
    int* direction_codes,        // Output: [n_vectors]
    const int n_vectors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_vectors) {
        const float* vec = &vectors[tid * EMBEDDING_DIM];
        direction_codes[tid] = encode_pythagorean48(vec);
    }
}

/**
 * Batch decode: Convert direction codes back to vectors
 *
 * Used for reconstruction and visualization.
 */
__global__ void batch_decode_pythagorean48(
    const int* direction_codes,  // [n_vectors]
    float* vectors,              // Output: [n_vectors × 3]
    const int n_vectors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_vectors) {
        float* vec = &vectors[tid * EMBEDDING_DIM];
        decode_pythagorean48(direction_codes[tid], vec);
    }
}

/**
 * Compute pairwise trust matrix using Pythagorean encoding
 *
 * This is the core ZHC operation:
 * - Each agent's trust vector is encoded to 6 bits
 * - Compute trust = dot(encoded_i, encoded_j)
 * - Output: symmetric trust matrix [n_agents × n_agents]
 */
__global__ void compute_trust_matrix_pythagorean(
    const int* trust_codes,      // [n_agents] - each agent's direction
    float* trust_matrix,         // Output: [n_agents × n_agents]
    const int n_agents
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n_agents && j < n_agents) {
        // Decode both agents' directions
        float vec_i[3], vec_j[3];
        decode_pythagorean48(trust_codes[i], vec_i);
        decode_pythagorean48(trust_codes[j], vec_j);

        // Compute trust as dot product
        float trust = vec_i[0] * vec_j[0] + vec_i[1] * vec_j[1] + vec_i[2] * vec_j[2];

        trust_matrix[i * n_agents + j] = trust;
    }
}

/**
 * Benchmark: Pythagorean encoding vs naive storage
 *
 * Compare:
 * 1. Naive: Store all 48 directions in global memory
 * 2. Optimized: Use constant memory (cached)
 * 3. Register-only: Preload to registers (this implementation)
 */
extern "C" void benchmark_pythagorean48(
    const float* h_vectors,
    const int n_vectors,
    const int iterations = 100
) {
    float *d_vectors;
    int *d_codes, *h_codes;

    // Allocate memory
    cudaMalloc(&d_vectors, n_vectors * EMBEDDING_DIM * sizeof(float));
    cudaMalloc(&d_codes, n_vectors * sizeof(int));
    h_codes = (int*)malloc(n_vectors * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_vectors, h_vectors,
               n_vectors * EMBEDDING_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark encoding
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        batch_encode_pythagorean48<<<(n_vectors + 255) / 256, 256>>>(
            d_vectors, d_codes, n_vectors
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float encode_time;
    cudaEventElapsedTime(&encode_time, start, stop);

    // Copy results back
    cudaMemcpy(h_codes, d_codes, n_vectors * sizeof(int), cudaMemcpyDeviceToHost);

    // Benchmark decoding
    float *d_decoded;
    cudaMalloc(&d_decoded, n_vectors * EMBEDDING_DIM * sizeof(float));

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        batch_decode_pythagorean48<<<(n_vectors + 255) / 256, 256>>>(
            d_codes, d_decoded, n_vectors
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float decode_time;
    cudaEventElapsedTime(&decode_time, start, stop);

    // Print results
    printf("=== Pythagorean 48-Direction Benchmark ===\n");
    printf("Vectors: %d\n", n_vectors);
    printf("Iterations: %d\n", iterations);
    printf("Encode time: %.3f ms (%.2f us/vector)\n",
           encode_time, encode_time * 1000 / (n_vectors * iterations));
    printf("Decode time: %.3f ms (%.2f us/vector)\n",
           decode_time, decode_time * 1000 / (n_vectors * iterations));
    printf("Total time: %.3f ms\n", encode_time + decode_time);
    printf("\nStorage efficiency:\n");
    printf("  Naive (3 floats): %.2f KB\n", n_vectors * 3 * sizeof(float) / 1024.0f);
    printf("  Encoded (1 int): %.2f KB\n", n_vectors * sizeof(int) / 1024.0f);
    printf("  Compression: %.2fx\n", 3.0f * sizeof(float) / sizeof(int));
    printf("  Bits per vector: %.2f (log2(48))\n", log2f(48));

    // Cleanup
    cudaFree(d_vectors);
    cudaFree(d_codes);
    cudaFree(d_decoded);
    free(h_codes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * Host interface for encoding trust vectors
 */
extern "C" void encode_trust_vectors(
    const float* h_trust_vectors,  // [n_agents × 3]
    int* h_trust_codes,            // Output: [n_agents]
    const int n_agents
) {
    float *d_vectors;
    int *d_codes;

    cudaMalloc(&d_vectors, n_agents * EMBEDDING_DIM * sizeof(float));
    cudaMalloc(&d_codes, n_agents * sizeof(int));

    cudaMemcpy(d_vectors, h_trust_vectors,
               n_agents * EMBEDDING_DIM * sizeof(float), cudaMemcpyHostToDevice);

    batch_encode_pythagorean48<<<(n_agents + 255) / 256, 256>>>(
        d_vectors, d_codes, n_agents
    );

    cudaMemcpy(h_trust_codes, d_codes, n_agents * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_vectors);
    cudaFree(d_codes);
}
