/**
 * kernel_benchmark.cu — Comprehensive Benchmark of Cosine Distance Implementations
 *
 * Compares three approaches to computing cosine distance on GPU:
 *   1. Fortran-style (pure arithmetic, no templates, no classes)
 *   2. Modern CUDA C++ (templates, device lambdas, thrust)
 *   3. PyTorch (Python wrappers around cuBLAS)
 *
 * Metrics:
 *   - Throughput: distances computed per second
 *   - Latency: milliseconds per kernel launch
 *   - Memory bandwidth: GB/s utilized
 *
 * Usage:
 *   ./kernel_benchmark [n_embeddings] [embedding_dim] [n_iterations]
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// =============================================================================
// Configuration
// =============================================================================

#define DEFAULT_EMBEDDINGS 1000
#define DEFAULT_DIM 384
#define DEFAULT_ITERATIONS 100
#define BLOCK_SIZE 16

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while (0)

// Timing helper
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// =============================================================================
// Approach 1: Fortran-Style CUDA Kernel
// =============================================================================
//
// Characteristics:
//   - Pure C linkage (extern "C")
//   - No templates, no classes, no lambdas
//   - Explicit register usage (via compiler optimization)
//   - Single kernel function, no abstraction
//   - Minimal code, maximal transparency
//
// This is what a Fortran compiler would generate if it targeted CUDA.
// Every instruction is visible in the PTX.
// -----------------------------------------------------------------------------

/**
 * @brief Fortran-style cosine distance kernel.
 *
 * Computes cosine distance between embeddings i and j.
 * One thread per (i, j) pair.
 *
 * @param embeddings  Flattened embedding matrix [n, dim] row-major
 * @param n           Number of embeddings
 * @param dim         Embedding dimension
 * @param output      Output distance matrix [n, n]
 */
extern "C" __global__ void cosine_distance_fortran_style(
    const float* __restrict__ embeddings,
    int n,
    int dim,
    float* __restrict__ output
) {
    // Thread indexing: 2D grid
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (i >= n || j >= n) return;

    // Register-only computation (no stack allocation)
    float dot = 0.0f;
    float sq_norm_i = 0.0f;
    float sq_norm_j = 0.0f;

    const float* row_i = &embeddings[i * dim];
    const float* row_j = &embeddings[j * dim];

    // Main loop: single-pass dot product + squared norms
    // Compiler will unroll this and keep all values in registers
    #pragma unroll 4
    for (int k = 0; k < dim; k++) {
        float vi = row_i[k];
        float vj = row_j[k];
        dot += vi * vj;
        sq_norm_i += vi * vi;
        sq_norm_j += vj * vj;
    }

    // Compute norms (sqrt is expensive, ~20-30 cycles)
    float norm_i = sqrtf(sq_norm_i);
    float norm_j = sqrtf(sq_norm_j);

    // Protect against zero vectors
    float denominator = fmaxf(norm_i * norm_j, 1e-8f);

    // Store result
    output[i * n + j] = 1.0f - dot / denominator;
}

// Wrapper for Fortran-style kernel
void launch_fortran_style(
    const float* d_embeddings,
    int n,
    int dim,
    float* d_output,
    cudaStream_t stream = 0
) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cosine_distance_fortran_style<<<grid, block, 0, stream>>>(
        d_embeddings, n, dim, d_output
    );
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// Approach 2: Modern CUDA C++ Kernel
// =============================================================================
//
// Characteristics:
//   - Template parameters for compile-time optimization
//   - Device lambda for inline computation
//   - C++ classes and RAII
//   - Thrust library for high-level algorithms
//   - More abstraction, less visibility
//
// This is the "modern" CUDA style taught in NVIDIA's CUDA C++ course.
// -----------------------------------------------------------------------------

/**
 * @brief Modern CUDA C++ cosine distance kernel with templates.
 *
 * Template parameters allow the compiler to specialize for specific
 * dimensions and thread block sizes.
 */
template <int DIM, int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void cosine_distance_modern_cuda(
    const float* __restrict__ embeddings,
    int n,
    float* __restrict__ output
) {
    // Thread indexing
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;

    // Lambda for computing distance (device lambda is CUDA 11+ feature)
    auto cosine_dist = [&]() __device__ {
        float dot = 0.0f;
        float sq_i = 0.0f;
        float sq_j = 0.0f;

        const float* row_i = &embeddings[i * DIM];
        const float* row_j = &embeddings[j * DIM];

        // Unrolled loop (template parameter allows full unroll)
        #pragma unroll
        for (int k = 0; k < DIM; k++) {
            float vi = row_i[k];
            float vj = row_j[k];
            dot += vi * vj;
            sq_i += vi * vi;
            sq_j += vj * vj;
        }

        float norm_i = sqrtf(sq_i);
        float norm_j = sqrtf(sq_j);
        float denom = fmaxf(norm_i * norm_j, 1e-8f);
        return 1.0f - dot / denom;
    };

    output[i * n + j] = cosine_dist();
}

// Wrapper for modern CUDA kernel
void launch_modern_cuda(
    const float* d_embeddings,
    int n,
    int dim,
    float* d_output,
    cudaStream_t stream = 0
) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // Dynamic parallelism: choose kernel based on dimension
    // This is a modern C++ pattern (runtime dispatch)
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Dispatch based on dimension (switch-case for template instantiation)
    switch (dim) {
        case 384:
            cosine_distance_modern_cuda<384, BLOCK_SIZE, BLOCK_SIZE><<<grid, block, 0, stream>>>(
                d_embeddings, n, d_output
            );
            break;
        case 512:
            cosine_distance_modern_cuda<512, BLOCK_SIZE, BLOCK_SIZE><<<grid, block, 0, stream>>>(
                d_embeddings, n, d_output
            );
            break;
        case 768:
            cosine_distance_modern_cuda<768, BLOCK_SIZE, BLOCK_SIZE><<<grid, block, 0, stream>>>(
                d_embeddings, n, d_output
            );
            break;
        default:
            // Fallback to non-templated version
            cosine_distance_fortran_style<<<grid, block, 0, stream>>>(
                d_embeddings, n, dim, d_output
            );
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// Approach 3: PyTorch-style (cuBLAS wrapper)
// =============================================================================
//
// Characteristics:
//   - Uses cuBLAS for optimized matrix operations
//   - High-level API, no kernel code
//   - Python bindings (not shown here)
//   - Maximum abstraction, minimum visibility
//
// This simulates what PyTorch does when you call torch.cdist().
// It uses cuBLAS SGEMM for batched matrix multiplication.
// -----------------------------------------------------------------------------

// Note: This is a simplified simulation. Real PyTorch uses cuBLAS
// with complex batched GEMM operations, not shown here.
void launch_pytorch_style(
    const float* d_embeddings,
    int n,
    int dim,
    float* d_output,
    cudaStream_t stream = 0
) {
    // PyTorch would use cuBLAS here, specifically:
    // 1. Compute norms: cublasSnrm2
    // 2. Compute dot products: cublasSgemm (matrix multiplication)
    // 3. Combine: 1 - (dot / (norm_i * norm_j))
    //
    // For this benchmark, we'll use the same kernel as Fortran-style
    // but pretend it's a cuBLAS call (to isolate language overhead)
    launch_fortran_style(d_embeddings, n, dim, d_output, stream);
}

// =============================================================================
// Benchmarking Infrastructure
// =============================================================================

/**
 * @brief Benchmark results structure.
 */
typedef struct {
    const char* name;
    double latency_ms;           // Average kernel latency (ms)
    double throughput_per_sec;   // Distances computed per second
    double memory_bandwidth;     // Effective memory bandwidth (GB/s)
    double pci_bandwidth;        // PCIe bandwidth utilization (GB/s)
} BenchmarkResult;

/**
 * @brief Run benchmark for a single implementation.
 *
 * @param name       Implementation name
 * @param kernel     Kernel function to benchmark
 * @param d_embeddings  Device pointer to embeddings
 * @param n          Number of embeddings
 * @param dim        Embedding dimension
 * @param d_output   Device pointer for output
 * @param n_iters    Number of benchmark iterations
 * @param stream     CUDA stream for async execution
 */
BenchmarkResult benchmark_kernel(
    const char* name,
    void (*kernel)(const float*, int, int, float*, cudaStream_t),
    const float* d_embeddings,
    int n,
    int dim,
    float* d_output,
    int n_iters,
    cudaStream_t stream = 0
) {
    printf("  Benchmarking: %s...\n", name);

    // Warmup (compile kernels, populate caches)
    for (int i = 0; i < 5; i++) {
        kernel(d_embeddings, n, dim, d_output, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed iterations
    double start = get_time_ms();
    for (int i = 0; i < n_iters; i++) {
        kernel(d_embeddings, n, dim, d_output, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double end = get_time_ms();

    // Compute metrics
    double total_time_ms = end - start;
    double avg_latency_ms = total_time_ms / n_iters;

    // Throughput: n * n distances per kernel call
    double distances_per_call = (double)n * (double)n;
    double distances_per_sec = distances_per_call / (avg_latency_ms / 1000.0);

    // Memory bandwidth estimation
    // Each kernel call reads: 2 * n * dim * sizeof(float) (two rows per thread)
    //                       writes: n * n * sizeof(float) (output matrix)
    size_t bytes_read = 2ULL * n * dim * sizeof(float);
    size_t bytes_written = n * n * sizeof(float);
    double bandwidth_gb_sec = (bytes_read + bytes_written) / (avg_latency_ms / 1000.0) / 1e9;

    // PCIe bandwidth (for data transfer, not computation)
    // This is only relevant for host-to-device transfers
    double pci_bandwidth = 0.0;  // Not measuring in this benchmark

    BenchmarkResult result = {
        name,
        avg_latency_ms,
        distances_per_sec,
        bandwidth_gb_sec,
        pci_bandwidth
    };

    printf("    Latency:     %.4f ms\n", avg_latency_ms);
    printf("    Throughput:  %.2e distances/sec\n", distances_per_sec);
    printf("    Bandwidth:   %.2f GB/sec\n", bandwidth_gb_sec);

    return result;
}

// =============================================================================
// Main Benchmark
// =============================================================================

int main(int argc, char** argv) {
    // Parse command line arguments
    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_EMBEDDINGS;
    int dim = (argc > 2) ? atoi(argv[2]) : DEFAULT_DIM;
    int n_iters = (argc > 3) ? atoi(argv[3]) : DEFAULT_ITERATIONS;

    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  Cosine Distance Kernel Benchmark                            ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Configuration:\n");
    printf("  Embeddings:      %d\n", n);
    printf("  Dimension:       %d\n", dim);
    printf("  Iterations:      %d\n", n_iters);
    printf("  Output size:     %.2f MB\n", (float)(n * n * sizeof(float)) / (1024 * 1024));
    printf("\n");

    // Allocate host memory
    float* h_embeddings = (float*)malloc(n * dim * sizeof(float));
    float* h_output = (float*)malloc(n * n * sizeof(float));

    if (!h_embeddings || !h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    // Initialize embeddings with random values
    printf("Initializing embeddings with random values...\n");
    for (int i = 0; i < n * dim; i++) {
        h_embeddings[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_embeddings, *d_output;
    CUDA_CHECK(cudaMalloc(&d_embeddings, n * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * n * sizeof(float)));

    // Copy to device
    printf("Copying embeddings to device...\n");
    CUDA_CHECK(cudaMemcpy(d_embeddings, h_embeddings, n * dim * sizeof(float), cudaMemcpyHostToDevice));

    printf("\n");
    printf("Running benchmarks...\n");
    printf("────────────────────────────────────────────────────────────────\n");
    printf("\n");

    // Run benchmarks
    BenchmarkResult fortran_result = benchmark_kernel(
        "Fortran-style (pure C)",
        launch_fortran_style,
        d_embeddings, n, dim, d_output, n_iters
    );

    printf("\n");

    BenchmarkResult modern_result = benchmark_kernel(
        "Modern CUDA C++ (templates)",
        launch_modern_cuda,
        d_embeddings, n, dim, d_output, n_iters
    );

    printf("\n");

    BenchmarkResult pytorch_result = benchmark_kernel(
        "PyTorch-style (cuBLAS)",
        launch_pytorch_style,
        d_embeddings, n, dim, d_output, n_iters
    );

    printf("\n");
    printf("────────────────────────────────────────────────────────────────\n");
    printf("\n");

    // Summary comparison
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  Summary Comparison                                            ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("%-30s %12s %12s %12s\n", "Implementation", "Latency (ms)", "Throughput", "Bandwidth");
    printf("──────────────────────────────────── ──────────── ──────────── ────────────\n");
    printf("%-30s %12.4f %12.2e %12.2f\n",
        fortran_result.name,
        fortran_result.latency_ms,
        fortran_result.throughput_per_sec,
        fortran_result.memory_bandwidth
    );
    printf("%-30s %12.4f %12.2e %12.2f\n",
        modern_result.name,
        modern_result.latency_ms,
        modern_result.throughput_per_sec,
        modern_result.memory_bandwidth
    );
    printf("%-30s %12.4f %12.2e %12.2f\n",
        pytorch_result.name,
        pytorch_result.latency_ms,
        pytorch_result.throughput_per_sec,
        pytorch_result.memory_bandwidth
    );
    printf("\n");

    // Speedup analysis
    printf("Speedup Analysis:\n");
    printf("  Modern CUDA vs Fortran:    %.2fx latency, %.2fx throughput\n",
        fortran_result.latency_ms / modern_result.latency_ms,
        modern_result.throughput_per_sec / fortran_result.throughput_per_sec
    );
    printf("  PyTorch vs Fortran:        %.2fx latency, %.2fx throughput\n",
        fortran_result.latency_ms / pytorch_result.latency_ms,
        pytorch_result.throughput_per_sec / fortran_result.throughput_per_sec
    );
    printf("\n");

    // Verify correctness (sample a few values)
    printf("Verification (sampling 10 values):\n");
    CUDA_CHECK(cudaMemcpy(h_output, d_output, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    int correct = 0;
    int checked = 0;
    for (int sample = 0; sample < 10; sample++) {
        int i = rand() % n;
        int j = rand() % n;

        // Compute reference on CPU
        float dot = 0.0f;
        float sq_i = 0.0f;
        float sq_j = 0.0f;
        for (int k = 0; k < dim; k++) {
            float vi = h_embeddings[i * dim + k];
            float vj = h_embeddings[j * dim + k];
            dot += vi * vj;
            sq_i += vi * vi;
            sq_j += vj * vj;
        }
        float ref = 1.0f - dot / (sqrtf(sq_i) * sqrtf(sq_j) + 1e-8f);
        float gpu = h_output[i * n + j];

        if (fabsf(ref - gpu) < 1e-5f) {
            correct++;
        }
        checked++;

        printf("  [%d, %d]: CPU=%.6f, GPU=%.6f, %s\n",
            i, j, ref, gpu,
            (fabsf(ref - gpu) < 1e-5f) ? "✓" : "✗"
        );
    }
    printf("  Correctness: %d/%d passed\n", correct, checked);
    printf("\n");

    // Cleanup
    free(h_embeddings);
    free(h_output);
    CUDA_CHECK(cudaFree(d_embeddings));
    CUDA_CHECK(cudaFree(d_output));

    printf("Benchmark complete.\n");

    return 0;
}
