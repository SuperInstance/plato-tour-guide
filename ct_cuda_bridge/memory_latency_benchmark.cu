/**
 * Memory Latency Benchmark: Measure Actual GPU Memory Hierarchy Performance
 *
 * This module measures performance across the GPU memory hierarchy:
 * - Registers: Fastest, zero latency (if no spill)
 * - Shared Memory: ~30 cycles latency, user-managed cache
 * - L1 Cache: ~80 cycles latency, hardware cache
 * - L2 Cache: ~200 cycles latency, larger cache
 * - Global Memory (DRAM): ~500 cycles latency, highest bandwidth
 *
 * Benchmark strategy:
 * 1. Tile headers: Frequent access → test L1 hit rate
 * 2. Partial answers: Working set → test L2 hit rate
 * 3. Embedding matrices: Large data → test global memory bandwidth
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

// Benchmark configurations
#define ARRAY_SIZE (1024 * 1024)  // 1M elements
#define ITERATIONS 1000

/**
 * Register-only benchmark: No memory access
 *
 * All data fits in registers. This is the theoretical upper bound.
 */
__global__ void register_benchmark(
    const float* input,
    float* output,
    const int size
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        // Preload to register
        float val = input[tid];

        // Pure register operations
        float result = 0.0f;
        for (int i = 0; i < 100; i++) {
            result += val * val;
        }

        output[tid] = result;
    }
}

/**
 * Shared memory benchmark: User-managed cache
 *
 * Data is explicitly loaded into shared memory for fast access.
 * Simulates tile header access pattern.
 */
__global__ void shared_memory_benchmark(
    const float* input,
    float* output,
    const int size
) {
    extern __shared__ float shared_data[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    // Load global memory → shared memory
    if (tid < size) {
        shared_data[local_tid] = input[tid];
    }
    __syncthreads();

    // Access from shared memory repeatedly
    if (tid < size) {
        float val = shared_data[local_tid];
        float result = 0.0f;
        for (int i = 0; i < 100; i++) {
            result += val * val;
        }
        output[tid] = result;
    }
}

/**
 * L1 cache benchmark: Hardware cache
 *
 * Let L1 cache handle frequently accessed data.
 * Simulates partial answer access pattern.
 */
__global__ void l1_cache_benchmark(
    const float* input,
    float* output,
    const int size,
    const int repeat_count
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        float result = 0.0f;

        // Repeatedly access same data (hits L1 cache)
        for (int i = 0; i < repeat_count; i++) {
            float val = input[tid];  // L1 cache hit after first access
            result += val * val;
        }

        output[tid] = result;
    }
}

/**
 * L2 cache benchmark: Larger cache, higher latency
 *
 * Access pattern that exceeds L1 but fits in L2.
 */
__global__ void l2_cache_benchmark(
    const float* input,
    float* output,
    const int size,
    const int stride
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        // Strided access to evict from L1, keep in L2
        float result = 0.0f;
        for (int i = 0; i < 100; i++) {
            int idx = (tid + i * stride) % size;
            float val = input[idx];
            result += val * val;
        }

        output[tid] = result;
    }
}

/**
 * Global memory benchmark: DRAM access
 *
 * Each access goes to global memory.
 * Simulates embedding matrix access pattern.
 */
__global__ void global_memory_benchmark(
    const float* input,
    float* output,
    const int size
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        // Each iteration accesses different memory (no cache reuse)
        float result = 0.0f;
        for (int i = 0; i < 100; i++) {
            int idx = (tid * 100 + i) % size;  // Stride to avoid cache
            float val = input[idx];  // Always misses cache
            result += val * val;
        }

        output[tid] = result;
    }
}

/**
 * Coalesced vs Strided access benchmark
 *
 * Coalesced: Adjacent threads access adjacent memory (fast)
 * Strided: Threads access far apart (slow, multiple transactions)
 */
__global__ void coalesced_access_kernel(
    const float* input,
    float* output,
    const int size
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        // Coalesced: thread i accesses element i
        output[tid] = input[tid] * input[tid];
    }
}

__global__ void strided_access_kernel(
    const float* input,
    float* output,
    const int size,
    const int stride
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        // Strided: thread i accesses element i * stride
        int idx = (tid * stride) % size;
        output[tid] = input[idx] * input[idx];
    }
}

/**
 * Register pressure benchmark
 *
 * Measure performance degradation when registers spill to local memory.
 */
__global__ void register_pressure_kernel(
    const float* input,
    float* output,
    const int size,
    const int compute_intensity
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        // Use many registers to cause pressure
        float r0 = input[tid];
        float r1 = r0 * 1.1f;
        float r2 = r0 * 1.2f;
        float r3 = r0 * 1.3f;
        float r4 = r0 * 1.4f;
        float r5 = r0 * 1.5f;
        float r6 = r0 * 1.6f;
        float r7 = r0 * 1.7f;
        float r8 = r0 * 1.8f;
        float r9 = r0 * 1.9f;

        // Intensive computation to keep registers live
        float result = 0.0f;
        for (int i = 0; i < compute_intensity; i++) {
            result += r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9;
        }

        output[tid] = result;
    }
}

/**
 * CUDA managed memory vs pre-allocated static memory benchmark
 */
__global__ void managed_memory_kernel(
    float* data,
    const int size
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        data[tid] = data[tid] * data[tid];
    }
}

/**
 * Run all benchmarks and output comparison table
 */
extern "C" void run_memory_hierarchy_benchmarks() {
    printf("=== GPU Memory Hierarchy Benchmark ===\n\n");

    // Allocate memory
    float *d_input, *d_output, *h_input;
    size_t bytes = ARRAY_SIZE * sizeof(float);

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    h_input = (float*)malloc(bytes);

    // Initialize data
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = (float)i / ARRAY_SIZE;
    }
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    // 1. Register benchmark
    printf("1. Register-only benchmark:\n");
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        register_benchmark<<<(ARRAY_SIZE + 255) / 256, 256>>>(
            d_input, d_output, ARRAY_SIZE
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("   Time: %.3f ms (%.2f us/iter)\n", elapsed_time, elapsed_time * 1000 / ITERATIONS);
    printf("   Bandwidth: %.2f GB/s\n",
           2 * bytes * ITERATIONS / (elapsed_time / 1000.0) / 1e9);

    // 2. Shared memory benchmark
    printf("\n2. Shared memory benchmark:\n");
    size_t shared_mem_size = 256 * sizeof(float);  // Max threads per block
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        shared_memory_benchmark<<<(ARRAY_SIZE + 255) / 256, 256, shared_mem_size>>>(
            d_input, d_output, ARRAY_SIZE
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("   Time: %.3f ms (%.2f us/iter)\n", elapsed_time, elapsed_time * 1000 / ITERATIONS);
    printf("   Bandwidth: %.2f GB/s\n",
           2 * bytes * ITERATIONS / (elapsed_time / 1000.0) / 1e9);

    // 3. L1 cache benchmark
    printf("\n3. L1 cache benchmark:\n");
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        l1_cache_benchmark<<<(ARRAY_SIZE + 255) / 256, 256>>>(
            d_input, d_output, ARRAY_SIZE, 100
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("   Time: %.3f ms (%.2f us/iter)\n", elapsed_time, elapsed_time * 1000 / ITERATIONS);
    printf("   Bandwidth: %.2f GB/s\n",
           2 * bytes * ITERATIONS / (elapsed_time / 1000.0) / 1e9);

    // 4. L2 cache benchmark
    printf("\n4. L2 cache benchmark:\n");
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        l2_cache_benchmark<<<(ARRAY_SIZE + 255) / 256, 256>>>(
            d_input, d_output, ARRAY_SIZE, 1024
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("   Time: %.3f ms (%.2f us/iter)\n", elapsed_time, elapsed_time * 1000 / ITERATIONS);
    printf("   Bandwidth: %.2f GB/s\n",
           2 * bytes * ITERATIONS / (elapsed_time / 1000.0) / 1e9);

    // 5. Global memory benchmark
    printf("\n5. Global memory benchmark:\n");
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        global_memory_benchmark<<<(ARRAY_SIZE + 255) / 256, 256>>>(
            d_input, d_output, ARRAY_SIZE
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("   Time: %.3f ms (%.2f us/iter)\n", elapsed_time, elapsed_time * 1000 / ITERATIONS);
    printf("   Bandwidth: %.2f GB/s\n",
           2 * bytes * ITERATIONS / (elapsed_time / 1000.0) / 1e9);

    // 6. Coalesced vs Strided access
    printf("\n6. Access pattern comparison:\n");
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        coalesced_access_kernel<<<(ARRAY_SIZE + 255) / 256, 256>>>(
            d_input, d_output, ARRAY_SIZE
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("   Coalesced: %.3f ms (%.2f GB/s)\n", elapsed_time,
           2 * bytes * ITERATIONS / (elapsed_time / 1000.0) / 1e9);

    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        strided_access_kernel<<<(ARRAY_SIZE + 255) / 256, 256>>>(
            d_input, d_output, ARRAY_SIZE, 32
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("   Strided:   %.3f ms (%.2f GB/s)\n", elapsed_time,
           2 * bytes * ITERATIONS / (elapsed_time / 1000.0) / 1e9);

    // 7. Register pressure
    printf("\n7. Register pressure analysis:\n");
    for (int intensity = 1; intensity <= 100; intensity *= 10) {
        cudaEventRecord(start);
        for (int i = 0; i < ITERATIONS / 10; i++) {
            register_pressure_kernel<<<(ARRAY_SIZE + 255) / 256, 256>>>(
                d_input, d_output, ARRAY_SIZE, intensity
            );
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("   Intensity %d: %.3f ms (%.2f GB/s)\n", intensity, elapsed_time,
               2 * bytes * (ITERATIONS / 10) / (elapsed_time / 1000.0) / 1e9);
    }

    // 8. Managed memory vs static memory
    printf("\n8. Memory allocation comparison:\n");

    // Static memory
    float *d_static_input, *d_static_output;
    cudaMalloc(&d_static_input, bytes);
    cudaMalloc(&d_static_output, bytes);
    cudaMemcpy(d_static_input, h_input, bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        managed_memory_kernel<<<(ARRAY_SIZE + 255) / 256, 256>>>(
            d_static_input, ARRAY_SIZE
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("   Static allocation: %.3f ms\n", elapsed_time);

    cudaFree(d_static_input);
    cudaFree(d_static_output);

    // Managed memory
    float *d_managed_input, *d_managed_output;
    cudaMallocManaged(&d_managed_input, bytes);
    cudaMallocManaged(&d_managed_output, bytes);
    memcpy(d_managed_input, h_input, bytes);

    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        managed_memory_kernel<<<(ARRAY_SIZE + 255) / 256, 256>>>(
            d_managed_input, ARRAY_SIZE
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("   Managed memory:   %.3f ms\n", elapsed_time);

    cudaFree(d_managed_input);
    cudaFree(d_managed_output);

    // Summary
    printf("\n=== Summary ===\n");
    printf("Memory hierarchy (fastest to slowest):\n");
    printf("  1. Registers: Zero latency (if no spill)\n");
    printf("  2. Shared Memory: ~30 cycles\n");
    printf("  3. L1 Cache: ~80 cycles\n");
    printf("  4. L2 Cache: ~200 cycles\n");
    printf("  5. Global Memory: ~500 cycles\n");
    printf("\nRecommendations:\n");
    printf("  - Keep hot data in registers (tile headers)\n");
    printf("  - Use shared memory for thread-group data\n");
    printf("  - Coalesce global memory accesses\n");
    printf("  - Avoid register spill (limit register usage)\n");
    printf("  - Prefer static allocation over managed memory\n");

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * Host interface for running specific benchmark
 */
extern "C" void run_benchmark(const char* benchmark_name) {
    if (strcmp(benchmark_name, "all") == 0) {
        run_memory_hierarchy_benchmarks();
    } else {
        printf("Running benchmark: %s\n", benchmark_name);
        // TODO: Implement individual benchmark selection
    }
}
