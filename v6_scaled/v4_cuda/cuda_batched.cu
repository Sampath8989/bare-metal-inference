// Scaled CUDA Batched Inference Benchmark (512x512 and 1024x1024).
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_SIZE 16

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

__global__ void matmul_batched_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        float accum = 0.0f;
        for (int q = 0; q < K; q++) {
            accum += A[i * K + q] * B[q * N + j];
        }
        C[i * N + j] = accum;
    }
}

int main(int argc, char* argv[]) {
    int scale = 512;
    if (argc >= 2) {
        int s = std::atoi(argv[1]);
        if (s > 0) scale = s;
    }

    std::cout << "=== V4 CUDA Batched Sweep Benchmark (Scaled: " << scale << "x" << scale << ") ===" << std::endl;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "WARNING: No CUDA device available or driver issue (" << cudaGetErrorString(err) << "). Skipping GPU execution.\n";
        return 0;
    }

    int N = scale;
    int K = scale;
    std::vector<int> batch_sizes = {1, 4, 16, 32, 64};

    for (int batch : batch_sizes) {
        int M = batch;
        size_t size_A = static_cast<size_t>(M) * K;
        size_t size_B = static_cast<size_t>(K) * N;
        size_t size_C = static_cast<size_t>(M) * N;

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));

        dim3 blockDim(16, 16);
        dim3 gridDim((N + 15) / 16, (M + 15) / 16);

        // Warmup
        for (int w = 0; w < 10; w++) {
            matmul_batched_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int RUNS = 100;
        cudaEventRecord(start);
        for (int r = 0; r < RUNS; r++) {
            matmul_batched_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        float avg_us = (ms / RUNS) * 1000.0f;
        float throughput = (M * 1000000.0f) / avg_us;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Batch size " << std::setw(3) << M << ": " << std::setw(8) << avg_us << " us/batch | Throughput: " << std::setw(10) << throughput << " samples/sec\n";

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    std::cout << "\n";
    return 0;
}
