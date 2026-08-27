// Scaled CUDA Shared-Memory Tiled Matmul Benchmark (512x512 and 1024x1024).
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

__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        float accum = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            accum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = accum;
    }
}

__global__ void matmul_shared(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float accum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (b_row < K && col < N)
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int q = 0; q < TILE_SIZE; q++) {
            accum += tile_A[threadIdx.y][q] * tile_B[q][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = accum;
    }
}

void fill_random(float* data, int n, unsigned seed = 42) {
    static std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < n; i++) data[i] = dis(gen);
}

int main(int argc, char* argv[]) {
    int scale = 512;
    if (argc >= 2) {
        int s = std::atoi(argv[1]);
        if (s > 0) scale = s;
    }

    std::cout << "=== V4 CUDA Shared-Memory Tiled Benchmark (Scaled: " << scale << "x" << scale << ") ===" << std::endl;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "WARNING: No CUDA device available or driver issue (" << cudaGetErrorString(err) << "). Skipping GPU execution.\n";
        return 0;
    }

    int N = scale;
    size_t mat_size = static_cast<size_t>(N) * N;

    std::vector<float> h_A(mat_size), h_B(mat_size), h_C(mat_size);
    fill_random(h_A.data(), mat_size, 42);
    fill_random(h_B.data(), mat_size, 12345);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, mat_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, mat_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, mat_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), mat_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), mat_size * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    int WARMUP = 50;
    int RUNS = 200;

    // Warmup
    for (int w = 0; w < WARMUP; w++) {
        matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, N, N);
        matmul_shared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, N, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark Naive
    std::vector<float> naive_times(RUNS);
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start);
        matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, N, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        naive_times[r] = ms * 1000.0f;
    }

    // Benchmark Shared
    std::vector<float> shared_times(RUNS);
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start);
        matmul_shared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N, N, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        shared_times[r] = ms * 1000.0f;
    }

    std::sort(naive_times.begin(), naive_times.end());
    std::sort(shared_times.begin(), shared_times.end());

    float naive_p50 = naive_times[RUNS * 50 / 100];
    float shared_p50 = shared_times[RUNS * 50 / 100];

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Naive Global Kernel p50:  " << naive_p50 << " us\n";
    std::cout << "Shared Tiled Kernel p50:  " << shared_p50 << " us\n";
    std::cout << "Speedup (Naive / Shared): " << std::setprecision(2) << (naive_p50 / shared_p50) << "x\n\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
