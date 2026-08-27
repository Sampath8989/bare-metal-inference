// Scaled CUDA Pipelined Overlap Benchmark (512x512 and 1024x1024).
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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

__global__ void matmul_stream_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
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

    std::cout << "=== V4 CUDA Pipelined Stream Overlap Benchmark (Scaled: " << scale << "x" << scale << ") ===" << std::endl;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "WARNING: No CUDA device available or driver issue (" << cudaGetErrorString(err) << "). Skipping GPU execution.\n";
        return 0;
    }

    int N = scale;
    int K = scale;
    int M = 1;
    const int NUM_STREAMS = 4;

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    float *d_A[NUM_STREAMS], *d_B[NUM_STREAMS], *d_C[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaMalloc(&d_A[i], M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B[i], K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C[i], M * N * sizeof(float)));
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_STREAMS; i++) {
        matmul_stream_kernel<<<gridDim, blockDim, 0, streams[i]>>>(d_A[i], d_B[i], d_C[i], M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Pipelined execution across " << NUM_STREAMS << " streams: " << (ms * 1000.0f) << " us total\n\n";

    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
