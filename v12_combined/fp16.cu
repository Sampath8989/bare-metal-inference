#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================
// 2. FP16: Native Half-Precision CUDA Implementation
// Half-precision arithmetic (16-bit float) using standard CUDA ALUs
// ============================================================

__global__ void matmul_naive_fp16_kernel(const half* __restrict__ A,
                                         const half* __restrict__ B,
                                         half* __restrict__ C,
                                         int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        half sum = __float2half(0.0f);
        for (int k = 0; k < K; k++) {
            sum = __hadd(sum, __hmul(A[row * K + k], B[k * N + col]));
        }
        C[row * N + col] = sum;
    }
}

inline float run_fp16(int N,
                      const half* d_A,
                      const half* d_B,
                      half* d_C,
                      half* h_C,
                      const float* h_ref,
                      float& max_err,
                      float& p95,
                      float& p99,
                      int warmup = 20,
                      int runs = 100) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        matmul_naive_fp16_kernel<<<grid, block>>>(d_A, d_B, d_C, N, N, N);
    }
    cudaDeviceSynchronize();

    // Verify output & calculate max numerical error against FP32 baseline reference
    if (h_C != nullptr && h_ref != nullptr) {
        cudaMemcpy(h_C, d_C, (size_t)N * N * sizeof(half), cudaMemcpyDeviceToHost);
        max_err = 0.0f;
        for (size_t i = 0; i < (size_t)N * N; i++) {
            float val = __half2float(h_C[i]);
            max_err = std::max(max_err, std::abs(val - h_ref[i]));
        }
    }

    // Precise CUDA event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times(runs);
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(start);
        matmul_naive_fp16_kernel<<<grid, block>>>(d_A, d_B, d_C, N, N, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
        times[i] *= 1000.0f; // convert ms to us
    }

    std::sort(times.begin(), times.end());
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    p95 = times[static_cast<size_t>(runs * 0.95)];
    p99 = times[static_cast<size_t>(runs * 0.99)];
    return times[runs / 2]; // p50 in us
}
