#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================
// 5. KERNEL FUSION: Fused GEMM + Activation Epilogue
// Copied from: v9_fusion_streams/cuda_fusion.cu
// Fuses memory writes and eliminates intermediate DRAM roundtrips
// ============================================================

__global__ void matmul_fused_relu_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        // Fused Activation Epilogue (avoids separate DRAM load/store pass)
        C[row * N + col] = fmaxf(sum, 0.0f);
    }
}

inline float run_kernel_fusion(int N,
                               const float* d_A,
                               const float* d_B,
                               float* d_C,
                               float* h_C,
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
        matmul_fused_relu_kernel<<<grid, block>>>(d_A, d_B, d_C, N, N, N);
    }
    cudaDeviceSynchronize();

    // Verify output & calculate max numerical error against ReLU(FP32 baseline reference)
    if (h_C != nullptr && h_ref != nullptr) {
        cudaMemcpy(h_C, d_C, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost);
        max_err = 0.0f;
        for (size_t i = 0; i < (size_t)N * N; i++) {
            float expected = std::max(0.0f, h_ref[i]);
            max_err = std::max(max_err, std::abs(h_C[i] - expected));
        }
    }

    // Precise CUDA event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times(runs);
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(start);
        matmul_fused_relu_kernel<<<grid, block>>>(d_A, d_B, d_C, N, N, N);
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
