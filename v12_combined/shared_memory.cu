#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

// ============================================================
// 3. SHARED MEMORY: 2D Tiled FP32 Implementation
// Copied from: v4_cuda/cuda_shared.cu
// Stages inputs into 16x16 shared memory tiles to minimize global memory roundtrips
// ============================================================

#define SMEM_TILE_SIZE 16

__global__ void matmul_shared_fp32_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K) {
    __shared__ float tile_A[SMEM_TILE_SIZE][SMEM_TILE_SIZE];
    __shared__ float tile_B[SMEM_TILE_SIZE][SMEM_TILE_SIZE];

    int row = blockIdx.y * SMEM_TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * SMEM_TILE_SIZE + threadIdx.x;

    float accum = 0.0f;

    for (int t = 0; t < (K + SMEM_TILE_SIZE - 1) / SMEM_TILE_SIZE; t++) {
        int a_col = t * SMEM_TILE_SIZE + threadIdx.x;
        int b_row = t * SMEM_TILE_SIZE + threadIdx.y;

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
        for (int q = 0; q < SMEM_TILE_SIZE; q++) {
            accum += tile_A[threadIdx.y][q] * tile_B[q][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = accum;
    }
}

inline float run_shared_memory(int N,
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
    dim3 block(SMEM_TILE_SIZE, SMEM_TILE_SIZE);
    dim3 grid((N + SMEM_TILE_SIZE - 1) / SMEM_TILE_SIZE, (N + SMEM_TILE_SIZE - 1) / SMEM_TILE_SIZE);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        matmul_shared_fp32_kernel<<<grid, block>>>(d_A, d_B, d_C, N, N, N);
    }
    cudaDeviceSynchronize();

    // Verify output & calculate max numerical error against FP32 baseline reference
    if (h_C != nullptr && h_ref != nullptr) {
        cudaMemcpy(h_C, d_C, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost);
        max_err = 0.0f;
        for (size_t i = 0; i < (size_t)N * N; i++) {
            max_err = std::max(max_err, std::abs(h_C[i] - h_ref[i]));
        }
    }

    // Precise CUDA event timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> times(runs);
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(start);
        matmul_shared_fp32_kernel<<<grid, block>>>(d_A, d_B, d_C, N, N, N);
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
