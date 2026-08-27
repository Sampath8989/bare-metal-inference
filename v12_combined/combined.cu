#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace nvcuda;

// ============================================================
// 6. FINAL COMBINED: FP16 + Shared Memory + WMMA + Kernel Fusion
// Real integrated kernel combining all 4 GPU optimizations:
//   - FP16: Half-precision storage and input loads
//   - Shared Memory: 64x64 multi-warp on-chip staging with bank conflict padding (SMEM_PAD=8)
//   - WMMA Tensor Cores: 16x16x16 matrix-multiply-accumulate fragments
//   - Kernel Fusion: In-register activation epilogue (ReLU) before global memory store
// ============================================================

constexpr int COMB_WMMA_M = 16;
constexpr int COMB_WMMA_N = 16;
constexpr int COMB_WMMA_K = 16;

constexpr int COMB_TILE_M = 64;
constexpr int COMB_TILE_N = 64;
constexpr int COMB_TILE_K = 16;

constexpr int COMB_SMEM_PAD = 8;
constexpr int COMB_SMEM_A_STRIDE = COMB_TILE_K + COMB_SMEM_PAD;
constexpr int COMB_SMEM_B_STRIDE = COMB_TILE_N + COMB_SMEM_PAD;

__global__ void matmul_combined_fp16_smem_wmma_fused_kernel(const half* __restrict__ A,
                                                           const half* __restrict__ B,
                                                           float* __restrict__ C,
                                                           int M, int N, int K) {
    __shared__ half smem_A[COMB_TILE_M][COMB_TILE_K + COMB_SMEM_PAD];
    __shared__ half smem_B[COMB_TILE_K][COMB_TILE_N + COMB_SMEM_PAD];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int block_row = blockIdx.y * COMB_TILE_M;
    int block_col = blockIdx.x * COMB_TILE_N;

    // 8 warps compute 16 (4x4) WMMA tiles
    constexpr int TILES_PER_WARP = (COMB_TILE_M / COMB_WMMA_M) * (COMB_TILE_N / COMB_WMMA_N) / 8;
    int tile_row[TILES_PER_WARP];
    int tile_col[TILES_PER_WARP];
    #pragma unroll
    for (int t = 0; t < TILES_PER_WARP; t++) {
        int tile_idx = warp_id + t * 8;
        tile_row[t] = tile_idx / 4;
        tile_col[t] = tile_idx % 4;
    }

    wmma::fragment<wmma::accumulator, COMB_WMMA_M, COMB_WMMA_N, COMB_WMMA_K, float> c_frag[TILES_PER_WARP];
    #pragma unroll
    for (int t = 0; t < TILES_PER_WARP; t++) {
        wmma::fill_fragment(c_frag[t], 0.0f);
    }

    wmma::fragment<wmma::matrix_a, COMB_WMMA_M, COMB_WMMA_N, COMB_WMMA_K, half, wmma::row_major> a_frag[TILES_PER_WARP];
    wmma::fragment<wmma::matrix_b, COMB_WMMA_M, COMB_WMMA_N, COMB_WMMA_K, half, wmma::row_major> b_frag[TILES_PER_WARP];

    constexpr int TOTAL_LOADS_A = COMB_TILE_M * COMB_TILE_K;
    constexpr int TOTAL_LOADS_B = COMB_TILE_K * COMB_TILE_N;
    constexpr int NTHREADS = 256;

    for (int k = 0; k < K; k += COMB_WMMA_K) {
        // Stage FP16 tile A into shared memory
        #pragma unroll
        for (int i = tid; i < TOTAL_LOADS_A; i += NTHREADS) {
            int smem_row = i / COMB_TILE_K;
            int smem_col = i % COMB_TILE_K;
            int global_row = block_row + smem_row;
            int global_col = k + smem_col;

            half val = (global_row < M && global_col < K)
                           ? A[global_row * K + global_col]
                           : __float2half(0.0f);
            smem_A[smem_row][smem_col] = val;
        }

        // Stage FP16 tile B into shared memory
        #pragma unroll
        for (int i = tid; i < TOTAL_LOADS_B; i += NTHREADS) {
            int smem_row = i / COMB_TILE_N;
            int smem_col = i % COMB_TILE_N;
            int global_row = k + smem_row;
            int global_col = block_col + smem_col;

            half val = (global_row < K && global_col < N)
                           ? B[global_row * N + global_col]
                           : __float2half(0.0f);
            smem_B[smem_row][smem_col] = val;
        }

        __syncthreads();

        // Compute sub-matrix multiplication using WMMA Tensor Cores
        #pragma unroll
        for (int t = 0; t < TILES_PER_WARP; t++) {
            half* a_ptr = &smem_A[tile_row[t] * COMB_WMMA_M][0];
            wmma::load_matrix_sync(a_frag[t], a_ptr, COMB_SMEM_A_STRIDE);

            half* b_ptr = &smem_B[0][tile_col[t] * COMB_WMMA_N];
            wmma::load_matrix_sync(b_frag[t], b_ptr, COMB_SMEM_B_STRIDE);

            wmma::mma_sync(c_frag[t], a_frag[t], b_frag[t], c_frag[t]);
        }

        __syncthreads();
    }

    // Fused Kernel Epilogue: apply ReLU activation directly to fragment registers
    #pragma unroll
    for (int t = 0; t < TILES_PER_WARP; t++) {
        for (int elem = 0; elem < c_frag[t].num_elements; elem++) {
            c_frag[t].x[elem] = fmaxf(c_frag[t].x[elem], 0.0f);
        }

        int out_row = block_row + tile_row[t] * COMB_WMMA_M;
        int out_col = block_col + tile_col[t] * COMB_WMMA_N;

        if (out_row < M && out_col < N) {
            float* c_ptr = C + out_row * N + out_col;
            wmma::store_matrix_sync(c_ptr, c_frag[t], N, wmma::mem_row_major);
        }
    }
}

inline float run_combined(int N,
                          const half* d_A,
                          const half* d_B,
                          float* d_C,
                          float* h_C,
                          const float* h_ref,
                          float& max_err,
                          float& p95,
                          float& p99,
                          int warmup = 20,
                          int runs = 100) {
    dim3 block(16, 16);
    dim3 grid((N + COMB_TILE_N - 1) / COMB_TILE_N, (N + COMB_TILE_M - 1) / COMB_TILE_M);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        matmul_combined_fp16_smem_wmma_fused_kernel<<<grid, block>>>(d_A, d_B, d_C, N, N, N);
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
        matmul_combined_fp16_smem_wmma_fused_kernel<<<grid, block>>>(d_A, d_B, d_C, N, N, N);
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
