#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace nvcuda;

// ============================================================
// 4. WMMA TENSOR CORES: Hardware Tensor Core Matrix Multiply
// Copied from: v8_tensorcores/cuda_wmma_master.cu
// Utilizes specialized hardware Tensor Cores for 16x16x16 sub-matrix MAC
// ============================================================

constexpr int WMMA_M_DIM = 16;
constexpr int WMMA_N_DIM = 16;
constexpr int WMMA_K_DIM = 16;

__global__ void matmul_wmma_naive_kernel(const half* __restrict__ A,
                                         const half* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K) {
    int block_row = blockIdx.y * WMMA_M_DIM;
    int block_col = blockIdx.x * WMMA_N_DIM;

    if (block_row >= M || block_col >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K_DIM) {
        const half* a_ptr = A + block_row * K + k;
        wmma::load_matrix_sync(a_frag, a_ptr, K);

        const half* b_ptr = B + k * N + block_col;
        wmma::load_matrix_sync(b_frag, b_ptr, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_ptr = C + block_row * N + block_col;
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}

inline float run_wmma(int N,
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
    dim3 block(32); // 1 warp (32 threads) per WMMA block
    dim3 grid((N + WMMA_N_DIM - 1) / WMMA_N_DIM, (N + WMMA_M_DIM - 1) / WMMA_M_DIM);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        matmul_wmma_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N, N, N);
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
        matmul_wmma_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N, N, N);
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
