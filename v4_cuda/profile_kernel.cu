/*
 * v4_cuda/profile_kernel.cu
 *
 * Month 3 — Track A, Prompt 4
 * Minimal standalone CUDA file that ONLY launches the shared-memory matmul kernel
 * in a tight loop. No file I/O, no CPU comparison, no benchmarking prints.
 * Exists purely so NVIDIA NSight Compute has a clean, isolated kernel launch.
 *
 * Build:  nvcc -O2 -arch=sm_75 v4_cuda/profile_kernel.cu -o profile_kernel
 * Profile: ncu --set full -o profile ./profile_kernel
 */

#include <cuda_runtime.h>
#include <cstdint>

#define TILE_SIZE 16

// Shared-memory tiled int8 matmul kernel
__global__ void matmul_int8_shared(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    __shared__ int8_t tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t tile_B[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int32_t accum = 0;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        tile_A[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
        tile_B[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0;
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < TILE_SIZE; q++)
            accum += static_cast<int32_t>(tile_A[threadIdx.y][q]) * static_cast<int32_t>(tile_B[q][threadIdx.x]);
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = accum;
}

int main() {
    const int M = 1, K = 128, N = 64;
    int8_t *d_A, *d_B;
    int32_t* d_C;
    cudaMalloc(&d_A, M * K);
    cudaMalloc(&d_B, K * N);
    cudaMalloc(&d_C, M * N * sizeof(int32_t));

    // Initialize with dummy data (fixed, no random needed)
    int8_t h_A[128]; int8_t h_B[8192];
    for (int i = 0; i < 128; i++) h_A[i] = 1;
    for (int i = 0; i < 8192; i++) h_B[i] = 1;
    cudaMemcpy(d_A, h_A, 128, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, 8192, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Tight loop — 100 kernel launches for profiler to capture
    for (int i = 0; i < 100; i++) {
        matmul_int8_shared<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    cudaDeviceSynchronize();
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
