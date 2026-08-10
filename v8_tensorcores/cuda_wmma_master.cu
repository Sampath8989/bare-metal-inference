// FP16 WMMA (Tensor Core) GEMM: global-load vs shared-memory tiled (T4, sm_75).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>

using namespace nvcuda;

constexpr int WMMA_M = 16;   // fragment dims are fixed 16×16×16 on sm_75
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int TILE_M = 64;   // must be multiples of the fragment dims
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;

constexpr int SMEM_PAD = 8;   // 24-wide rows break 32-byte bank-stride alignment

constexpr int SMEM_A_STRIDE = TILE_K + SMEM_PAD;
constexpr int SMEM_B_STRIDE = TILE_N + SMEM_PAD;   // 72, not 24: fragment rows are 72 apart

constexpr int WARMUP_RUNS = 20;
constexpr int BENCH_RUNS = 200;

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__global__ void matmul_naive_fp32(const float* __restrict__ A,
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
        C[row * N + col] = sum;
    }
}

__global__ void matmul_wmma_naive(const half* __restrict__ A,
                                   const half* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {

    int block_row = blockIdx.y * WMMA_M;
    int block_col = blockIdx.x * WMMA_N;

    if (block_row >= M || block_col >= N) return;

    // Both matrices are row-major; ldm is the row stride (K for A, N for B).
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {

        const half* a_ptr = A + block_row * K + k;
        wmma::load_matrix_sync(a_frag, a_ptr, K);

        const half* b_ptr = B + k * N + block_col;
        wmma::load_matrix_sync(b_frag, b_ptr, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_ptr = C + block_row * N + block_col;
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}

__global__ void matmul_wmma_shared(const half* __restrict__ A,
                                    const half* __restrict__ B,
                                    float* __restrict__ C,
                                    int M, int N, int K) {

    __shared__ half smem_A[TILE_M][TILE_K + SMEM_PAD];
    __shared__ half smem_B[TILE_K][TILE_N + SMEM_PAD];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int block_row = blockIdx.y * TILE_M;
    int block_col = blockIdx.x * TILE_N;

    // WMMA requires all 32 lanes on the same fragment: warp w → tiles w and w+8.
    constexpr int TILES_PER_WARP = (TILE_M / WMMA_M) * (TILE_N / WMMA_N) / 8;
    int tile_row[TILES_PER_WARP];
    int tile_col[TILES_PER_WARP];
    #pragma unroll
    for (int t = 0; t < TILES_PER_WARP; t++) {
        int tile_idx = warp_id + t * 8;
        tile_row[t] = tile_idx / 4;
        tile_col[t] = tile_idx % 4;
    }

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[TILES_PER_WARP];
    #pragma unroll
    for (int t = 0; t < TILES_PER_WARP; t++) {
        wmma::fill_fragment(c_frag[t], 0.0f);
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[TILES_PER_WARP];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[TILES_PER_WARP];

    constexpr int TOTAL_LOADS_A = TILE_M * TILE_K;
    constexpr int TOTAL_LOADS_B = TILE_K * TILE_N;
    constexpr int NTHREADS = 256;

    // Reload smem every K-step; the k offset advances here.
    for (int k = 0; k < K; k += WMMA_K) {

        #pragma unroll
        for (int i = tid; i < TOTAL_LOADS_A; i += NTHREADS) {
            int smem_row = i / TILE_K;
            int smem_col = i % TILE_K;
            int global_row = block_row + smem_row;
            int global_col = k + smem_col;

            half val = (global_row < M && global_col < K)
                           ? A[global_row * K + global_col]
                           : __float2half(0.0f);
            smem_A[smem_row][smem_col] = val;

        }

        #pragma unroll
        for (int i = tid; i < TOTAL_LOADS_B; i += NTHREADS) {
            int smem_row = i / TILE_N;
            int smem_col = i % TILE_N;
            int global_row = k + smem_row;
            int global_col = block_col + smem_col;

            half val = (global_row < K && global_col < N)
                           ? B[global_row * N + global_col]
                           : __float2half(0.0f);
            smem_B[smem_row][smem_col] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILES_PER_WARP; t++) {

            half* a_ptr = &smem_A[tile_row[t] * WMMA_M][0];
            wmma::load_matrix_sync(a_frag[t], a_ptr, SMEM_A_STRIDE);

            half* b_ptr = &smem_B[0][tile_col[t] * WMMA_N];
            wmma::load_matrix_sync(b_frag[t], b_ptr, SMEM_B_STRIDE);

            wmma::mma_sync(c_frag[t], a_frag[t], b_frag[t], c_frag[t]);
        }

        __syncthreads();
    }

    #pragma unroll
    for (int t = 0; t < TILES_PER_WARP; t++) {
        int out_row = block_row + tile_row[t] * WMMA_M;
        int out_col = block_col + tile_col[t] * WMMA_N;

        if (out_row < M && out_col < N) {
            float* c_ptr = C + out_row * N + out_col;
            wmma::store_matrix_sync(c_ptr, c_frag[t], N, wmma::mem_row_major);
        }
    }
}

void rand_matrix_fp32(float* ptr, int rows, int cols, unsigned seed) {
    // Small ±0.5 values: far from FP16 overflow and keeps FP16 rounding tiny.
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < rows * cols; i++) {
        ptr[i] = dist(gen);
    }
}

void fp32_to_fp16(const float* src, half* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = __float2half(src[i]);
    }
}

void reference_matmul_fp32(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

float max_abs_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

float max_rel_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    float max_val = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
        float val = fabsf(b[i]);
        if (val > max_val) max_val = val;
    }
    return (max_val > 0.0f) ? (max_err / max_val) : 0.0f;
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  CUDA WMMA Master — FP16 Tensor Core GEMM Benchmark\n");
    printf("  Target: NVIDIA T4 (sm_75, Turing) — 65 TFLOPS FP16 Tensor Core Peak\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Shared Memory per Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Shared Memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("\n");

    int test_sizes[][3] = {
        {512, 512, 512},
        {1024, 1024, 1024}
    };
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_tests; t++) {
        int M = test_sizes[t][0];
        int N = test_sizes[t][1];
        int K = test_sizes[t][2];

        printf("═══════════════════════════════════════════════════════════════════════════\n");
        printf("  Matrix Size: %d × %d × %d\n", M, N, K);
        printf("  Total FP16 elements: A=%d, B=%d, C=%d\n", M*K, K*N, M*N);
        printf("  Memory: A=%.1f MB, B=%.1f MB, C=%.1f MB\n",
               (float)M*K*2/1e6, (float)K*N*2/1e6, (float)M*N*4/1e6);
        printf("═══════════════════════════════════════════════════════════════════════════\n\n");

        std::vector<float> h_A_fp32(M * K);
        std::vector<float> h_B_fp32(K * N);
        std::vector<float> h_C_ref(M * N);
        std::vector<float> h_C_gpu(M * N);

        rand_matrix_fp32(h_A_fp32.data(), M, K, 42);
        rand_matrix_fp32(h_B_fp32.data(), K, N, 1337);

        std::vector<half> h_A_fp16(M * K);
        std::vector<half> h_B_fp16(K * N);
        fp32_to_fp16(h_A_fp32.data(), h_A_fp16.data(), M * K);
        fp32_to_fp16(h_B_fp32.data(), h_B_fp16.data(), K * N);

        printf("Computing reference FP32 result on CPU...\n");
        reference_matmul_fp32(h_A_fp32.data(), h_B_fp32.data(), h_C_ref.data(), M, N, K);
        printf("  Reference computed.\n\n");

        float* d_A_fp32;
        float* d_B_fp32;
        float* d_C_fp32;
        half*  d_A_fp16;
        half*  d_B_fp16;
        float* d_C_wmma;

        CUDA_CHECK(cudaMalloc(&d_A_fp32, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B_fp32, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C_fp32, M * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_A_fp16, M * K * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B_fp16, K * N * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C_wmma, M * N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A_fp32, h_A_fp32.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_fp32, h_B_fp32.data(), K*N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_A_fp16, h_A_fp16.data(), M*K*sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_fp16, h_B_fp16.data(), K*N*sizeof(half), cudaMemcpyHostToDevice));

        printf("─── Kernel 1: matmul_naive_fp32 ───\n");
        {
            dim3 block(16, 16);
            dim3 grid((N + 15) / 16, (M + 15) / 16);
            matmul_naive_fp32<<<grid, block>>>(d_A_fp32, d_B_fp32, d_C_fp32, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
            float err = max_abs_error(h_C_gpu.data(), h_C_ref.data(), M*N);
            printf("  Max abs error vs reference: %.6f\n", err);
            printf("  Status: %s\n\n", err < 1e-3f ? "✓ PASS" : "✗ FAIL");
        }

        printf("─── Kernel 2: matmul_wmma_naive ───\n");
        {
            dim3 block(32);   // single warp per block: WMMA is warp-cooperative
            dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
            matmul_wmma_naive<<<grid, block>>>(d_A_fp16, d_B_fp16, d_C_wmma, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C_wmma, M*N*sizeof(float), cudaMemcpyDeviceToHost));
            float err = max_abs_error(h_C_gpu.data(), h_C_ref.data(), M*N);
            float rel = max_rel_error(h_C_gpu.data(), h_C_ref.data(), M*N);
            printf("  Max abs error vs FP32 ref: %.6f\n", err);
            printf("  Max rel error vs FP32 ref: %.6f (%.2f%%)\n", rel, rel * 100.0f);
            printf("  Status: %s\n\n", err < 0.1f ? "✓ PASS" : "✗ FAIL");
        }

        printf("─── Kernel 3: matmul_wmma_shared (MASTER) ───\n");
        {

            dim3 block(16, 16);
            dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
            matmul_wmma_shared<<<grid, block>>>(d_A_fp16, d_B_fp16, d_C_wmma, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C_wmma, M*N*sizeof(float), cudaMemcpyDeviceToHost));
            float err = max_abs_error(h_C_gpu.data(), h_C_ref.data(), M*N);
            float rel = max_rel_error(h_C_gpu.data(), h_C_ref.data(), M*N);
            printf("  Max abs error vs FP32 ref: %.6f\n", err);
            printf("  Max rel error vs FP32 ref: %.6f (%.2f%%)\n", rel, rel * 100.0f);
            printf("  Status: %s\n\n", err < 0.1f ? "✓ PASS" : "✗ FAIL");
        }

        printf("─── Benchmarking (%d warmup + %d timed runs) ───\n\n", WARMUP_RUNS, BENCH_RUNS);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        struct KernelBench {
            const char* name;
            float p50_us;
            float tflops;
        };

        std::vector<KernelBench> results;

        {
            dim3 block(16, 16);
            dim3 grid((N + 15) / 16, (M + 15) / 16);

            for (int i = 0; i < WARMUP_RUNS; i++) {
                matmul_naive_fp32<<<grid, block>>>(d_A_fp32, d_B_fp32, d_C_fp32, M, N, K);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> times(BENCH_RUNS);
            for (int i = 0; i < BENCH_RUNS; i++) {
                CUDA_CHECK(cudaEventRecord(start));
                matmul_naive_fp32<<<grid, block>>>(d_A_fp32, d_B_fp32, d_C_fp32, M, N, K);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
            }
            std::sort(times.begin(), times.end());
            float p50_ms = times[BENCH_RUNS / 2];
            float p50_us = p50_ms * 1000.0f;
            double flops = 2.0 * M * N * K;
            double tflops = flops / (p50_us * 1e6);
            results.push_back({"matmul_naive_fp32", p50_us, (float)tflops});
        }

        {
            dim3 block(32);
            dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);

            for (int i = 0; i < WARMUP_RUNS; i++) {
                matmul_wmma_naive<<<grid, block>>>(d_A_fp16, d_B_fp16, d_C_wmma, M, N, K);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> times(BENCH_RUNS);
            for (int i = 0; i < BENCH_RUNS; i++) {
                CUDA_CHECK(cudaEventRecord(start));
                matmul_wmma_naive<<<grid, block>>>(d_A_fp16, d_B_fp16, d_C_wmma, M, N, K);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
            }
            std::sort(times.begin(), times.end());
            float p50_ms = times[BENCH_RUNS / 2];
            float p50_us = p50_ms * 1000.0f;
            double flops = 2.0 * M * N * K;
            double tflops = flops / (p50_us * 1e6);
            results.push_back({"matmul_wmma_naive", p50_us, (float)tflops});
        }

        {
            dim3 block(16, 16);
            dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

            for (int i = 0; i < WARMUP_RUNS; i++) {
                matmul_wmma_shared<<<grid, block>>>(d_A_fp16, d_B_fp16, d_C_wmma, M, N, K);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> times(BENCH_RUNS);
            for (int i = 0; i < BENCH_RUNS; i++) {
                CUDA_CHECK(cudaEventRecord(start));
                matmul_wmma_shared<<<grid, block>>>(d_A_fp16, d_B_fp16, d_C_wmma, M, N, K);
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));
                CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
            }
            std::sort(times.begin(), times.end());
            float p50_ms = times[BENCH_RUNS / 2];
            float p50_us = p50_ms * 1000.0f;
            double flops = 2.0 * M * N * K;
            double tflops = flops / (p50_us * 1e6);
            results.push_back({"matmul_wmma_shared", p50_us, (float)tflops});
        }

        printf("┌──────────────────────┬──────────────┬──────────────┐\n");
        printf("│ Kernel               │ p50 (µs)     │ TFLOPS       │\n");
        printf("├──────────────────────┼──────────────┼──────────────┤\n");
        for (const auto& r : results) {
            printf("│ %-20s │ %10.1f   │ %10.2f    │\n", r.name, r.p50_us, r.tflops);
        }
        printf("└──────────────────────┴──────────────┴──────────────┘\n");
        printf("\n");

        printf("─── Performance Analysis ───\n\n");
        printf("  T4 Theoretical Peak (FP16 Tensor Core): 65.0 TFLOPS\n");
        printf("  T4 Theoretical Peak (FP32 CUDA Core):    8.1 TFLOPS\n\n");

        float naive_tflops = results[0].tflops;
        float wmma_naive_tflops = results[1].tflops;
        float wmma_shared_tflops = results[2].tflops;

        printf("  matmul_naive_fp32:    %6.2f TFLOPS  (%5.1f%% of FP32 peak)\n",
               naive_tflops, naive_tflops / 8.1f * 100.0f);
        printf("  matmul_wmma_naive:    %6.2f TFLOPS  (%5.1f%% of FP16 TC peak)\n",
               wmma_naive_tflops, wmma_naive_tflops / 65.0f * 100.0f);
        printf("  matmul_wmma_shared:   %6.2f TFLOPS  (%5.1f%% of FP16 TC peak)\n",
               wmma_shared_tflops, wmma_shared_tflops / 65.0f * 100.0f);

        printf("\n  Speedup (WMMA Shared vs Naive FP32): %.1fx\n",
               wmma_shared_tflops / naive_tflops);
        printf("  Speedup (WMMA Shared vs WMMA Naive): %.1fx\n",
               wmma_shared_tflops / wmma_naive_tflops);

        if (wmma_shared_tflops > 30.0f) {
            printf("\n  ✓ Excellent! Achieving >46%% of T4 Tensor Core peak.\n");
        } else if (wmma_shared_tflops > 15.0f) {
            printf("\n  ✓ Good. Achieving >23%% of T4 Tensor Core peak.\n");
        } else if (wmma_shared_tflops > 5.0f) {
            printf("\n  △ Moderate. Consider further optimizations (larger tiles, register blocking).\n");
        } else {
            printf("\n  ✗ Below expectations. Check GPU utilization and memory bandwidth.\n");
        }
        printf("\n");

        CUDA_CHECK(cudaFree(d_A_fp32));
        CUDA_CHECK(cudaFree(d_B_fp32));
        CUDA_CHECK(cudaFree(d_C_fp32));
        CUDA_CHECK(cudaFree(d_A_fp16));
        CUDA_CHECK(cudaFree(d_B_fp16));
        CUDA_CHECK(cudaFree(d_C_wmma));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  All benchmarks complete.\n");
    printf("═══════════════════════════════════════════════════════════════════════════\n");

    return 0;
}
