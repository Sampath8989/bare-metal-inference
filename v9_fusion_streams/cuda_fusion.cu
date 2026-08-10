// Fused LayerNorm + MatMul + ReLU: one kernel launch vs three.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Memory-bound inference profile: wide normalization, narrow projection.
constexpr int M = 65536;
constexpr int K = 64;
constexpr int N = 64;

constexpr int TILE_M = 16;
constexpr int TILE_N = 64;
constexpr int THREADS = 256;

constexpr int FUSE_TILE_M = 64;

constexpr int WARMUP_RUNS = 20;
constexpr int BENCH_RUNS  = 100;

constexpr float LN_EPS = 1e-5f;

__global__ void kernel_layernorm(const float* __restrict__ X,
                                 float* __restrict__ Y,
                                 int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int base = row * cols;

    float x = (tid < cols) ? X[base + tid] : 0.0f;

    __shared__ float s_sum[K];
    __shared__ float s_sq[K];
    __shared__ float s_mean;
    __shared__ float s_rstd;

    s_sum[tid] = x;
    s_sq[tid]  = x * x;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sq[tid]  += s_sq[tid + stride];
        }
    }
    if (tid == 0) {
        float mean = s_sum[0] / (float)cols;
        float var  = s_sq[0] / (float)cols - mean * mean;
        s_mean = mean;
        s_rstd = 1.0f / sqrtf(var + eps);
    }
    __syncthreads();

    Y[base + tid] = (x - s_mean) * s_rstd;
}

__global__ void kernel_matmul(const float* __restrict__ Y,
                              const float* __restrict__ W,
                              float* __restrict__ Z,
                              int rows, int kdim, int ncols) {
    // Whole K dim staged in smem (20 KB) → 2 blocks/SM.
    __shared__ float sY[TILE_M][K];
    __shared__ float sW[K][TILE_N];

    int block_row = blockIdx.x * TILE_M;
    int tid = threadIdx.x;

    for (int i = tid; i < TILE_M * K; i += THREADS) {
        int r = i >> 6;
        int c = i & 63;
        sY[r][c] = (block_row + r < rows) ? Y[(block_row + r) * kdim + c] : 0.0f;
    }

    for (int i = tid; i < K * TILE_N; i += THREADS) {
        sW[i >> 6][i & 63] = W[(i >> 6) * ncols + (i & 63)];
    }
    __syncthreads();

    int rg = tid >> 5;
    int cg = tid & 31;
    int r0 = rg * 2, c0 = cg * 2;
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int k = 0; k < K; k++) {
        float a0 = sY[r0][k], a1 = sY[r0 + 1][k];
        float b0 = sW[k][c0], b1 = sW[k][c0 + 1];
        acc[0][0] += a0 * b0;  acc[0][1] += a0 * b1;
        acc[1][0] += a1 * b0;  acc[1][1] += a1 * b1;
    }

    int orow = block_row + r0;
    if (orow + 1 < rows) {
        float* z0 = Z + orow * ncols + c0;
        float* z1 = Z + (orow + 1) * ncols + c0;
        z0[0] = acc[0][0];  z0[1] = acc[0][1];
        z1[0] = acc[1][0];  z1[1] = acc[1][1];
    }
}

__global__ void kernel_relu(const float* __restrict__ Z,
                            float* __restrict__ Out,
                            int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) Out[i] = fmaxf(Z[i], 0.0f);
}

__global__ void kernel_fused_layernorm_matmul_relu(
        const float* __restrict__ X,
        const float* __restrict__ W,
        float* __restrict__ Out,
        int rows, int kdim, int ncols, float eps) {

    // 32 KB smem → 2 blocks/SM. LN scratch aliases sW (dead once normalized).
    __shared__ float sX[FUSE_TILE_M][K];
    __shared__ float sW[K][TILE_N];
    float* sPartSum = &sW[0][0];
    float* sPartSq  = sPartSum + THREADS;
    float* sMean    = sPartSq + THREADS;
    float* sRstd    = sMean + FUSE_TILE_M;

    int block_row = blockIdx.x * FUSE_TILE_M;
    int tid = threadIdx.x;
    int rg = tid >> 4;
    int cg = tid & 15;

    for (int i = tid; i < FUSE_TILE_M * K; i += THREADS) {
        int r = i >> 6;
        int c = i & 63;
        sX[r][c] = (block_row + r < rows) ? X[(block_row + r) * kdim + c] : 0.0f;
    }
    __syncthreads();

    int lane = tid & 3;
    int srow = tid >> 2;
    float sum = 0.0f, sq = 0.0f;
    #pragma unroll
    for (int c = lane * 16; c < lane * 16 + 16; c++) {
        float x = sX[srow][c];
        sum += x;
        sq  += x * x;
    }
    sPartSum[tid] = sum;
    sPartSq[tid]  = sq;
    __syncthreads();

    if (tid < FUSE_TILE_M) {
        float tsum = 0.0f, tsq = 0.0f;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            tsum += sPartSum[tid * 4 + c];
            tsq  += sPartSq[tid * 4 + c];
        }
        float mean = tsum / (float)K;
        float var  = tsq / (float)K - mean * mean;
        sMean[tid] = mean;
        sRstd[tid] = 1.0f / sqrtf(var + eps);
    }
    __syncthreads();

    #pragma unroll
    for (int c = lane * 16; c < lane * 16 + 16; c++) {
        sX[srow][c] = (sX[srow][c] - sMean[srow]) * sRstd[srow];
    }
    __syncthreads();

    for (int i = tid; i < K * TILE_N; i += THREADS) {
        sW[i >> 6][i & 63] = W[(i >> 6) * ncols + (i & 63)];
    }
    __syncthreads();

    // 4×4 register tile → DRAM-bound, not LDS-bound.
    int r0 = rg * 4, c0 = cg * 4;
    float acc[4][4] = {{0.0f}};
    for (int k = 0; k < K; k++) {
        float a0 = sX[r0][k],     a1 = sX[r0 + 1][k];
        float a2 = sX[r0 + 2][k], a3 = sX[r0 + 3][k];
        float b0 = sW[k][c0],     b1 = sW[k][c0 + 1];
        float b2 = sW[k][c0 + 2], b3 = sW[k][c0 + 3];
        acc[0][0] += a0 * b0;  acc[0][1] += a0 * b1;  acc[0][2] += a0 * b2;  acc[0][3] += a0 * b3;
        acc[1][0] += a1 * b0;  acc[1][1] += a1 * b1;  acc[1][2] += a1 * b2;  acc[1][3] += a1 * b3;
        acc[2][0] += a2 * b0;  acc[2][1] += a2 * b1;  acc[2][2] += a2 * b2;  acc[2][3] += a2 * b3;
        acc[3][0] += a3 * b0;  acc[3][1] += a3 * b1;  acc[3][2] += a3 * b2;  acc[3][3] += a3 * b3;
    }

    int orow = block_row + r0;
    if (orow + 3 < rows) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            float* o = Out + (orow + r) * ncols + c0;
            o[0] = fmaxf(acc[r][0], 0.0f);
            o[1] = fmaxf(acc[r][1], 0.0f);
            o[2] = fmaxf(acc[r][2], 0.0f);
            o[3] = fmaxf(acc[r][3], 0.0f);
        }
    }
}

void reference_fused(const float* X, const float* W, float* Out,
                     int rows, int kdim, int ncols, float eps) {
    for (int m = 0; m < rows; m++) {
        const float* xrow = X + m * kdim;
        float mean = 0.0f;
        for (int k = 0; k < kdim; k++) mean += xrow[k];
        mean /= (float)kdim;
        float var = 0.0f;
        for (int k = 0; k < kdim; k++) { float d = xrow[k] - mean; var += d * d; }
        var /= (float)kdim;
        float rstd = 1.0f / sqrtf(var + eps);
        for (int n = 0; n < ncols; n++) {
            float s = 0.0f;
            for (int k = 0; k < kdim; k++)
                s += (xrow[k] - mean) * rstd * W[k * ncols + n];
            Out[m * ncols + n] = fmaxf(s, 0.0f);
        }
    }
}

float max_abs_error(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) mx = fmaxf(mx, fabsf(a[i] - b[i]));
    return mx;
}

float bench_p50_us(const std::function<void()>& pipeline, int warmup, int runs) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; i++) pipeline();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> t(runs);
    for (int i = 0; i < runs; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        pipeline();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&t[i], start, stop));
    }
    std::sort(t.begin(), t.end());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return t[runs / 2] * 1000.0f;
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  Kernel Fusion Demo — LayerNorm + MatMul + ReLU (AI Inference)\n");
    printf("  Size: M=%d, K=%d, N=%d   (%.1f MB input)\n", M, K, N, (double)M * K * 4 / 1e6);
    printf("═══════════════════════════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // Max smem carveout so 2 blocks fit per SM (better latency hiding).
    CUDA_CHECK(cudaFuncSetAttribute(kernel_fused_layernorm_matmul_relu,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    const int nX = M * K, nW = K * N, nOut = M * N;

    std::vector<float> h_X(nX), h_W(nW), h_ref(nOut), h_sep(nOut), h_fus(nOut);
    {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (float& v : h_X) v = d(gen);
        for (float& v : h_W) v = d(gen);
    }

    float *d_X, *d_W, *d_Y, *d_Z, *d_sep, *d_fus;
    CUDA_CHECK(cudaMalloc(&d_X,  nX   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W,  nW   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y,  nX   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z,  nOut * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sep, nOut * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_fus, nOut * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, h_X.data(), nX * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), nW * sizeof(float), cudaMemcpyHostToDevice));

    printf("Computing CPU FP32 reference (M·N·K = %.2f GFLOP)...\n", (double)M * N * K / 1e9);
    reference_fused(h_X.data(), h_W.data(), h_ref.data(), M, K, N, LN_EPS);
    printf("  done.\n\n");

    dim3 ln_grid(M),   ln_block(K);
    dim3 mm_grid(M / TILE_M), mm_block(THREADS);
    dim3 relu_grid((nOut + 255) / 256), relu_block(256);
    dim3 fus_grid(M / FUSE_TILE_M), fus_block(THREADS);

    auto pipeline_separate = [&]() {
        kernel_layernorm<<<ln_grid, ln_block>>>(d_X, d_Y, M, K, LN_EPS);
        kernel_matmul<<<mm_grid, mm_block>>>(d_Y, d_W, d_Z, M, K, N);
        kernel_relu<<<relu_grid, relu_block>>>(d_Z, d_sep, nOut);
    };
    auto pipeline_fused = [&]() {
        kernel_fused_layernorm_matmul_relu<<<fus_grid, fus_block>>>(d_X, d_W, d_fus, M, K, N, LN_EPS);
    };

    pipeline_separate();
    pipeline_fused();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_sep.data(), d_sep, nOut * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fus.data(), d_fus, nOut * sizeof(float), cudaMemcpyDeviceToHost));

    float err_sep = max_abs_error(h_sep.data(), h_ref.data(), nOut);
    float err_fus = max_abs_error(h_fus.data(), h_ref.data(), nOut);
    float err_cross = max_abs_error(h_sep.data(), h_fus.data(), nOut);
    printf("─── Correctness (vs FP32 CPU reference) ───\n");
    printf("  Separate pipeline max abs err : %.6f  [%s]\n", err_sep, err_sep < 1e-3f ? "✓ PASS" : "✗ FAIL");
    printf("  Fused pipeline max abs err    : %.6f  [%s]\n", err_fus, err_fus < 1e-3f ? "✓ PASS" : "✗ FAIL");
    printf("  Fused vs Separate max diff    : %.6f\n\n", err_cross);

    float t_sep = bench_p50_us(pipeline_separate, WARMUP_RUNS, BENCH_RUNS);
    float t_fus = bench_p50_us(pipeline_fused,   WARMUP_RUNS, BENCH_RUNS);

    const double bytes_sep = 3.0 * M * K + K * N + 2.0 * M * N;
    const double bytes_fus = (double)M * K + K * N + M * N;

    printf("─── Results (%d warmup + %d timed runs, p50) ───\n\n", WARMUP_RUNS, BENCH_RUNS);
    printf("┌──────────────────────────────┬────────────┬────────────┬─────────────┐\n");
    printf("│ Pipeline                     │ p50 (µs)   │ GB/s       │ DRAM moved  │\n");
    printf("├──────────────────────────────┼────────────┼────────────┼─────────────┤\n");
    printf("│ Separate (3 kernels)         │ %9.1f  │ %9.1f  │ %9.1f MB  │\n",
           t_sep, bytes_sep * 4 / (t_sep * 1e-6) / 1e9, bytes_sep * 4 / 1e6);
    printf("│ Fused   (1 kernel)           │ %9.1f  │ %9.1f  │ %9.1f MB  │\n",
           t_fus, bytes_fus * 4 / (t_fus * 1e-6) / 1e9, bytes_fus * 4 / 1e6);
    printf("├──────────────────────────────┼────────────┼────────────┼─────────────┤\n");
    printf("│ Speedup                      │ %9.2fx   │            │ %-5.1f%%   │\n",
           t_sep / t_fus, (1.0 - bytes_fus / bytes_sep) * 100.0);
    printf("└──────────────────────────────┴────────────┴────────────┴─────────────┘\n\n");

    printf("─── Why fusion wins: global-memory round-trips eliminated ───\n\n");
    printf("  Separate:  X→Y (2·M·K)  +  Y→Z with W (M·K+K·N)  +  Z→Out (2·M·N)\n");
    printf("            = %.1f MB moved through DRAM\n", bytes_sep * 4 / 1e6);
    printf("  Fused:     X→smem (M·K)  +  W→smem (K·N)  +  Out (M·N)\n");
    printf("            = %.1f MB moved through DRAM\n", bytes_fus * 4 / 1e6);
    printf("\n  The LayerNorm'ed matrix Y is kept in shared memory (never written to DRAM).\n");
    printf("  The pre-activation GEMM result Z lives in registers — ReLU is folded into\n");
    printf("  the store epilogue. Two full round-trips and one kernel launch are saved.\n\n");

    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_Y)); CUDA_CHECK(cudaFree(d_Z));
    CUDA_CHECK(cudaFree(d_sep)); CUDA_CHECK(cudaFree(d_fus));
    return 0;
}
