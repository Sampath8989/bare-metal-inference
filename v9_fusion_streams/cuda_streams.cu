// CUDA streams demo: overlap H2D / compute / D2H across two streams.
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

constexpr int BS      = 1024;   // matrix dimension per batch
constexpr int NB      = 6;      // batches; more amortize prologue/tail

constexpr int TILE    = 32;
constexpr int TK      = 16;
constexpr int WARMUP  = 10;
constexpr int RUNS    = 30;

constexpr size_t BYTES = (size_t)BS * BS * sizeof(float);

__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int n) {
    __shared__ float sA[TILE][TK];
    __shared__ float sB[TK][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * 16 + tx;
    int row0 = blockIdx.y * TILE + ty * 2;
    int col0 = blockIdx.x * TILE + tx * 2;

    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int k0 = 0; k0 < n; k0 += TK) {

        for (int i = tid; i < TILE * TK; i += 256) {
            sA[i / TK][i % TK] = A[(blockIdx.y * TILE + i / TK) * n + k0 + i % TK];
        }

        for (int i = tid; i < TK * TILE; i += 256) {
            sB[i / TILE][i % TILE] = B[(k0 + i / TILE) * n + blockIdx.x * TILE + i % TILE];
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TK; kk++) {
            float a0 = sA[ty * 2][kk];
            float a1 = sA[ty * 2 + 1][kk];
            float b0 = sB[kk][tx * 2];
            float b1 = sB[kk][tx * 2 + 1];
            acc[0][0] += a0 * b0;  acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0;  acc[1][1] += a1 * b1;
        }
        __syncthreads();
    }

    C[row0 * n + col0]     = acc[0][0];
    C[row0 * n + col0 + 1] = acc[0][1];
    C[(row0 + 1) * n + col0]     = acc[1][0];
    C[(row0 + 1) * n + col0 + 1] = acc[1][1];
}

float bench_p50(cudaStream_t start_stream, cudaStream_t stop_stream,
                const std::function<void()>& pipeline) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP; i++) pipeline();
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> t(RUNS);
    for (int i = 0; i < RUNS; i++) {
        // Pipeline's own stream: the default stream implicitly syncs all others.
        CUDA_CHECK(cudaEventRecord(start, start_stream));
        pipeline();
        CUDA_CHECK(cudaEventRecord(stop, stop_stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&t[i], start, stop));
    }
    std::sort(t.begin(), t.end());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return t[RUNS / 2] * 1000.0f;
}

int main() {
    printf("═══════════════════════════════════════════════════════════════════════════\n");
    printf("  CUDA Streams Demo — Overlapping H2D / Compute / D2H\n");
    printf("  %d batches of %d×%d GEMM (%.1f MB per batch)\n", NB, BS, BS, 3.0 * BYTES / 1e6);
    printf("═══════════════════════════════════════════════════════════════════════════\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Pinned host memory: required for cudaMemcpyAsync to overlap with kernels.\n\n");

    float *hA[NB], *hB[NB], *hC[NB];
    float *dA[NB], *dB[NB], *dC[NB];

    {
        std::mt19937 gen(7);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        for (int b = 0; b < NB; b++) {
            CUDA_CHECK(cudaMallocHost(&hA[b], BYTES));   // pinned: required for Async overlap
            CUDA_CHECK(cudaMallocHost(&hB[b], BYTES));
            CUDA_CHECK(cudaMallocHost(&hC[b], BYTES));
            for (size_t i = 0; i < BS * BS; i++) { hA[b][i] = d(gen); hB[b][i] = d(gen); }
            CUDA_CHECK(cudaMalloc(&dA[b], BYTES));
            CUDA_CHECK(cudaMalloc(&dB[b], BYTES));
            CUDA_CHECK(cudaMalloc(&dC[b], BYTES));
        }
    }

    dim3 grid(BS / TILE, BS / TILE);
    dim3 block(16, 16);

    cudaStream_t s_seq, s1, s2;
    CUDA_CHECK(cudaStreamCreate(&s_seq));
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaStreamCreate(&s2));

    auto pipeline_sequential = [&]() {
        for (int b = 0; b < NB; b++) {
            CUDA_CHECK(cudaMemcpyAsync(dA[b], hA[b], BYTES, cudaMemcpyHostToDevice, s_seq));
            CUDA_CHECK(cudaMemcpyAsync(dB[b], hB[b], BYTES, cudaMemcpyHostToDevice, s_seq));
            matmul_kernel<<<grid, block, 0, s_seq>>>(dA[b], dB[b], dC[b], BS);
            CUDA_CHECK(cudaMemcpyAsync(hC[b], dC[b], BYTES, cudaMemcpyDeviceToHost, s_seq));
        }
    };

    cudaEvent_t h2d_ev[NB], kern_ev[NB];
    for (int b = 0; b < NB; b++) {
        CUDA_CHECK(cudaEventCreate(&h2d_ev[b]));
        CUDA_CHECK(cudaEventCreate(&kern_ev[b]));
    }

    auto pipeline_overlapped = [&]() {
        // Prologue: H2D of batch 0 on stream2.
        CUDA_CHECK(cudaMemcpyAsync(dA[0], hA[0], BYTES, cudaMemcpyHostToDevice, s2));
        CUDA_CHECK(cudaMemcpyAsync(dB[0], hB[0], BYTES, cudaMemcpyHostToDevice, s2));
        CUDA_CHECK(cudaEventRecord(h2d_ev[0], s2));

        for (int b = 0; b < NB; b++) {
            // s1: GEMM(batch b) once its H2D lands.
            CUDA_CHECK(cudaStreamWaitEvent(s1, h2d_ev[b], 0));
            matmul_kernel<<<grid, block, 0, s1>>>(dA[b], dB[b], dC[b], BS);
            CUDA_CHECK(cudaEventRecord(kern_ev[b], s1));

            if (b + 1 < NB) {
                // s2: prefetch batch b+1 while s1 computes batch b.
                CUDA_CHECK(cudaMemcpyAsync(dA[b + 1], hA[b + 1], BYTES, cudaMemcpyHostToDevice, s2));
                CUDA_CHECK(cudaMemcpyAsync(dB[b + 1], hB[b + 1], BYTES, cudaMemcpyHostToDevice, s2));
                CUDA_CHECK(cudaEventRecord(h2d_ev[b + 1], s2));
            }

            CUDA_CHECK(cudaStreamWaitEvent(s2, kern_ev[b], 0));
            CUDA_CHECK(cudaMemcpyAsync(hC[b], dC[b], BYTES, cudaMemcpyDeviceToHost, s2));   // s2: drain output
        }
    };

    pipeline_sequential();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> refC(NB * BS * BS);
    for (int b = 0; b < NB; b++)
        std::copy(hC[b], hC[b] + BS * BS, refC.data() + (size_t)b * BS * BS);

    pipeline_overlapped();
    CUDA_CHECK(cudaDeviceSynchronize());

    float max_diff = 0.0f;
    for (int b = 0; b < NB; b++)
        for (size_t i = 0; i < (size_t)BS * BS; i++)
            max_diff = fmaxf(max_diff, fabsf(hC[b][i] - refC[(size_t)b * BS * BS + i]));
    printf("─── Correctness ───\n");
    printf("  Overlapped vs Sequential max diff: %.6f  [%s]\n\n",
           max_diff, max_diff < 1e-6f ? "✓ PASS (identical)" : "✗ FAIL");

    float t_seq = bench_p50(s_seq, s_seq, pipeline_sequential);
    float t_ovl = bench_p50(s2,    s2,    pipeline_overlapped);

    const double bytes_per_batch = 3.0 * BYTES;
    const double total_bytes = bytes_per_batch * NB;

    printf("─── Results (%d warmup + %d timed runs, p50) ───\n\n", WARMUP, RUNS);
    printf("┌──────────────────────────────┬────────────┬────────────┐\n");
    printf("│ Execution mode               │ p50 (µs)   │ GB/s moved │\n");
    printf("├──────────────────────────────┼────────────┼────────────┤\n");
    printf("│ Sequential (1 stream)        │ %9.1f  │ %9.1f  │\n",
           t_seq, total_bytes / (t_seq * 1e-6) / 1e9);
    printf("│ Overlapped (2 streams)       │ %9.1f  │ %9.1f  │\n",
           t_ovl, total_bytes / (t_ovl * 1e-6) / 1e9);
    printf("├──────────────────────────────┼────────────┼────────────┤\n");
    printf("│ Speedup                      │ %9.2fx   │            │\n", t_seq / t_ovl);
    printf("└──────────────────────────────┴────────────┴────────────┘\n\n");

    printf("─── How the overlap works ───\n\n");
    printf("  Sequential:  %d batches × (H2D + kernel + D2H)  — SMs idle during copies,\n", NB);
    printf("               copy engines idle during compute.\n");
    printf("  Overlapped:  stream2 prefetches batch %d's inputs while stream1 computes\n",
           NB > 1 ? 2 : 1);
    printf("               batch 1 and drains batch 1's output — copies hidden under\n");
    printf("               compute. Events (cudaStreamWaitEvent) keep the dependency\n");
    printf("               order: kernel(b) → after H2D(b),  D2H(b) → after kernel(b).\n\n");
    printf("  Theoretical: sequential ≈ NB·(H2D+kernel+D2H);  overlapped ≈ H2D(0) +\n");
    printf("               NB·max(kernel, H2D+D2H pipeline). The win grows with the\n");
    printf("               number of batches.\n\n");

    for (int b = 0; b < NB; b++) {
        CUDA_CHECK(cudaFreeHost(hA[b])); CUDA_CHECK(cudaFreeHost(hB[b])); CUDA_CHECK(cudaFreeHost(hC[b]));
        CUDA_CHECK(cudaFree(dA[b]));    CUDA_CHECK(cudaFree(dB[b]));    CUDA_CHECK(cudaFree(dC[b]));
        CUDA_CHECK(cudaEventDestroy(h2d_ev[b]));
        CUDA_CHECK(cudaEventDestroy(kern_ev[b]));
    }
    CUDA_CHECK(cudaStreamDestroy(s_seq));
    CUDA_CHECK(cudaStreamDestroy(s1));
    CUDA_CHECK(cudaStreamDestroy(s2));
    return 0;
}
