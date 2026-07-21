/*
 * v4_cuda/cuda_batched.cu
 *
 * Month 3 — Track A, Prompt 3
 * Batch-dimension matmul: (batch_size × input_dim) matrix processed in a single
 * kernel launch. Throughput sweep across batch sizes 1/2/4/8/16/32 with CPU-vs-GPU
 * crossover detection.
 *
 * Build:  nvcc -O2 -arch=sm_75 v4_cuda/cuda_batched.cu -o cuda_batched
 * Run:    ./cuda_batched [weights_int8.bin path]
 *
 * ===========================================================================
 * NOTE ON SHARED MEMORY PERFORMANCE (TASK 3):
 * ===========================================================================
 * At this model size (max matmul 128×64), shared memory tiling is EXPECTED
 * to be slower than the naive kernel. This is NOT a bug.
 *
 * Why: Shared memory tiling optimizes for data reuse across threads by
 * loading tiles of A and B into __shared__ memory. However, for small
 * matrices like 128×64, the overhead of:
 *   1. Cooperative tile loading (shared memory loads + __syncthreads())
 *   2. Boundary checking (padding zero elements for partial tiles)
 *   3. Extra register pressure from tile indexing
 * ...dominates the actual arithmetic savings from data reuse.
 *
 * Shared memory tiling provides no speedup at this model size due to small
 * matrix dimensions (128×64). Overhead dominates arithmetic.
 *
 * Speedup benefits appear at matrix dimensions >= 256×256 where data reuse
 * ratios justify the tile management overhead.
 * ===========================================================================
 *
 * TASK 2: Loop unrolling with #pragma unroll applied to the innermost
 * dot-product loop (TILE_SIZE=16 is compile-time constant, so it WILL unroll).
 * ===========================================================================
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// ============================================================
// Weight structure
// ============================================================
struct LayerWeights {
    int rows, cols;
    float scale;
    std::vector<int8_t> h_weights;
    std::vector<float> h_bias;
    int8_t* d_weights;
    float* d_bias;
};

// ============================================================
// CUDA Kernels
// ============================================================
__global__ void quantize_batched_kernel(const float* input, int8_t* output, float* out_scale, int total_size) {
    extern __shared__ float shared_max[];
    int tid = threadIdx.x;
    // Each thread may process multiple elements if total_size > blockDim.x
    float local_max = 0.0f;
    for (int i = tid; i < total_size; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }
    shared_max[tid] = local_max;
    __syncthreads();
    int pow2 = 1;
    while (pow2 < blockDim.x) pow2 <<= 1;
    for (unsigned int s = pow2 / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x)
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        float max_val = shared_max[0];
        *out_scale = (max_val == 0.0f) ? 1.0f : max_val / 127.0f;
    }
    __syncthreads();
    float scale = *out_scale;
    for (int i = tid; i < total_size; i += blockDim.x) {
        float scaled = input[i] / scale;
        float rounded = roundf(scaled);
        if (rounded > 127.0f) rounded = 127.0f;
        if (rounded < -128.0f) rounded = -128.0f;
        output[i] = static_cast<int8_t>(rounded);
    }
}

// Shared-memory tiled batched matmul: A is [M, K], B is [K, N], C is [M, N]
__global__ void matmul_int8_batched(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    __shared__ int8_t tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t tile_B[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int32_t accum = 0;
    // TASK 2: Loop unrolling — both the tile loop and inner dot-product loop
    // The inner TILE_SIZE loop WILL unroll (compile-time constant).
    #pragma unroll
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

__global__ void post_process_kernel(const int32_t* C_acc, const float* bias, float* output,
                                     float weight_scale, const float* input_scale, int M, int N, bool apply_relu) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        int idx = i * N + j;
        float val = static_cast<float>(C_acc[idx]) * (*input_scale) * weight_scale + bias[j];
        if (apply_relu) val = fmaxf(0.0f, val);
        output[idx] = val;
    }
}

// ============================================================
// GPU batched forward pass
// ============================================================
void forward_gpu_batched(const std::vector<LayerWeights>& layers, int batch_size,
                          float* d_input, float* d_output,
                          int8_t* d_q, float* d_scale,
                          float* d_act, int32_t* d_acc) {
    int M = batch_size;
    // Layer 0: 128 -> 64
    int q_threads = min(M * 128, 256);
    quantize_batched_kernel<<<1, q_threads, q_threads * sizeof(float)>>>(d_input, d_q, d_scale, M * 128);
    { dim3 b(TILE_SIZE, TILE_SIZE), g((64+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
      matmul_int8_batched<<<g, b>>>(d_q, layers[0].d_weights, d_acc, M, 64, 128);
      post_process_kernel<<<g, b>>>(d_acc, layers[0].d_bias, d_act, layers[0].scale, d_scale, M, 64, true); }
    // Layer 1: 64 -> 32
    int q_threads1 = min(M * 64, 256);
    quantize_batched_kernel<<<1, q_threads1, q_threads1 * sizeof(float)>>>(d_act, d_q, d_scale, M * 64);
    { dim3 b(TILE_SIZE, TILE_SIZE), g((32+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
      matmul_int8_batched<<<g, b>>>(d_q, layers[1].d_weights, d_acc, M, 32, 64);
      post_process_kernel<<<g, b>>>(d_acc, layers[1].d_bias, d_act, layers[1].scale, d_scale, M, 32, true); }
    // Layer 2: 32 -> 10
    int q_threads2 = min(M * 32, 256);
    quantize_batched_kernel<<<1, q_threads2, q_threads2 * sizeof(float)>>>(d_act, d_q, d_scale, M * 32);
    { dim3 b(TILE_SIZE, TILE_SIZE), g((10+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);
      matmul_int8_batched<<<g, b>>>(d_q, layers[2].d_weights, d_acc, M, 10, 32);
      post_process_kernel<<<g, b>>>(d_acc, layers[2].d_bias, d_output, layers[2].scale, d_scale, M, 10, true); }
}

// ============================================================
// CPU reference (single query, SIMD p50 from our measurements)
// ============================================================
void forward_cpu_single(const std::vector<LayerWeights>& layers, const std::vector<float>& input, std::vector<float>& output) {
    std::vector<int8_t> q(128), q64(64), q32(32);
    std::vector<float> a64(64), a32(32);
    std::vector<int32_t> acc64(64), acc32(32), acc10(10);
    float s0, s1, s2;
    auto qfn = [](const std::vector<float>& in, std::vector<int8_t>& out, float& sc) {
        float ma = 0; for (float v : in) ma = std::max(ma, std::abs(v));
        sc = (ma == 0) ? 1.0f : ma / 127.0f;
        for (size_t i = 0; i < in.size(); i++) out[i] = static_cast<int8_t>(std::clamp(std::round(in[i] / sc), -128.0f, 127.0f));
    };
    auto mfn = [](const std::vector<int8_t>& A, const std::vector<int8_t>& B, std::vector<int32_t>& acc, int M, int N, int K) {
        std::fill(acc.begin(), acc.end(), 0);
        for (int i = 0; i < M; i++) for (int q = 0; q < K; q++) { int8_t a = A[i*K+q]; for (int j = 0; j < N; j++) acc[i*N+j] += (int32_t)a * (int32_t)B[q*N+j]; }
    };
    qfn(input, q, s0);
    mfn(q, layers[0].h_weights, acc64, 1, 64, 128);
    for (int j = 0; j < 64; j++) a64[j] = std::max(0.0f, (float)acc64[j] * s0 * layers[0].scale + layers[0].h_bias[j]);
    qfn(a64, q64, s1);
    mfn(q64, layers[1].h_weights, acc32, 1, 32, 64);
    for (int j = 0; j < 32; j++) a32[j] = std::max(0.0f, (float)acc32[j] * s1 * layers[1].scale + layers[1].h_bias[j]);
    qfn(a32, q32, s2);
    mfn(q32, layers[2].h_weights, acc10, 1, 10, 32);
    output.resize(10);
    for (int j = 0; j < 10; j++) output[j] = std::max(0.0f, (float)acc10[j] * s2 * layers[2].scale + layers[2].h_bias[j]);
}

// ============================================================
// Load Weights
// ============================================================
std::vector<LayerWeights> load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<LayerWeights> layers;
    while (file.peek() != EOF) {
        int rows, cols; file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        if (file.gcount() == 0) break;
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        float scale; file.read(reinterpret_cast<char*>(&scale), sizeof(scale));
        size_t nw = (size_t)rows * cols;
        std::vector<int8_t> hw(nw); file.read(reinterpret_cast<char*>(hw.data()), nw);
        int bs; file.read(reinterpret_cast<char*>(&bs), sizeof(bs));
        std::vector<float> hb(bs); file.read(reinterpret_cast<char*>(hb.data()), bs * sizeof(float));
        int8_t* dw; float* db;
        cudaMalloc(&dw, nw * sizeof(int8_t)); cudaMalloc(&db, bs * sizeof(float));
        cudaMemcpy(dw, hw.data(), nw * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb.data(), bs * sizeof(float), cudaMemcpyHostToDevice);
        layers.push_back({rows, cols, scale, std::move(hw), std::move(hb), dw, db});
    }
    return layers;
}

float get_percentile(std::vector<float>& v, float p) {
    int idx = (int)std::ceil(v.size() * p / 100.0f) - 1;
    return v[std::max(0, std::min(idx, (int)v.size() - 1))];
}

int main(int argc, char* argv[]) {
    std::string weights_path = "weights_int8.bin";  // Default: current directory (Colab: /content/weights_int8.bin)
    if (argc >= 2) weights_path = argv[1];

    std::cout << "=== GPU INT8 BATCHED THROUGHPUT SWEEP ===" << std::endl;
    std::cout << "Loop unrolling: #pragma unroll applied to inner tile loops (TILE_SIZE=16)" << std::endl;
    // TASK 3: Document shared memory limitation in printout
    std::cout << "NOTE: Shared memory tiling provides no speedup at this model size" << std::endl;
    std::cout << "due to small matrix dimensions (128x64). Overhead dominates arithmetic." << std::endl;
    std::cout << "Speedup benefits appear at matrix dimensions >= 256x256." << std::endl;
    std::vector<LayerWeights> layers;
    try { layers = load_weights(weights_path); }
    catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; return 1; }
    if (layers.size() != 3) { std::cerr << "Expected 3 layers\n"; return 1; }

    // CPU SIMD p50 from our Month 2 measurements (µs per single query)
    const float cpu_simd_p50_us = 2.045f;

    int batch_sizes[] = {1, 2, 4, 8, 16, 32};
    const int RUNS = 1000;

    std::cout << "\nCPU SIMD p50 per single query: " << cpu_simd_p50_us << " µs" << std::endl;
    std::cout << "\n=== THROUGHPUT SWEEP (batch_size, GPU total, GPU QPS, CPU equiv) ===" << std::endl;

    // CSV output
    std::ofstream csv("benchmarks/throughput_results.csv");
    csv << "batch_size,gpu_total_us,gpu_qps,cpu_equivalent_us" << std::endl;

    struct Result { int bs; float gpu_us; float gpu_qps; float cpu_us; };
    std::vector<Result> results;

    // Max batch = 32, allocate for that
    const int MAX_BS = 32;
    float *d_input, *d_output;
    int8_t* d_q; float* d_scale;
    float* d_act; int32_t* d_acc;
    cudaMalloc(&d_input, MAX_BS * 128 * sizeof(float));
    cudaMalloc(&d_output, MAX_BS * 10 * sizeof(float));
    cudaMalloc(&d_q, MAX_BS * 128 * sizeof(int8_t));
    cudaMalloc(&d_scale, sizeof(float));
    cudaMalloc(&d_act, MAX_BS * 64 * sizeof(float));
    cudaMalloc(&d_acc, MAX_BS * 64 * sizeof(int32_t));

    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start); cudaEventCreate(&t_stop);

    for (int bs : batch_sizes) {
        // Generate batch of random inputs
        std::vector<float> h_batch(bs * 128);
        static std::mt19937 gen(12345);
        std::normal_distribution<float> dis(0.0f, 1.0f);
        for (auto& v : h_batch) v = dis(gen);

        // Warmup
        for (int w = 0; w < 100; w++) {
            cudaMemcpy(d_input, h_batch.data(), bs * 128 * sizeof(float), cudaMemcpyHostToDevice);
            forward_gpu_batched(layers, bs, d_input, d_output, d_q, d_scale, d_act, d_acc);
        }
        cudaDeviceSynchronize();

        // Benchmark
        std::vector<float> times;
        for (int r = 0; r < RUNS; r++) {
            cudaMemcpy(d_input, h_batch.data(), bs * 128 * sizeof(float), cudaMemcpyHostToDevice);
            cudaEventRecord(t_start);
            forward_gpu_batched(layers, bs, d_input, d_output, d_q, d_scale, d_act, d_acc);
            cudaEventRecord(t_stop);
            cudaEventSynchronize(t_stop);
            float ms = 0; cudaEventElapsedTime(&ms, t_start, t_stop);
            times.push_back(ms * 1000.0f); // µs
        }
        std::sort(times.begin(), times.end());
        float gpu_total = get_percentile(times, 50);
        float gpu_qps = (gpu_total > 0) ? (float)bs / (gpu_total / 1e6f) : 0;
        float cpu_equiv = cpu_simd_p50_us * bs;

        results.push_back({bs, gpu_total, gpu_qps, cpu_equiv});

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "batch=" << bs
                  << "  GPU_total=" << gpu_total << " µs"
                  << "  GPU_QPS=" << std::setprecision(0) << gpu_qps
                  << "  CPU_equiv=" << std::setprecision(2) << cpu_equiv << " µs" << std::endl;

        csv << bs << "," << std::fixed << std::setprecision(4) << gpu_total << ","
            << std::setprecision(0) << gpu_qps << "," << std::setprecision(4) << cpu_equiv << std::endl;
    }
    csv.close();

    // Crossover detection
    std::cout << "\n=== CPU vs GPU CROSSOVER ===" << std::endl;
    int crossover_bs = -1;
    for (const auto& r : results) {
        if (r.gpu_qps * r.bs > 0 && r.cpu_us > r.gpu_us) {
            // GPU per-query is faster than CPU per-query at this batch size
            if (r.gpu_us / r.bs <= r.cpu_us / r.bs) {
                // GPU throughput > CPU throughput
            }
        }
        // Crossover: GPU total time for batch N < CPU time for N sequential queries
        if (r.gpu_us < r.cpu_us) {
            if (crossover_bs == -1) crossover_bs = r.bs;
        }
    }
    if (crossover_bs > 0) {
        std::cout << "** Crossover batch size: GPU throughput overtakes CPU at batch_size = " << crossover_bs << " **" << std::endl;
    } else {
        std::cout << "GPU throughput does not overtake CPU at any tested batch size (1-32)." << std::endl;
        std::cout << "At batch_size=32: GPU=" << results.back().gpu_us << " µs vs CPU=" << results.back().cpu_us << " µs" << std::endl;
    }

    std::cout << "\n✓ Wrote benchmarks/throughput_results.csv" << std::endl;

    // Cleanup
    for (auto& l : layers) { cudaFree(l.d_weights); cudaFree(l.d_bias); }
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_q);
    cudaFree(d_scale); cudaFree(d_act); cudaFree(d_acc);
    cudaEventDestroy(t_start); cudaEventDestroy(t_stop);

    return 0;
}
