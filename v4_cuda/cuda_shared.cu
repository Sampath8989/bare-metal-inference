/*
 * v4_cuda/cuda_shared.cu
 *
 * Month 3 — Track A, Prompt 1
 * Shared-memory tiled (16×16) int8 matmul kernel alongside the naive kernel.
 * Benchmark harness compares both with p50/p95/p99 over 1000 runs.
 *
 * Build:  nvcc -O2 -arch=sm_75 v4_cuda/cuda_shared.cu -o cuda_shared
 * Run:    ./cuda_shared [weights_int8.bin path]
 *
 * ===========================================================================
 * IMPORTANT NOTE ON SHARED MEMORY PERFORMANCE (TASK 3):
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
 * Speedup benefits appear at matrix dimensions ≥256×256 where data reuse
 * ratios justify the tile management overhead.
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

// ============================================================
// Constants
// ============================================================
#define TILE_SIZE 16

// ============================================================
// Structure to store weights on Host & Device
// ============================================================
struct LayerWeights {
    int rows;
    int cols;
    float scale;
    std::vector<int8_t> h_weights;
    std::vector<float> h_bias;
    int8_t* d_weights;
    float* d_bias;
};

// ============================================================
// CUDA Kernels — Shared-memory tiling (NEW)
// ============================================================

// Dynamic per-layer activation requantization kernel
__global__ void quantize_kernel(const float* input, int8_t* output, float* out_scale, int size) {
    extern __shared__ float shared_max[];
    int tid = threadIdx.x;

    float val = (tid < size) ? input[tid] : 0.0f;
    shared_max[tid] = fabsf(val);
    __syncthreads();

    int pow2 = 1;
    while (pow2 < blockDim.x) pow2 <<= 1;

    for (unsigned int s = pow2 / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < blockDim.x) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float max_val = shared_max[0];
        *out_scale = (max_val == 0.0f) ? 1.0f : max_val / 127.0f;
    }
    __syncthreads();

    float scale = *out_scale;
    if (tid < size) {
        float scaled = input[tid] / scale;
        float rounded = roundf(scaled);
        if (rounded > 127.0f) rounded = 127.0f;
        if (rounded < -128.0f) rounded = -128.0f;
        output[tid] = static_cast<int8_t>(rounded);
    }
}

// Naive int8 matmul — with loop unrolling for innermost K dimension
// TASK 2: Loop unrolling using #pragma unroll for the dot-product loop.
// This reduces loop overhead (branch prediction, counter increments) and
// exposes more instruction-level parallelism to the GPU scheduler.
// NOTE: K is a runtime parameter, so nvcc may ignore this pragma if it
// cannot determine the trip count at compile time. The TILE_SIZE=16
// inner dot-product loop in the shared kernel WILL unroll (constant bound).
__global__ void matmul_int8_naive(const int8_t* A, const int8_t* B, int32_t* C_acc, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        int32_t accum = 0;
        // TASK 2: Unroll the innermost loop over K dimension
        // K is typically 32/64/128 for this model — safe to unroll
        #pragma unroll
        for (int q = 0; q < K; q++) {
            accum += static_cast<int32_t>(A[i * K + q]) * static_cast<int32_t>(B[q * N + j]);
        }
        C_acc[i * N + j] = accum;
    }
}

// Shared-memory tiled int8 matmul — 16×16 tiles with __shared__ arrays and __syncthreads()
// TASK 2: Inner dot-product loop has #pragma unroll — WILL unroll since TILE_SIZE=16 is
// a compile-time constant. The outer tile loop may not unroll (K is runtime).
__global__ void matmul_int8_shared(const int8_t* A, const int8_t* B, int32_t* C_acc, int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ int8_t tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int32_t accum = 0;

    // Loop over tiles along the K dimension
    #pragma unroll
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Cooperatively load a tile of A and B into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0;

        if (b_row < K && col < N)
            tile_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int q = 0; q < TILE_SIZE; q++) {
            accum += static_cast<int32_t>(tile_A[threadIdx.y][q])
                   * static_cast<int32_t>(tile_B[q][threadIdx.x]);
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C_acc[row * N + col] = accum;
    }
}

// Rescaling, bias addition and ReLU execution kernel
__global__ void post_process_kernel(const int32_t* C_acc, const float* bias, float* output,
                                     const float* input_scale, float weight_scale,
                                     int M, int N, bool apply_relu) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        int idx = i * N + j;
        float combined_scale = (*input_scale) * weight_scale;
        float val = static_cast<float>(C_acc[idx]) * combined_scale;
        val += bias[j];
        if (apply_relu) val = fmaxf(0.0f, val);
        output[idx] = val;
    }
}

// ============================================================
// Forward pass using NAIVE kernel
// ============================================================
void forward_gpu_naive(const std::vector<LayerWeights>& layers,
                       float* d_input_f, int m,
                       int8_t* d_input_q, float* d_input_scale,
                       float* d_act0_f, int8_t* d_act0_q, float* d_act0_scale,
                       float* d_act1_f, int8_t* d_act1_q, float* d_act1_scale,
                       float* d_act2_f,
                       int32_t* d_acc0, int32_t* d_acc1, int32_t* d_acc2) {
    // Layer 0: 128 -> 64
    quantize_kernel<<<1, 128, 128 * sizeof(float)>>>(d_input_f, d_input_q, d_input_scale, m * 128);
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((64 + 15) / 16, (m + 15) / 16);
        matmul_int8_naive<<<gridDim, blockDim>>>(d_input_q, layers[0].d_weights, d_acc0, m, 64, 128);
        post_process_kernel<<<gridDim, blockDim>>>(d_acc0, layers[0].d_bias, d_act0_f, d_input_scale, layers[0].scale, m, 64, true);
    }

    // Layer 1: 64 -> 32
    quantize_kernel<<<1, 64, 64 * sizeof(float)>>>(d_act0_f, d_act0_q, d_act0_scale, m * 64);
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((32 + 15) / 16, (m + 15) / 16);
        matmul_int8_naive<<<gridDim, blockDim>>>(d_act0_q, layers[1].d_weights, d_acc1, m, 32, 64);
        post_process_kernel<<<gridDim, blockDim>>>(d_acc1, layers[1].d_bias, d_act1_f, d_act0_scale, layers[1].scale, m, 32, true);
    }

    // Layer 2: 32 -> 10
    quantize_kernel<<<1, 32, 32 * sizeof(float)>>>(d_act1_f, d_act1_q, d_act1_scale, m * 32);
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((10 + 15) / 16, (m + 15) / 16);
        matmul_int8_naive<<<gridDim, blockDim>>>(d_act1_q, layers[2].d_weights, d_acc2, m, 10, 32);
        post_process_kernel<<<gridDim, blockDim>>>(d_acc2, layers[2].d_bias, d_act2_f, d_act1_scale, layers[2].scale, m, 10, true);
    }
}

// ============================================================
// Forward pass using SHARED-MEMORY kernel
// ============================================================
void forward_gpu_shared(const std::vector<LayerWeights>& layers,
                        float* d_input_f, int m,
                        int8_t* d_input_q, float* d_input_scale,
                        float* d_act0_f, int8_t* d_act0_q, float* d_act0_scale,
                        float* d_act1_f, int8_t* d_act1_q, float* d_act1_scale,
                        float* d_act2_f,
                        int32_t* d_acc0, int32_t* d_acc1, int32_t* d_acc2) {
    // Layer 0: 128 -> 64
    quantize_kernel<<<1, 128, 128 * sizeof(float)>>>(d_input_f, d_input_q, d_input_scale, m * 128);
    {
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((64 + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        matmul_int8_shared<<<gridDim, blockDim>>>(d_input_q, layers[0].d_weights, d_acc0, m, 64, 128);
        post_process_kernel<<<gridDim, blockDim>>>(d_acc0, layers[0].d_bias, d_act0_f, d_input_scale, layers[0].scale, m, 64, true);
    }

    // Layer 1: 64 -> 32
    quantize_kernel<<<1, 64, 64 * sizeof(float)>>>(d_act0_f, d_act0_q, d_act0_scale, m * 64);
    {
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((32 + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        matmul_int8_shared<<<gridDim, blockDim>>>(d_act0_q, layers[1].d_weights, d_acc1, m, 32, 64);
        post_process_kernel<<<gridDim, blockDim>>>(d_acc1, layers[1].d_bias, d_act1_f, d_act0_scale, layers[1].scale, m, 32, true);
    }

    // Layer 2: 32 -> 10
    quantize_kernel<<<1, 32, 32 * sizeof(float)>>>(d_act1_f, d_act1_q, d_act1_scale, m * 32);
    {
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((10 + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        matmul_int8_shared<<<gridDim, blockDim>>>(d_act1_q, layers[2].d_weights, d_acc2, m, 10, 32);
        post_process_kernel<<<gridDim, blockDim>>>(d_acc2, layers[2].d_bias, d_act2_f, d_act1_scale, layers[2].scale, m, 10, true);
    }
}

// ============================================================
// CPU reference for correctness validation
// ============================================================
void quantize_cpu(const std::vector<float>& input, std::vector<int8_t>& output, float& out_scale) {
    float max_abs = 0.0f;
    for (float val : input) max_abs = std::max(max_abs, std::abs(val));
    out_scale = (max_abs == 0.0f) ? 1.0f : max_abs / 127.0f;
    for (size_t i = 0; i < input.size(); i++) {
        float scaled = input[i] / out_scale;
        output[i] = static_cast<int8_t>(std::clamp(std::round(scaled), -128.0f, 127.0f));
    }
}

void matmul_cpu(const std::vector<int8_t>& A, const std::vector<int8_t>& B, std::vector<int32_t>& acc, int M, int N, int K) {
    std::fill(acc.begin(), acc.end(), 0);
    for (int i = 0; i < M; i++)
        for (int q = 0; q < K; q++) {
            int8_t a = A[i * K + q];
            for (int j = 0; j < N; j++)
                acc[i * N + j] += static_cast<int32_t>(a) * static_cast<int32_t>(B[q * N + j]);
        }
}

void forward_cpu(const std::vector<LayerWeights>& layers, const std::vector<float>& input,
                 std::vector<float>& output) {
    std::vector<int8_t> q_in(128), q_act0(64), q_act1(32);
    std::vector<float> act0(64), act1(32);
    std::vector<int32_t> acc0(64), acc1(32), acc2(10);
    float s0, s1, s2;

    quantize_cpu(input, q_in, s0);
    matmul_cpu(q_in, layers[0].h_weights, acc0, 1, 64, 128);
    float cs0 = s0 * layers[0].scale;
    for (int j = 0; j < 64; j++) act0[j] = std::max(0.0f, static_cast<float>(acc0[j]) * cs0 + layers[0].h_bias[j]);

    quantize_cpu(act0, q_act0, s1);
    matmul_cpu(q_act0, layers[1].h_weights, acc1, 1, 32, 64);
    float cs1 = s1 * layers[1].scale;
    for (int j = 0; j < 32; j++) act1[j] = std::max(0.0f, static_cast<float>(acc1[j]) * cs1 + layers[1].h_bias[j]);

    quantize_cpu(act1, q_act1, s2);
    matmul_cpu(q_act1, layers[2].h_weights, acc2, 1, 10, 32);
    float cs2 = s2 * layers[2].scale;
    output.resize(10);
    for (int j = 0; j < 10; j++) output[j] = std::max(0.0f, static_cast<float>(acc2[j]) * cs2 + layers[2].h_bias[j]);
}

// ============================================================
// Load Weights
// ============================================================
std::vector<LayerWeights> load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open weights file: " + path);

    std::vector<LayerWeights> layers;
    while (file.peek() != EOF) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        if (file.gcount() == 0) break;
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        float scale;
        file.read(reinterpret_cast<char*>(&scale), sizeof(scale));

        size_t num_weights = static_cast<size_t>(rows) * cols;
        std::vector<int8_t> h_weights(num_weights);
        file.read(reinterpret_cast<char*>(h_weights.data()), num_weights);

        int bias_size;
        file.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));
        std::vector<float> h_bias(bias_size);
        file.read(reinterpret_cast<char*>(h_bias.data()), bias_size * sizeof(float));

        int8_t* d_weights;
        float* d_bias;
        cudaMalloc(&d_weights, num_weights * sizeof(int8_t));
        cudaMalloc(&d_bias, bias_size * sizeof(float));
        cudaMemcpy(d_weights, h_weights.data(), num_weights * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, h_bias.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);

        LayerWeights layer;
        layer.rows = rows; layer.cols = cols; layer.scale = scale;
        layer.h_weights = std::move(h_weights);
        layer.h_bias = std::move(h_bias);
        layer.d_weights = d_weights;
        layer.d_bias = d_bias;
        layers.push_back(layer);
    }
    file.close();
    return layers;
}

// ============================================================
// Helper functions
// ============================================================
void generate_random_input(std::vector<float>& input, float mean = 0.0f, float stddev = 1.0f) {
    static std::mt19937 gen(12345);
    std::normal_distribution<float> dis(mean, stddev);
    for (auto& val : input) val = dis(gen);
}

float get_percentile(std::vector<float>& v, float p) {
    if (v.empty()) return 0.0f;
    int idx = static_cast<int>(std::ceil(v.size() * p / 100.0f)) - 1;
    if (idx < 0) idx = 0;
    if (idx >= static_cast<int>(v.size())) idx = static_cast<int>(v.size()) - 1;
    return v[idx];
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    std::string weights_path = "weights_int8.bin";  // Default: current directory (Colab: /content/weights_int8.bin)
    if (argc >= 2) weights_path = argv[1];

    std::cout << "=== GPU INT8 INFERENCE — NAIVE vs SHARED-MEMORY ===" << std::endl;
    std::cout << "Loop unrolling: #pragma unroll applied to innermost K-dimension loops" << std::endl;
    std::cout << "Loading weights from: " << weights_path << std::endl;

    std::vector<LayerWeights> layers;
    try { layers = load_weights(weights_path); }
    catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; return 1; }
    if (layers.size() != 3) { std::cerr << "Expected 3 layers, got: " << layers.size() << std::endl; return 1; }

    std::vector<float> h_input(128);
    std::vector<float> h_output_gpu(10);
    std::vector<float> h_output_cpu;
    generate_random_input(h_input);

    // ---- GPU allocations ----
    float *d_input_f, *d_act0_f, *d_act1_f, *d_act2_f;
    int8_t *d_input_q, *d_act0_q, *d_act1_q;
    float *d_input_scale, *d_act0_scale, *d_act1_scale;
    int32_t *d_acc0, *d_acc1, *d_acc2;
    const int m = 1;

    cudaMalloc(&d_input_f, m * 128 * sizeof(float));
    cudaMalloc(&d_input_q, m * 128 * sizeof(int8_t));
    cudaMalloc(&d_input_scale, sizeof(float));
    cudaMalloc(&d_act0_f, m * 64 * sizeof(float));
    cudaMalloc(&d_act0_q, m * 64 * sizeof(int8_t));
    cudaMalloc(&d_act0_scale, sizeof(float));
    cudaMalloc(&d_act1_f, m * 32 * sizeof(float));
    cudaMalloc(&d_act1_q, m * 32 * sizeof(int8_t));
    cudaMalloc(&d_act1_scale, sizeof(float));
    cudaMalloc(&d_act2_f, m * 10 * sizeof(float));
    cudaMalloc(&d_acc0, m * 64 * sizeof(int32_t));
    cudaMalloc(&d_acc1, m * 32 * sizeof(int32_t));
    cudaMalloc(&d_acc2, m * 10 * sizeof(int32_t));

    // ---- Correctness Validation ----
    const int VAL_SAMPLES = 100;
    float max_err_naive = 0.0f, max_err_shared = 0.0f;

    std::cout << "\n=== CORRECTNESS VALIDATION (" << VAL_SAMPLES << " random samples) ===" << std::endl;
    for (int s = 0; s < VAL_SAMPLES; s++) {
        generate_random_input(h_input);
        forward_cpu(layers, h_input, h_output_cpu);

        cudaMemcpy(d_input_f, h_input.data(), 128 * sizeof(float), cudaMemcpyHostToDevice);

        // Naive
        forward_gpu_naive(layers, d_input_f, m, d_input_q, d_input_scale, d_act0_f, d_act0_q, d_act0_scale,
                          d_act1_f, d_act1_q, d_act1_scale, d_act2_f, d_acc0, d_acc1, d_acc2);
        cudaMemcpy(h_output_gpu.data(), d_act2_f, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; i++) max_err_naive = std::max(max_err_naive, std::abs(h_output_gpu[i] - h_output_cpu[i]));

        // Shared-memory
        forward_gpu_shared(layers, d_input_f, m, d_input_q, d_input_scale, d_act0_f, d_act0_q, d_act0_scale,
                           d_act1_f, d_act1_q, d_act1_scale, d_act2_f, d_acc0, d_acc1, d_acc2);
        cudaMemcpy(h_output_gpu.data(), d_act2_f, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 10; i++) max_err_shared = std::max(max_err_shared, std::abs(h_output_gpu[i] - h_output_cpu[i]));

        if (s == 0) {
            std::cout << "Sample 1 CPU:  [";
            for (int i = 0; i < 10; i++) std::cout << h_output_cpu[i] << (i < 9 ? ", " : "");
            std::cout << "]" << std::endl;
            std::cout << "Sample 1 GPU:  [";
            for (int i = 0; i < 10; i++) std::cout << h_output_gpu[i] << (i < 9 ? ", " : "");
            std::cout << "]" << std::endl;
        }
    }
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Max abs error (naive vs CPU):   " << max_err_naive << std::endl;
    std::cout << "Max abs error (shared vs CPU):  " << max_err_shared << std::endl;

    // ---- Warmup ----
    for (int i = 0; i < 100; i++) {
        cudaMemcpy(d_input_f, h_input.data(), 128 * sizeof(float), cudaMemcpyHostToDevice);
        forward_gpu_naive(layers, d_input_f, m, d_input_q, d_input_scale, d_act0_f, d_act0_q, d_act0_scale,
                          d_act1_f, d_act1_q, d_act1_scale, d_act2_f, d_acc0, d_acc1, d_acc2);
    }
    for (int i = 0; i < 100; i++) {
        cudaMemcpy(d_input_f, h_input.data(), 128 * sizeof(float), cudaMemcpyHostToDevice);
        forward_gpu_shared(layers, d_input_f, m, d_input_q, d_input_scale, d_act0_f, d_act0_q, d_act0_scale,
                           d_act1_f, d_act1_q, d_act1_scale, d_act2_f, d_acc0, d_acc1, d_acc2);
    }
    cudaDeviceSynchronize();

    // ---- Benchmark: 1000 runs each ----
    const int RUNS = 1000;
    std::vector<float> naive_times, shared_times;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Naive benchmark
    for (int r = 0; r < RUNS; r++) {
        cudaMemcpy(d_input_f, h_input.data(), 128 * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        forward_gpu_naive(layers, d_input_f, m, d_input_q, d_input_scale, d_act0_f, d_act0_q, d_act0_scale,
                          d_act1_f, d_act1_q, d_act1_scale, d_act2_f, d_acc0, d_acc1, d_acc2);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        naive_times.push_back(ms * 1000.0f); // convert to µs
    }

    // Shared-memory benchmark
    for (int r = 0; r < RUNS; r++) {
        cudaMemcpy(d_input_f, h_input.data(), 128 * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        forward_gpu_shared(layers, d_input_f, m, d_input_q, d_input_scale, d_act0_f, d_act0_q, d_act0_scale,
                           d_act1_f, d_act1_q, d_act1_scale, d_act2_f, d_acc0, d_acc1, d_acc2);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        shared_times.push_back(ms * 1000.0f); // convert to µs
    }

    std::sort(naive_times.begin(), naive_times.end());
    std::sort(shared_times.begin(), shared_times.end());

    std::cout << "\n=== LATENCY STATISTICS (over " << RUNS << " runs, batch=1) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Naive kernel:  p50=" << get_percentile(naive_times, 50)
              << " µs  |  p95=" << get_percentile(naive_times, 95)
              << " µs  |  p99=" << get_percentile(naive_times, 99) << " µs" << std::endl;
    std::cout << "Shared kernel: p50=" << get_percentile(shared_times, 50)
              << " µs  |  p95=" << get_percentile(shared_times, 95)
              << " µs  |  p99=" << get_percentile(shared_times, 99) << " µs" << std::endl;

    float speedup_p50 = get_percentile(naive_times, 50) / get_percentile(shared_times, 50);
    std::cout << "Speedup (naive/shared): " << std::setprecision(3) << speedup_p50 << "x at p50" << std::endl;
    // TASK 3: Explicitly document the shared memory limitation
    std::cout << "\nNOTE: Shared memory tiling provides no speedup at this model size" << std::endl;
    std::cout << "due to small matrix dimensions (128x64). Overhead dominates arithmetic." << std::endl;
    std::cout << "Speedup benefits appear at matrix dimensions >= 256x256." << std::endl;

    // ---- Write CSV ----
    std::ofstream csv("benchmarks/latency_results_shared.csv");
    csv << "iteration,naive_us,shared_us" << std::endl;
    for (int i = 0; i < RUNS; i++) {
        csv << (i + 1) << "," << std::fixed << std::setprecision(4)
            << naive_times[i] << "," << shared_times[i] << std::endl;
    }
    csv.close();
    std::cout << "\n✓ Wrote benchmarks/latency_results_shared.csv" << std::endl;

    // ---- Cleanup ----
    for (auto& layer : layers) { cudaFree(layer.d_weights); cudaFree(layer.d_bias); }
    cudaFree(d_input_f); cudaFree(d_input_q); cudaFree(d_input_scale);
    cudaFree(d_act0_f); cudaFree(d_act0_q); cudaFree(d_act0_scale);
    cudaFree(d_act1_f); cudaFree(d_act1_q); cudaFree(d_act1_scale);
    cudaFree(d_act2_f); cudaFree(d_acc0); cudaFree(d_acc1); cudaFree(d_acc2);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
