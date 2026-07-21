/*
 * v4_cuda/cuda_pipeline.cu
 *
 * Month 3 — Track A, Prompt 2
 * End-to-end timed inference: load weights_int8.bin from disk → H2D copy →
 * run ALL layers on GPU with shared-memory kernel → D2H copy → print prediction.
 * Reports H2D+D2H transfer time, kernel compute time, and combined total.
 *
 * Build:  nvcc -O2 -arch=sm_75 v4_cuda/cuda_pipeline.cu -o cuda_pipeline
 * Run:    ./cuda_pipeline [weights_int8.bin path]
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
// CUDA Kernels (shared-memory tiled matmul)
// ============================================================
__global__ void quantize_kernel(const float* input, int8_t* output, float* out_scale, int size) {
    extern __shared__ float shared_max[];
    int tid = threadIdx.x;
    // Each thread may process multiple elements if size > blockDim.x
    float local_max = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
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
    for (int i = tid; i < size; i += blockDim.x) {
        float scaled = input[i] / scale;
        float rounded = roundf(scaled);
        if (rounded > 127.0f) rounded = 127.0f;
        if (rounded < -128.0f) rounded = -128.0f;
        output[i] = static_cast<int8_t>(rounded);
    }
}

__global__ void matmul_int8_shared(const int8_t* A, const int8_t* B, int32_t* C_acc, int M, int N, int K) {
    __shared__ int8_t tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t tile_B[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int32_t accum = 0;
    // TASK 2: Loop unrolling — both the tile loop and inner dot-product loop
    // The inner TILE_SIZE loop WILL unroll (compile-time constant).
    // The outer tile loop has #pragma unroll; nvcc may or may not unroll
    // it since K is runtime, but the hint is still useful for small K.
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
    if (row < M && col < N) C_acc[row * N + col] = accum;
}

__global__ void post_process_kernel(const int32_t* C_acc, const float* bias, float* output,
                                     const float* input_scale, float weight_scale, int M, int N, bool apply_relu) {
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
// GPU forward pass (shared-memory kernel, returns device pointer to output)
// ============================================================
void forward_gpu_pipeline(const std::vector<LayerWeights>& layers,
                           float* d_input_f, float* d_output_f,
                           int8_t* d_buf_q, float* d_buf_scale,
                           float* d_act_f, int32_t* d_acc) {
    const int m = 1;
    // Layer 0: 128 -> 64
    quantize_kernel<<<1, 128, 128 * sizeof(float)>>>(d_input_f, d_buf_q, d_buf_scale, m * 128);
    { dim3 b(TILE_SIZE, TILE_SIZE), g((64+TILE_SIZE-1)/TILE_SIZE, (m+TILE_SIZE-1)/TILE_SIZE);
      matmul_int8_shared<<<g, b>>>(d_buf_q, layers[0].d_weights, d_acc, m, 64, 128);
      post_process_kernel<<<g, b>>>(d_acc, layers[0].d_bias, d_act_f, d_buf_scale, layers[0].scale, m, 64, true); }
    // Layer 1: 64 -> 32
    quantize_kernel<<<1, 64, 64 * sizeof(float)>>>(d_act_f, d_buf_q, d_buf_scale, m * 64);
    { dim3 b(TILE_SIZE, TILE_SIZE), g((32+TILE_SIZE-1)/TILE_SIZE, (m+TILE_SIZE-1)/TILE_SIZE);
      matmul_int8_shared<<<g, b>>>(d_buf_q, layers[1].d_weights, d_acc, m, 32, 64);
      post_process_kernel<<<g, b>>>(d_acc, layers[1].d_bias, d_act_f, d_buf_scale, layers[1].scale, m, 32, true); }
    // Layer 2: 32 -> 10
    quantize_kernel<<<1, 32, 32 * sizeof(float)>>>(d_act_f, d_buf_q, d_buf_scale, m * 32);
    { dim3 b(TILE_SIZE, TILE_SIZE), g((10+TILE_SIZE-1)/TILE_SIZE, (m+TILE_SIZE-1)/TILE_SIZE);
      matmul_int8_shared<<<g, b>>>(d_buf_q, layers[2].d_weights, d_acc, m, 10, 32);
      post_process_kernel<<<g, b>>>(d_acc, layers[2].d_bias, d_output_f, d_buf_scale, layers[2].scale, m, 10, true); }
}

// ============================================================
// CPU reference
// ============================================================
void quantize_cpu(const std::vector<float>& input, std::vector<int8_t>& output, float& out_scale) {
    float max_abs = 0.0f;
    for (float val : input) max_abs = std::max(max_abs, std::abs(val));
    out_scale = (max_abs == 0.0f) ? 1.0f : max_abs / 127.0f;
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = static_cast<int8_t>(std::clamp(std::round(input[i] / out_scale), -128.0f, 127.0f));
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

void forward_cpu(const std::vector<LayerWeights>& layers, const std::vector<float>& input, std::vector<float>& output) {
    std::vector<int8_t> q(128), q64(64), q32(32);
    std::vector<float> a64(64), a32(32);
    std::vector<int32_t> acc64(64), acc32(32), acc10(10);
    float s0, s1, s2;
    quantize_cpu(input, q, s0);
    matmul_cpu(q, layers[0].h_weights, acc64, 1, 64, 128);
    for (int j = 0; j < 64; j++) a64[j] = std::max(0.0f, static_cast<float>(acc64[j]) * s0 * layers[0].scale + layers[0].h_bias[j]);
    quantize_cpu(a64, q64, s1);
    matmul_cpu(q64, layers[1].h_weights, acc32, 1, 32, 64);
    for (int j = 0; j < 32; j++) a32[j] = std::max(0.0f, static_cast<float>(acc32[j]) * s1 * layers[1].scale + layers[1].h_bias[j]);
    quantize_cpu(a32, q32, s2);
    matmul_cpu(q32, layers[2].h_weights, acc10, 1, 10, 32);
    output.resize(10);
    for (int j = 0; j < 10; j++) output[j] = std::max(0.0f, static_cast<float>(acc10[j]) * s2 * layers[2].scale + layers[2].h_bias[j]);
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
        size_t nw = static_cast<size_t>(rows) * cols;
        std::vector<int8_t> hw(nw);
        file.read(reinterpret_cast<char*>(hw.data()), nw);
        int bs; file.read(reinterpret_cast<char*>(&bs), sizeof(bs));
        std::vector<float> hb(bs);
        file.read(reinterpret_cast<char*>(hb.data()), bs * sizeof(float));
        int8_t* dw; float* db;
        cudaMalloc(&dw, nw * sizeof(int8_t));
        cudaMalloc(&db, bs * sizeof(float));
        cudaMemcpy(dw, hw.data(), nw * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb.data(), bs * sizeof(float), cudaMemcpyHostToDevice);
        layers.push_back({rows, cols, scale, std::move(hw), std::move(hb), dw, db});
    }
    return layers;
}

void generate_random_input(std::vector<float>& input) {
    static std::mt19937 gen(12345);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (auto& v : input) v = dis(gen);
}

float get_percentile(std::vector<float>& v, float p) {
    int idx = static_cast<int>(std::ceil(v.size() * p / 100.0f)) - 1;
    return v[std::max(0, std::min(idx, static_cast<int>(v.size()) - 1))];
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    std::string weights_path = "weights_int8.bin";  // Default: current directory (Colab: /content/weights_int8.bin)
    if (argc >= 2) weights_path = argv[1];

    std::cout << "=== GPU INT8 PIPELINE — CPU→GPU→CPU ===" << std::endl;
    std::cout << "Loop unrolling: #pragma unroll applied to inner tile loops (TILE_SIZE=16)" << std::endl;
    // TASK 3: Document shared memory limitation in printout
    std::cout << "NOTE: Shared memory tiling provides no speedup at this model size" << std::endl;
    std::cout << "due to small matrix dimensions (128x64). Overhead dominates arithmetic." << std::endl;
    std::cout << "Speedup benefits appear at matrix dimensions >= 256x256." << std::endl;
    std::cout << "Loading weights from: " << weights_path << std::endl;

    std::vector<LayerWeights> layers;
    try { layers = load_weights(weights_path); }
    catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; return 1; }
    if (layers.size() != 3) { std::cerr << "Expected 3 layers\n"; return 1; }

    // GPU allocations
    float *d_input_f, *d_output_f;
    int8_t* d_buf_q; float* d_buf_scale;
    float* d_act_f; int32_t* d_acc;
    cudaMalloc(&d_input_f, 128 * sizeof(float));
    cudaMalloc(&d_output_f, 10 * sizeof(float));
    cudaMalloc(&d_buf_q, 128 * sizeof(int8_t)); // large enough for any layer
    cudaMalloc(&d_buf_scale, sizeof(float));
    cudaMalloc(&d_act_f, 64 * sizeof(float));   // large enough for any layer
    cudaMalloc(&d_acc, 64 * sizeof(int32_t));

    std::vector<float> h_input(128), h_output_gpu(10), h_output_cpu;
    std::vector<float> h_output_golden;
    generate_random_input(h_input);

    // CPU reference
    forward_cpu(layers, h_input, h_output_cpu);

    // Correctness
    cudaMemcpy(d_input_f, h_input.data(), 128 * sizeof(float), cudaMemcpyHostToDevice);
    forward_gpu_pipeline(layers, d_input_f, d_output_f, d_buf_q, d_buf_scale, d_act_f, d_acc);
    cudaMemcpy(h_output_gpu.data(), d_output_f, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f, mean_err = 0.0f;
    for (int i = 0; i < 10; i++) {
        float e = std::abs(h_output_gpu[i] - h_output_cpu[i]);
        max_err = std::max(max_err, e);
        mean_err += e;
    }
    mean_err /= 10.0f;

    std::cout << "\n=== CORRECTNESS ===" << std::endl;
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Max absolute error (GPU vs CPU):  " << max_err << std::endl;
    std::cout << "Mean absolute error (GPU vs CPU): " << mean_err << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "GPU output: [";
    for (int i = 0; i < 10; i++) std::cout << h_output_gpu[i] << (i < 9 ? ", " : "");
    std::cout << "]" << std::endl;
    std::cout << "CPU output: [";
    for (int i = 0; i < 10; i++) std::cout << h_output_cpu[i] << (i < 9 ? ", " : "");
    std::cout << "]" << std::endl;

    // Warmup
    for (int i = 0; i < 100; i++) {
        cudaMemcpy(d_input_f, h_input.data(), 128 * sizeof(float), cudaMemcpyHostToDevice);
        forward_gpu_pipeline(layers, d_input_f, d_output_f, d_buf_q, d_buf_scale, d_act_f, d_acc);
    }
    cudaDeviceSynchronize();

    // Benchmark: 1000 runs — measure transfer time, kernel time, total time
    const int RUNS = 1000;
    std::vector<float> transfer_times, kernel_times, total_times;

    cudaEvent_t t_start, t_h2d_done, t_kernel_done, t_d2h_done;
    cudaEventCreate(&t_start); cudaEventCreate(&t_h2d_done);
    cudaEventCreate(&t_kernel_done); cudaEventCreate(&t_d2h_done);

    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(t_start);
        cudaMemcpy(d_input_f, h_input.data(), 128 * sizeof(float), cudaMemcpyHostToDevice);
        cudaEventRecord(t_h2d_done);

        forward_gpu_pipeline(layers, d_input_f, d_output_f, d_buf_q, d_buf_scale, d_act_f, d_acc);
        cudaEventRecord(t_kernel_done);

        cudaMemcpy(h_output_gpu.data(), d_output_f, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(t_d2h_done);
        cudaEventSynchronize(t_d2h_done);

        float t_h2d = 0, t_k = 0, t_d2h = 0, t_total = 0;
        cudaEventElapsedTime(&t_h2d, t_start, t_h2d_done);
        cudaEventElapsedTime(&t_k, t_h2d_done, t_kernel_done);
        cudaEventElapsedTime(&t_d2h, t_kernel_done, t_d2h_done);
        cudaEventElapsedTime(&t_total, t_start, t_d2h_done);

        transfer_times.push_back((t_h2d + t_d2h) * 1000.0f);
        kernel_times.push_back(t_k * 1000.0f);
        total_times.push_back(t_total * 1000.0f);
    }

    std::sort(transfer_times.begin(), transfer_times.end());
    std::sort(kernel_times.begin(), kernel_times.end());
    std::sort(total_times.begin(), total_times.end());

    std::cout << "\n=== LATENCY STATISTICS (over " << RUNS << " runs, batch=1) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "H2D+D2H Transfer: p50=" << get_percentile(transfer_times, 50)
              << " µs  |  p95=" << get_percentile(transfer_times, 95)
              << " µs  |  p99=" << get_percentile(transfer_times, 99) << " µs" << std::endl;
    std::cout << "Kernel Compute:   p50=" << get_percentile(kernel_times, 50)
              << " µs  |  p95=" << get_percentile(kernel_times, 95)
              << " µs  |  p99=" << get_percentile(kernel_times, 99) << " µs" << std::endl;
    std::cout << "Combined Total:   p50=" << get_percentile(total_times, 50)
              << " µs  |  p95=" << get_percentile(total_times, 95)
              << " µs  |  p99=" << get_percentile(total_times, 99) << " µs" << std::endl;

    // Write CSV
    std::ofstream csv("benchmarks/latency_results_pipeline.csv");
    csv << "iteration,component,us" << std::endl;
    for (int i = 0; i < RUNS; i++) {
        csv << (i+1) << ",transfer," << std::fixed << std::setprecision(4) << transfer_times[i] << std::endl;
        csv << (i+1) << ",kernel," << std::fixed << std::setprecision(4) << kernel_times[i] << std::endl;
        csv << (i+1) << ",total," << std::fixed << std::setprecision(4) << total_times[i] << std::endl;
    }
    csv.close();
    std::cout << "\n✓ Wrote benchmarks/latency_results_pipeline.csv" << std::endl;

    // Cleanup
    for (auto& l : layers) { cudaFree(l.d_weights); cudaFree(l.d_bias); }
    cudaFree(d_input_f); cudaFree(d_output_f); cudaFree(d_buf_q);
    cudaFree(d_buf_scale); cudaFree(d_act_f); cudaFree(d_acc);
    cudaEventDestroy(t_start); cudaEventDestroy(t_h2d_done);
    cudaEventDestroy(t_kernel_done); cudaEventDestroy(t_d2h_done);

    return 0;
}
