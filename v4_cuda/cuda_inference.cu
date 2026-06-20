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
// CUDA Kernels
// ============================================================

// Dynamic per-layer activation requantization kernel (block size matches layer dimensions)
// NOTE: Generalized to support non-power-of-two block dimensions via padded reduction.
__global__ void quantize_kernel(const float* input, int8_t* output, float* out_scale, int size) {
    extern __shared__ float shared_max[];
    int tid = threadIdx.x;

    float val = (tid < size) ? input[tid] : 0.0f;
    shared_max[tid] = fabsf(val);
    __syncthreads();

    // Find the next power of two relative to blockDim.x
    int pow2 = 1;
    while (pow2 < blockDim.x) {
        pow2 <<= 1;
    }

    // Generic reduction supporting arbitrary block dimensions
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

// Naive int8 x int8 -> int32 matmul (One thread per output element)
// OVERFLOW RISK COMMENT:
// The maximum product of two int8 elements is 16384. When summing products over dimension K,
// the maximum theoretical accumulation value is K * 16384. With int32_t accumulation,
// overflow occurs if the accumulator exceeds 2,147,483,647. Thus, K can go up to
// (2147483647 / 16384) = 131,072 before overflow is mathematically possible.
// For our network (max K=128), overflow is mathematically impossible.
__global__ void matmul_int8_kernel(const int8_t* A, const int8_t* B, int32_t* C_acc, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row Index (Batch dimension)
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column Index (Output feature dimension)

    if (i < M && j < N) {
        int32_t accum = 0;
        for (int q = 0; q < K; q++) {
            // INDEXING MATH EXPLANATION:
            // A has shape [M, K]. Element at row i, col q is: A[i * K + q] (i = batch dimension)
            // B is the weight matrix of shape [K, N]. Element at row q, col j is: B[q * N + j] (NO batch dimension)
            // C_acc has shape [M, N]. Element at row i, col j is: C_acc[i * N + j]
            accum += static_cast<int32_t>(A[i * K + q]) * static_cast<int32_t>(B[q * N + j]);
        }
        C_acc[i * N + j] = accum;
    }
}

// Rescaling, bias addition and ReLU execution kernel
__global__ void post_process_kernel(const int32_t* C_acc, const float* bias, float* output, const float* input_scale, float weight_scale, int M, int N, bool apply_relu) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        int idx = i * N + j;
        // combined_scale = input_scale * weight_scale
        float combined_scale = (*input_scale) * weight_scale;
        float val = static_cast<float>(C_acc[idx]) * combined_scale;
        val += bias[j];
        if (apply_relu) {
            val = fmaxf(0.0f, val);
        }
        output[idx] = val;
    }
}

// ============================================================
// Pre-allocated Contexts (to completely isolate memory overhead)
// ============================================================
struct GPUContext {
    float* d_input_f;
    int8_t* d_input_q;
    float* d_input_scale;
    float* d_act0_f;
    int8_t* d_act0_q;
    float* d_act0_scale;
    float* d_act1_f;
    int8_t* d_act1_q;
    float* d_act1_scale;
    float* d_act2_f;
    int32_t* d_acc0;
    int32_t* d_acc1;
    int32_t* d_acc2;
    int m;

    GPUContext(int batch_size) : m(batch_size) {
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
    }

    ~GPUContext() {
        cudaFree(d_input_f);
        cudaFree(d_input_q);
        cudaFree(d_input_scale);
        cudaFree(d_act0_f);
        cudaFree(d_act0_q);
        cudaFree(d_act0_scale);
        cudaFree(d_act1_f);
        cudaFree(d_act1_q);
        cudaFree(d_act1_scale);
        cudaFree(d_act2_f);
        cudaFree(d_acc0);
        cudaFree(d_acc1);
        cudaFree(d_acc2);
    }
};

struct CPUContext {
    std::vector<int8_t> q_in;
    std::vector<float> act0;
    std::vector<int8_t> q_act0;
    std::vector<float> act1;
    std::vector<int8_t> q_act1;
    std::vector<float> act2;
    std::vector<int32_t> acc0;
    std::vector<int32_t> acc1;
    std::vector<int32_t> acc2;

    CPUContext() :
        q_in(128, 0),
        act0(64, 0.0f), q_act0(64, 0),
        act1(32, 0.0f), q_act1(32, 0),
        act2(10, 0.0f),
        acc0(64, 0), acc1(32, 0), acc2(10, 0) {}
};

// ============================================================
// Load Weights Function
// ============================================================
std::vector<LayerWeights> load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open weights file: " + path);
    }

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
        layer.rows = rows;
        layer.cols = cols;
        layer.scale = scale;
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
// Forward Pass Routines
// ============================================================
void forward_gpu(const std::vector<LayerWeights>& layers, GPUContext& ctx) {
    // Layer 0: Input -> Act0 (128 -> 64)
    quantize_kernel<<<1, 128, 128 * sizeof(float)>>>(ctx.d_input_f, ctx.d_input_q, ctx.d_input_scale, ctx.m * 128);
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((64 + 15) / 16, (ctx.m + 15) / 16);
        matmul_int8_kernel<<<gridDim, blockDim>>>(ctx.d_input_q, layers[0].d_weights, ctx.d_acc0, ctx.m, 64, 128);
        post_process_kernel<<<gridDim, blockDim>>>(ctx.d_acc0, layers[0].d_bias, ctx.d_act0_f, ctx.d_input_scale, layers[0].scale, ctx.m, 64, true);
    }

    // Layer 1: Act0 -> Act1 (64 -> 32)
    quantize_kernel<<<1, 64, 64 * sizeof(float)>>>(ctx.d_act0_f, ctx.d_act0_q, ctx.d_act0_scale, ctx.m * 64);
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((32 + 15) / 16, (ctx.m + 15) / 16);
        matmul_int8_kernel<<<gridDim, blockDim>>>(ctx.d_act0_q, layers[1].d_weights, ctx.d_acc1, ctx.m, 32, 64);
        post_process_kernel<<<gridDim, blockDim>>>(ctx.d_acc1, layers[1].d_bias, ctx.d_act1_f, ctx.d_act0_scale, layers[1].scale, ctx.m, 32, true);
    }

    // Layer 2: Act1 -> Act2 (32 -> 10)
    quantize_kernel<<<1, 32, 32 * sizeof(float)>>>(ctx.d_act1_f, ctx.d_act1_q, ctx.d_act1_scale, ctx.m * 32);
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((10 + 15) / 16, (ctx.m + 15) / 16);
        matmul_int8_kernel<<<gridDim, blockDim>>>(ctx.d_act1_q, layers[2].d_weights, ctx.d_acc2, ctx.m, 10, 32);
        post_process_kernel<<<gridDim, blockDim>>>(ctx.d_acc2, layers[2].d_bias, ctx.d_act2_f, ctx.d_act1_scale, layers[2].scale, ctx.m, 10, true);
    }
}

void quantize_cpu(const std::vector<float>& input, std::vector<int8_t>& output, float& out_scale) {
    float max_abs = 0.0f;
    for (float val : input) {
        max_abs = std::max(max_abs, std::abs(val));
    }
    out_scale = (max_abs == 0.0f) ? 1.0f : max_abs / 127.0f;
    for (size_t i = 0; i < input.size(); i++) {
        float scaled = input[i] / out_scale;
        output[i] = static_cast<int8_t>(std::clamp(std::round(scaled), -128.0f, 127.0f));    }
}

void matmul_cpu(const std::vector<int8_t>& A, const std::vector<int8_t>& B, std::vector<int32_t>& acc, int M, int N, int K) {
    std::fill(acc.begin(), acc.end(), 0);
    for (int i = 0; i < M; i++) {
        for (int q = 0; q < K; q++) {
            int8_t a = A[i * K + q];
            for (int j = 0; j < N; j++) {
                acc[i * N + j] += static_cast<int32_t>(a) * static_cast<int32_t>(B[q * N + j]);
            }
        }
    }
}

void forward_cpu(const std::vector<LayerWeights>& layers, const std::vector<float>& input, CPUContext& ctx) {
    float scale0;
    quantize_cpu(input, ctx.q_in, scale0);
    matmul_cpu(ctx.q_in, layers[0].h_weights, ctx.acc0, 1, 64, 128);
    float cs0 = scale0 * layers[0].scale;
    for (int j = 0; j < 64; j++) {
        ctx.act0[j] = std::max(0.0f, static_cast<float>(ctx.acc0[j]) * cs0 + layers[0].h_bias[j]);
    }

    float scale1;
    quantize_cpu(ctx.act0, ctx.q_act0, scale1);
    matmul_cpu(ctx.q_act0, layers[1].h_weights, ctx.acc1, 1, 32, 64);
    float cs1 = scale1 * layers[1].scale;
    for (int j = 0; j < 32; j++) {
        ctx.act1[j] = std::max(0.0f, static_cast<float>(ctx.acc1[j]) * cs1 + layers[1].h_bias[j]);
    }

    float scale2;
    quantize_cpu(ctx.act1, ctx.q_act1, scale2);
    matmul_cpu(ctx.q_act1, layers[2].h_weights, ctx.acc2, 1, 10, 32);
    float cs2 = scale2 * layers[2].scale;
    for (int j = 0; j < 10; j++) {
        ctx.act2[j] = std::max(0.0f, static_cast<float>(ctx.acc2[j]) * cs2 + layers[2].h_bias[j]);
    }
}

// ============================================================
// Helper: Random normal distribution input generator
// ============================================================
void generate_random_input(std::vector<float>& input, float mean = 0.0f, float stddev = 1.0f) {
    static std::mt19937 gen(12345); // Fixed seed for reproducible validation
    std::normal_distribution<float> dis(mean, stddev);
    for (auto& val : input) {
        val = dis(gen);
    }
}

// ============================================================
// Exact Percentile Indexing
// ============================================================
float get_percentile(std::vector<float>& v, float p) {
    if (v.empty()) return 0.0f;
    int idx = static_cast<int>(std::ceil(v.size() * p / 100.0f)) - 1;
    if (idx < 0) idx = 0;
    if (idx >= static_cast<int>(v.size())) idx = static_cast<int>(v.size()) - 1;
    return v[idx];
}

int main(int argc, char* argv[]) {
    // UPDATED PATH HERE
    std::string weights_path = "/kaggle/input/datasets/kingshiva8989/bare-metal-weights/weights_int8.bin";
    
    if (argc >= 2) {
        weights_path = argv[1];
    }

    std::cout << "=== GPU INT8 INFERENCE ENGINE & BENCHMARK ===" << std::endl;
    std::cout << "Loading weights from: " << weights_path << std::endl;
    std::vector<LayerWeights> layers;
    try {
        layers = load_weights(weights_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        std::cerr << "Verify path and ensure you uploaded the weights as a Kaggle dataset named 'bare-metal-weights'." << std::endl;
        return 1;
    }

    if (layers.size() != 3) {
        std::cerr << "Expected 3 layers, got: " << layers.size() << std::endl;
        return 1;
    }

    std::vector<float> h_input(128, 0.0f);
    std::vector<float> h_output_gpu(10, 0.0f);

    GPUContext gpu_ctx(1);
    CPUContext cpu_ctx;

    // Correctness Validation (100 random input samples)
    const int VAL_SAMPLES = 100;
    float max_global_err = 0.0f;
    double sum_mean_err = 0.0;

    std::cout << "\n=== CORRECTNESS VALIDATION (100 random input samples) ===" << std::endl;
    for (int s = 0; s < VAL_SAMPLES; s++) {
        generate_random_input(h_input, 0.0f, 1.0f);

        forward_cpu(layers, h_input, cpu_ctx);

        cudaMemcpy(gpu_ctx.d_input_f, h_input.data(), 128 * sizeof(float), cudaMemcpyHostToDevice);
        forward_gpu(layers, gpu_ctx);
        cudaMemcpy(h_output_gpu.data(), gpu_ctx.d_act2_f, 10 * sizeof(float), cudaMemcpyDeviceToHost);

        float max_sample_err = 0.0f;
        float mean_sample_err = 0.0f;
        for (int i = 0; i < 10; i++) {
            float err = std::abs(h_output_gpu[i] - cpu_ctx.act2[i]);
            max_sample_err = std::max(max_sample_err, err);
            mean_sample_err += err;
        }
        mean_sample_err /= 10.0f;

        max_global_err = std::max(max_global_err, max_sample_err);
        sum_mean_err += mean_sample_err;

        if (s == 0) {
            std::cout << "Sample 1 GPU Output: [";
            for (int i = 0; i < 10; i++) std::cout << h_output_gpu[i] << (i < 9 ? ", " : "");
            std::cout << "]" << std::endl;

            std::cout << "Sample 1 CPU Output: [";
            for (int i = 0; i < 10; i++) std::cout << cpu_ctx.act2[i] << (i < 9 ? ", " : "");
            std::cout << "]" << std::endl;
        }
    }

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "Max absolute error over all samples:  " << max_global_err << std::endl;
    std::cout << "Mean absolute error over all samples: " << (sum_mean_err / VAL_SAMPLES) << std::endl;

    // Warmup (100 runs)
    generate_random_input(h_input, 0.0f, 1.0f);
    for (int i = 0; i < 100; i++) {
        cudaMemcpy(gpu_ctx.d_input_f, h_input.data(), 128 * sizeof(float), cudaMemcpyHostToDevice);
        forward_gpu(layers, gpu_ctx);
        cudaMemcpy(h_output_gpu.data(), gpu_ctx.d_act2_f, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    // Timed Loop (1000 runs)
    const int RUNS = 1000;
    std::vector<float> gpu_transfer_times;
    std::vector<float> gpu_compute_times;
    std::vector<float> gpu_total_times;
    std::vector<float> cpu_times;

    cudaEvent_t start_transfer, start_compute, end_compute, end_transfer;
    cudaEventCreate(&start_transfer);
    cudaEventCreate(&start_compute);
    cudaEventCreate(&end_compute);
    cudaEventCreate(&end_transfer);

    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start_transfer);
        cudaMemcpy(gpu_ctx.d_input_f, h_input.data(), 128 * sizeof(float), cudaMemcpyHostToDevice);

        cudaEventRecord(start_compute);
        forward_gpu(layers, gpu_ctx);
        cudaEventRecord(end_compute);

        cudaMemcpy(h_output_gpu.data(), gpu_ctx.d_act2_f, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaEventRecord(end_transfer);

        cudaEventSynchronize(end_transfer);

        float t_h2d_ms = 0.0f;
        cudaEventElapsedTime(&t_h2d_ms, start_transfer, start_compute);

        float t_compute_ms = 0.0f;
        cudaEventElapsedTime(&t_compute_ms, start_compute, end_compute);

        float t_d2h_ms = 0.0f;
        cudaEventElapsedTime(&t_d2h_ms, end_compute, end_transfer);

        float t_total_ms = 0.0f;
        cudaEventElapsedTime(&t_total_ms, start_transfer, end_transfer);

        gpu_transfer_times.push_back((t_h2d_ms + t_d2h_ms) * 1000.0f);
        gpu_compute_times.push_back(t_compute_ms * 1000.0f);
        gpu_total_times.push_back(t_total_ms * 1000.0f);
    }

    // CPU Baseline Timing (1000 runs)
    for (int r = 0; r < RUNS; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        forward_cpu(layers, h_input, cpu_ctx);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
        cpu_times.push_back(static_cast<float>(elapsed_us));
    }

    std::sort(gpu_transfer_times.begin(), gpu_transfer_times.end());
    std::sort(gpu_compute_times.begin(), gpu_compute_times.end());
    std::sort(gpu_total_times.begin(), gpu_total_times.end());
    std::sort(cpu_times.begin(), cpu_times.end());

    std::cout << "\n=== LATENCY STATISTICS (over " << RUNS << " runs) ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "GPU Memcpy (H2D + D2H) latency:" << std::endl;
    std::cout << "  p50: " << get_percentile(gpu_transfer_times, 50) << " us  |  p95: " << get_percentile(gpu_transfer_times, 95) << " us  |  p99: " << get_percentile(gpu_transfer_times, 99) << " us" << std::endl;
    std::cout << "GPU Kernel Compute latency:" << std::endl;
    std::cout << "  p50: " << get_percentile(gpu_compute_times, 50) << " us  |  p95: " << get_percentile(gpu_compute_times, 95) << " us  |  p99: " << get_percentile(gpu_compute_times, 99) << " us" << std::endl;
    std::cout << "GPU Combined (Memcpy + Compute) latency:" << std::endl;
    std::cout << "  p50: " << get_percentile(gpu_total_times, 50) << " us  |  p95: " << get_percentile(gpu_total_times, 95) << " us  |  p99: " << get_percentile(gpu_total_times, 99) << " us" << std::endl;
    std::cout << "CPU Int8 Reference Baseline latency:" << std::endl;
    std::cout << "  p50: " << get_percentile(cpu_times, 50) << " us  |  p95: " << get_percentile(cpu_times, 95) << " us  |  p99: " << get_percentile(cpu_times, 99) << " us" << std::endl;

    for (auto& layer : layers) {
        cudaFree(layer.d_weights);
        cudaFree(layer.d_bias);
    }

    cudaEventDestroy(start_transfer);
    cudaEventDestroy(start_compute);
    cudaEventDestroy(end_compute);
    cudaEventDestroy(end_transfer);

    return 0;
}
