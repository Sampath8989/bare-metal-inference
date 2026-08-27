// GPU int8 inference engine + benchmark (Scaled: 512x512 and 1024x1024).
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>

struct LayerWeights {
    int rows;
    int cols;
    float scale;
    std::vector<int8_t> h_weights;
    std::vector<float> h_bias;
    int8_t* d_weights;
    float* d_bias;
};

__global__ void quantize_kernel(const float* input, int8_t* output, float* out_scale, int size) {
    extern __shared__ float shared_max[];
    int tid = threadIdx.x;

    float val = (tid < size) ? input[tid] : 0.0f;
    shared_max[tid] = fabsf(val);
    __syncthreads();

    int pow2 = 1;
    while (pow2 < blockDim.x) {
        pow2 <<= 1;
    }

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

__global__ void matmul_int8_kernel(const int8_t* A, const int8_t* B, int32_t* C_acc, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // row (batch)
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // col (output feature)

    if (i < M && j < N) {
        int32_t accum = 0;
        for (int q = 0; q < K; q++) {
            accum += static_cast<int32_t>(A[i * K + q]) * static_cast<int32_t>(B[q * N + j]);
        }
        C_acc[i * N + j] = accum;
    }
}

__global__ void post_process_kernel(const int32_t* C_acc, const float* bias, float* output, const float* input_scale, float weight_scale, int M, int N, bool apply_relu) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        int idx = i * N + j;
        float combined_scale = (*input_scale) * weight_scale;
        float val = static_cast<float>(C_acc[idx]) * combined_scale;
        val += bias[j];
        if (apply_relu) {
            val = fmaxf(0.0f, val);
        }
        output[idx] = val;
    }
}

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
    int input_dim, hidden1, hidden2, output_dim;

    GPUContext(int batch_size, int in_dim, int h1, int h2, int out_dim)
        : m(batch_size), input_dim(in_dim), hidden1(h1), hidden2(h2), output_dim(out_dim) {
        cudaMalloc(&d_input_f, m * input_dim * sizeof(float));
        cudaMalloc(&d_input_q, m * input_dim * sizeof(int8_t));
        cudaMalloc(&d_input_scale, sizeof(float));

        cudaMalloc(&d_act0_f, m * hidden1 * sizeof(float));
        cudaMalloc(&d_act0_q, m * hidden1 * sizeof(int8_t));
        cudaMalloc(&d_act0_scale, sizeof(float));

        cudaMalloc(&d_act1_f, m * hidden2 * sizeof(float));
        cudaMalloc(&d_act1_q, m * hidden2 * sizeof(int8_t));
        cudaMalloc(&d_act1_scale, sizeof(float));

        cudaMalloc(&d_act2_f, m * output_dim * sizeof(float));

        cudaMalloc(&d_acc0, m * hidden1 * sizeof(int32_t));
        cudaMalloc(&d_acc1, m * hidden2 * sizeof(int32_t));
        cudaMalloc(&d_acc2, m * output_dim * sizeof(int32_t));
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

void forward_gpu(const std::vector<LayerWeights>& layers, GPUContext& ctx) {
    int m = ctx.m;

    // Layer 0: input_dim → hidden1
    {
        int size0 = m * ctx.input_dim;
        int block = std::min(1024, std::max(32, size0));
        quantize_kernel<<<1, block, block * sizeof(float)>>>(ctx.d_input_f, ctx.d_input_q, ctx.d_input_scale, size0);
        dim3 blockDim(16, 16);
        dim3 gridDim((ctx.hidden1 + 15) / 16, (m + 15) / 16);
        matmul_int8_kernel<<<gridDim, blockDim>>>(ctx.d_input_q, layers[0].d_weights, ctx.d_acc0, m, ctx.hidden1, ctx.input_dim);
        post_process_kernel<<<gridDim, blockDim>>>(ctx.d_acc0, layers[0].d_bias, ctx.d_act0_f, ctx.d_input_scale, layers[0].scale, m, ctx.hidden1, true);
    }

    // Layer 1: hidden1 → hidden2
    {
        int size1 = m * ctx.hidden1;
        int block = std::min(1024, std::max(32, size1));
        quantize_kernel<<<1, block, block * sizeof(float)>>>(ctx.d_act0_f, ctx.d_act0_q, ctx.d_act0_scale, size1);
        dim3 blockDim(16, 16);
        dim3 gridDim((ctx.hidden2 + 15) / 16, (m + 15) / 16);
        matmul_int8_kernel<<<gridDim, blockDim>>>(ctx.d_act0_q, layers[1].d_weights, ctx.d_acc1, m, ctx.hidden2, ctx.hidden1);
        post_process_kernel<<<gridDim, blockDim>>>(ctx.d_acc1, layers[1].d_bias, ctx.d_act1_f, ctx.d_act0_scale, layers[1].scale, m, ctx.hidden2, true);
    }

    // Layer 2: hidden2 → output_dim
    {
        int size2 = m * ctx.hidden2;
        int block = std::min(1024, std::max(32, size2));
        quantize_kernel<<<1, block, block * sizeof(float)>>>(ctx.d_act1_f, ctx.d_act1_q, ctx.d_act1_scale, size2);
        dim3 blockDim(16, 16);
        dim3 gridDim((ctx.output_dim + 15) / 16, (m + 15) / 16);
        matmul_int8_kernel<<<gridDim, blockDim>>>(ctx.d_act1_q, layers[2].d_weights, ctx.d_acc2, m, ctx.output_dim, ctx.hidden2);
        post_process_kernel<<<gridDim, blockDim>>>(ctx.d_acc2, layers[2].d_bias, ctx.d_act2_f, ctx.d_act1_scale, layers[2].scale, m, ctx.output_dim, true);
    }
}

int main(int argc, char* argv[]) {
    int scale = 512;
    if (argc >= 2) {
        int s = std::atoi(argv[1]);
        if (s > 0) scale = s;
    }

    std::cout << "=== V4 CUDA INT8 Inference Engine (Scaled: " << scale << "x" << scale << ") ===" << std::endl;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "WARNING: No CUDA device available or driver issue (" << cudaGetErrorString(err) << "). Skipping GPU execution.\n";
        return 0;
    }

    int in_dim = scale;
    int h1_dim = scale;
    int h2_dim = scale;
    int out_dim = scale;

    std::cout << "Architecture: " << in_dim << " -> " << h1_dim << " -> " << h2_dim << " -> " << out_dim << std::endl;

    std::vector<LayerWeights> layers(3);
    int dims[4] = {in_dim, h1_dim, h2_dim, out_dim};

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dis(-64, 64);

    for (int l = 0; l < 3; l++) {
        int rows = dims[l];
        int cols = dims[l+1];
        layers[l].rows = rows;
        layers[l].cols = cols;
        layers[l].scale = 0.001f;
        layers[l].h_weights.resize(rows * cols);
        for (int i = 0; i < rows * cols; i++) layers[l].h_weights[i] = static_cast<int8_t>(dis(gen));
        layers[l].h_bias.resize(cols, 0.0f);

        cudaMalloc(&layers[l].d_weights, rows * cols * sizeof(int8_t));
        cudaMalloc(&layers[l].d_bias, cols * sizeof(float));
        cudaMemcpy(layers[l].d_weights, layers[l].h_weights.data(), rows * cols * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(layers[l].d_bias, layers[l].h_bias.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
    }

    GPUContext gpu_ctx(1, in_dim, h1_dim, h2_dim, out_dim);

    std::vector<float> h_input(in_dim, 1.0f);
    std::vector<float> h_output_gpu(out_dim, 0.0f);

    cudaMemcpy(gpu_ctx.d_input_f, h_input.data(), in_dim * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 50; i++) {
        forward_gpu(layers, gpu_ctx);
    }
    cudaDeviceSynchronize();

    const int RUNS = 500;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> gpu_times_us(RUNS);

    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start);
        forward_gpu(layers, gpu_ctx);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        gpu_times_us[r] = ms * 1000.0f;
    }

    cudaMemcpy(h_output_gpu.data(), gpu_ctx.d_act2_f, out_dim * sizeof(float), cudaMemcpyDeviceToHost);

    std::sort(gpu_times_us.begin(), gpu_times_us.end());
    float p50 = gpu_times_us[RUNS * 50 / 100];
    float p95 = gpu_times_us[RUNS * 95 / 100];
    float p99 = gpu_times_us[RUNS * 99 / 100];

    std::cout << "Size: " << scale << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "p50: " << p50 << " us\n";
    std::cout << "p95: " << p95 << " us\n";
    std::cout << "p99: " << p99 << " us\n";

    std::cout << "Output sample (first 5): [";
    for (int i = 0; i < std::min(5, out_dim); i++) {
        std::cout << std::fixed << std::setprecision(4) << h_output_gpu[i];
        if (i < std::min(5, out_dim) - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";

    for (auto& layer : layers) {
        cudaFree(layer.d_weights);
        cudaFree(layer.d_bias);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
