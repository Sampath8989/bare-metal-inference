/*
 * gpu_benchmark/main.cu
 *
 * Master GPU Matrix Optimization Benchmark
 * Benchmarks:
 *   1. Baseline (Naive FP32 CUDA)
 *   2. FP16 (Native Half-Precision CUDA)
 *   3. Shared Memory (16x16 2D Tiled FP32)
 *   4. WMMA Tensor Cores (Hardware 16x16x16 WMMA Fragments)
 *   5. Kernel Fusion (GEMM + ReLU Activation Epilogue)
 *   6. Combined (FP16 + Shared Memory + WMMA + Kernel Fusion)
 *
 * Matrix Sizes: 128x128, 256x256, 512x512, 1024x1024, 2048x2048
 *
 * Compile: nvcc -O3 -arch=sm_75 -std=c++17 main.cu -o benchmark_gpu
 * Run:     ./benchmark_gpu
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "baseline.cu"
#include "fp16.cu"
#include "shared_memory.cu"
#include "wmma.cu"
#include "kernel_fusion.cu"
#include "combined.cu"

struct GpuBenchmarkResult {
    int size;
    std::string opt_name;
    float p50_us;
    float p95_us;
    float p99_us;
    float speedup;
    float max_error;
};

void generate_random_matrix(float* ptr, size_t n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (size_t i = 0; i < n; i++) {
        ptr[i] = dist(gen);
    }
}

void convert_fp32_to_fp16(const float* src, half* dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = __float2half(src[i]);
    }
}

void compute_cpu_baseline_ref(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0f;
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "Error: No CUDA capable GPU detected!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "============================================================\n";
    std::cout << "GPU OPTIMIZATION BENCHMARK\n";
    std::cout << "Device: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << "============================================================\n\n";

    std::vector<int> matrix_sizes = {128, 256, 512, 1024, 2048};
    std::vector<GpuBenchmarkResult> all_results;
    std::vector<std::pair<float, float>> baseline_vs_combined;

    const int WARMUP_RUNS = 20;
    const int TIMED_RUNS  = 100;

    for (int N : matrix_sizes) {
        size_t count = (size_t)N * N;
        size_t bytes_f32 = count * sizeof(float);
        size_t bytes_f16 = count * sizeof(half);

        // Host memory allocation
        std::vector<float> h_A_f32(count), h_B_f32(count);
        generate_random_matrix(h_A_f32.data(), count, 42);
        generate_random_matrix(h_B_f32.data(), count, 1337);

        std::vector<half> h_A_f16(count), h_B_f16(count);
        convert_fp32_to_fp16(h_A_f32.data(), h_A_f16.data(), count);
        convert_fp32_to_fp16(h_B_f32.data(), h_B_f16.data(), count);

        std::vector<float> h_ref_f32(count, 0.0f);
        compute_cpu_baseline_ref(h_A_f32.data(), h_B_f32.data(), h_ref_f32.data(), N);

        std::vector<float> h_out_f32(count, 0.0f);
        std::vector<half>  h_out_f16(count);

        // Device memory allocation
        float *d_A_f32, *d_B_f32, *d_C_f32;
        half  *d_A_f16, *d_B_f16, *d_C_f16;
        cudaMalloc(&d_A_f32, bytes_f32);
        cudaMalloc(&d_B_f32, bytes_f32);
        cudaMalloc(&d_C_f32, bytes_f32);
        cudaMalloc(&d_A_f16, bytes_f16);
        cudaMalloc(&d_B_f16, bytes_f16);
        cudaMalloc(&d_C_f16, bytes_f16);

        cudaMemcpy(d_A_f32, h_A_f32.data(), bytes_f32, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_f32, h_B_f32.data(), bytes_f32, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_f16, h_A_f16.data(), bytes_f16, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_f16, h_B_f16.data(), bytes_f16, cudaMemcpyHostToDevice);

        // 1. Baseline FP32
        float err_base = 0.0f, p95_base = 0.0f, p99_base = 0.0f;
        float lat_base = run_baseline(N, d_A_f32, d_B_f32, d_C_f32, h_out_f32.data(), h_ref_f32.data(),
                                      err_base, p95_base, p99_base, WARMUP_RUNS, TIMED_RUNS);
        all_results.push_back({N, "Baseline", lat_base, p95_base, p99_base, 1.0f, err_base});

        // 2. FP16
        float err_fp16 = 0.0f, p95_fp16 = 0.0f, p99_fp16 = 0.0f;
        float lat_fp16 = run_fp16(N, d_A_f16, d_B_f16, d_C_f16, h_out_f16.data(), h_ref_f32.data(),
                                  err_fp16, p95_fp16, p99_fp16, WARMUP_RUNS, TIMED_RUNS);
        all_results.push_back({N, "FP16", lat_fp16, p95_fp16, p99_fp16, lat_base / lat_fp16, err_fp16});

        // 3. Shared Memory
        float err_smem = 0.0f, p95_smem = 0.0f, p99_smem = 0.0f;
        float lat_smem = run_shared_memory(N, d_A_f32, d_B_f32, d_C_f32, h_out_f32.data(), h_ref_f32.data(),
                                           err_smem, p95_smem, p99_smem, WARMUP_RUNS, TIMED_RUNS);
        all_results.push_back({N, "Shared Memory", lat_smem, p95_smem, p99_smem, lat_base / lat_smem, err_smem});

        // 4. WMMA Tensor Cores
        float err_wmma = 0.0f, p95_wmma = 0.0f, p99_wmma = 0.0f;
        float lat_wmma = run_wmma(N, d_A_f16, d_B_f16, d_C_f32, h_out_f32.data(), h_ref_f32.data(),
                                  err_wmma, p95_wmma, p99_wmma, WARMUP_RUNS, TIMED_RUNS);
        all_results.push_back({N, "WMMA Tensor Cores", lat_wmma, p95_wmma, p99_wmma, lat_base / lat_wmma, err_wmma});

        // 5. Kernel Fusion
        float err_fuse = 0.0f, p95_fuse = 0.0f, p99_fuse = 0.0f;
        float lat_fuse = run_kernel_fusion(N, d_A_f32, d_B_f32, d_C_f32, h_out_f32.data(), h_ref_f32.data(),
                                           err_fuse, p95_fuse, p99_fuse, WARMUP_RUNS, TIMED_RUNS);
        all_results.push_back({N, "Kernel Fusion", lat_fuse, p95_fuse, p99_fuse, lat_base / lat_fuse, err_fuse});

        // 6. FINAL COMBINED (FP16 + Shared Memory + WMMA + Kernel Fusion)
        float err_comb = 0.0f, p95_comb = 0.0f, p99_comb = 0.0f;
        float lat_comb = run_combined(N, d_A_f16, d_B_f16, d_C_f32, h_out_f32.data(), h_ref_f32.data(),
                                      err_comb, p95_comb, p99_comb, WARMUP_RUNS, TIMED_RUNS);
        float speedup_comb = lat_base / lat_comb;
        all_results.push_back({N, "Combined", lat_comb, p95_comb, p99_comb, speedup_comb, err_comb});
        baseline_vs_combined.push_back({lat_base, lat_comb});

        // Terminal Block Output
        std::cout << N << " × " << N << "\n\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Baseline FP32       : " << std::setw(8) << lat_base << " us\n";
        std::cout << "FP16                : " << std::setw(8) << lat_fp16 << " us\n";
        std::cout << "Shared Memory       : " << std::setw(8) << lat_smem << " us\n";
        std::cout << "WMMA Tensor Cores   : " << std::setw(8) << lat_wmma << " us\n";
        std::cout << "Kernel Fusion       : " << std::setw(8) << lat_fuse << " us\n";
        std::cout << "------------------------------------------------------------\n";
        std::cout << "FINAL COMBINED      : " << std::setw(8) << lat_comb << " us\n";
        std::cout << "Combined Speedup    : " << speedup_comb << "×\n\n";

        cudaFree(d_A_f32); cudaFree(d_B_f32); cudaFree(d_C_f32);
        cudaFree(d_A_f16); cudaFree(d_B_f16); cudaFree(d_C_f16);
    }

    // Save CSV to gpu_results.csv
    std::ofstream csv_out("gpu_results.csv");
    if (csv_out.is_open()) {
        csv_out << "matrix_size,optimization,p50_us,p95_us,p99_us,speedup,max_error\n";
        for (const auto& r : all_results) {
            csv_out << r.size << "," << r.opt_name << ","
                    << std::fixed << std::setprecision(2)
                    << r.p50_us << "," << r.p95_us << "," << r.p99_us << ","
                    << r.speedup << "," << std::scientific << std::setprecision(4)
                    << r.max_error << "\n";
        }
        csv_out.close();
    }

    // Save Human-Readable Table to gpu_results.txt
    std::ofstream txt_out("gpu_results.txt");
    if (txt_out.is_open()) {
        txt_out << "========================================================================================\n";
        txt_out << "GPU OPTIMIZATION BENCHMARK RESULTS — " << prop.name << "\n";
        txt_out << "========================================================================================\n\n";
        txt_out << std::left << std::setw(12) << "Size"
                << std::setw(22) << "Optimization"
                << std::setw(14) << "p50 (us)"
                << std::setw(14) << "p95 (us)"
                << std::setw(14) << "p99 (us)"
                << std::setw(12) << "Speedup"
                << "Max Error\n";
        txt_out << "----------------------------------------------------------------------------------------\n";
        for (const auto& r : all_results) {
            txt_out << std::left << std::setw(12) << (std::to_string(r.size) + "x" + std::to_string(r.size))
                    << std::setw(22) << r.opt_name
                    << std::fixed << std::setprecision(2)
                    << std::setw(14) << r.p50_us
                    << std::setw(14) << r.p95_us
                    << std::setw(14) << r.p99_us
                    << std::setw(12) << (std::to_string(r.speedup).substr(0, 4) + "x")
                    << std::scientific << std::setprecision(3) << r.max_error << "\n";
            if (r.opt_name == "Combined") {
                txt_out << "----------------------------------------------------------------------------------------\n";
            }
        }

        txt_out << "\n============================================================\n";
        txt_out << "FINAL GPU SUMMARY: BASELINE vs COMBINED\n";
        txt_out << "============================================================\n\n";
        txt_out << std::left << std::setw(16) << "Matrix Size"
                << std::setw(16) << "Baseline FP32"
                << std::setw(16) << "Final Combined"
                << "Speedup\n";
        txt_out << "------------------------------------------------------------\n";
        for (size_t i = 0; i < matrix_sizes.size(); i++) {
            std::string size_label = std::to_string(matrix_sizes[i]) + " × " + std::to_string(matrix_sizes[i]);
            std::string base_label = std::to_string((int)baseline_vs_combined[i].first) + " us";
            std::string comb_label = std::to_string((int)baseline_vs_combined[i].second) + " us";
            float sp = baseline_vs_combined[i].first / baseline_vs_combined[i].second;

            txt_out << std::left << std::setw(16) << size_label
                    << std::setw(16) << base_label
                    << std::setw(16) << comb_label
                    << std::fixed << std::setprecision(2) << sp << "×\n";
        }
        txt_out.close();
    }

    // Terminal Summary
    std::cout << "============================================================\n";
    std::cout << "FINAL GPU RESULT\n";
    std::cout << "============================================================\n\n";
    std::cout << std::left << std::setw(16) << "Matrix Size"
              << std::setw(15) << "Baseline"
              << std::setw(15) << "Combined"
              << "Speedup\n\n";

    for (size_t i = 0; i < matrix_sizes.size(); i++) {
        std::string size_label = std::to_string(matrix_sizes[i]) + " × " + std::to_string(matrix_sizes[i]);
        std::string base_label = std::to_string((int)baseline_vs_combined[i].first) + " us";
        std::string comb_label = std::to_string((int)baseline_vs_combined[i].second) + " us";
        float sp = baseline_vs_combined[i].first / baseline_vs_combined[i].second;

        std::cout << std::left << std::setw(16) << size_label
                  << std::setw(15) << base_label
                  << std::setw(15) << comb_label
                  << std::fixed << std::setprecision(2) << sp << "×\n";
    }
    std::cout << "\n============================================================\n";

    return 0;
}
