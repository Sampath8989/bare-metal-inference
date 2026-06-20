/*
 * v3_simd/bench_simd.cpp
 *
 * Benchmark: scalar int8 vs SIMD int8 vs float32.
 * Uses pre-allocated contexts to completely isolate dynamic allocator latency.
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <string>
#include <cstdint>
#include <numeric>
#include <cstdlib>
#include <immintrin.h>

// Baseline structures (Tensor, Layer, relu, matmul)
#include "../src/tensor.hpp"
#include "../src/Layer.hpp"
#include "../src/relu.hpp"

using Clock = std::chrono::high_resolution_clock;

// ============================================================
// Pre-allocated Contexts (allocation-free timed runs)
// ============================================================

struct Float32Context {
    Tensor act0, act1, act2;
    Float32Context() : act0(1, 64), act1(1, 32), act2(1, 10) {}
};

struct Int8Tensor {
    int8_t* data;
    int rows, cols;
    float scale;

    Int8Tensor(int r, int c, float s = 1.0f) : rows(r), cols(c), scale(s) {
        size_t n = static_cast<size_t>(r) * c;
        size_t aligned = (n + 63) & ~63ULL;
        data = static_cast<int8_t*>(std::aligned_alloc(64, aligned));
        if (!data) throw std::bad_alloc();
        std::memset(data, 0, n);
    }
    Int8Tensor(const Int8Tensor&) = delete;
    Int8Tensor& operator=(const Int8Tensor&) = delete;
    Int8Tensor(Int8Tensor&& o) noexcept
        : data(o.data), rows(o.rows), cols(o.cols), scale(o.scale) {
        o.data = nullptr; o.rows = 0; o.cols = 0;
    }
    Int8Tensor& operator=(Int8Tensor&& o) noexcept {
        if (this != &o) {
            std::free(data);
            data = o.data; rows = o.rows; cols = o.cols; scale = o.scale;
            o.data = nullptr; o.rows = 0; o.cols = 0;
        }
        return *this;
    }
    ~Int8Tensor() { if (data) std::free(data); }
};

struct Int8Layer {
    Int8Tensor weights;
    std::vector<float> bias;
    int in_size, out_size;

    Int8Layer(int in_sz, int out_sz, float sc)
        : weights(in_sz, out_sz, sc),
          bias(static_cast<size_t>(out_sz), 0.0f),
          in_size(in_sz), out_size(out_sz) {}
    Int8Layer(Int8Layer&&) = default;
};

struct Int8Context {
    Int8Tensor q_in;
    Tensor act0;
    Int8Tensor q_act0;
    Tensor act1;
    Int8Tensor q_act1;
    Tensor act2;
    std::vector<int32_t> acc0;
    std::vector<int32_t> acc1;
    std::vector<int32_t> acc2;

    Int8Context()
        : q_in(1, 128),
          act0(1, 64), q_act0(1, 64),
          act1(1, 32), q_act1(1, 32),
          act2(1, 10),
          acc0(64, 0), acc1(32, 0), acc2(10, 0) {}
};

// ============================================================
// Loaders
// ============================================================

std::vector<Layer> load_float32_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<Layer> layers;
    while (file.peek() != EOF) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        if (file.gcount() == 0) break;
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        Layer layer(rows, cols);
        file.read(reinterpret_cast<char*>(layer.weights.data), static_cast<size_t>(rows) * cols * sizeof(float));
        int bs; file.read(reinterpret_cast<char*>(&bs), sizeof(bs));
        file.read(reinterpret_cast<char*>(layer.bias.data), static_cast<size_t>(bs) * sizeof(float));
        layers.push_back(std::move(layer));
    }
    return layers;
}

std::vector<Int8Layer> load_int8_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<Int8Layer> layers;
    while (file.peek() != EOF) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        if (file.gcount() == 0) break;
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        float scale;
        file.read(reinterpret_cast<char*>(&scale), sizeof(scale));
        Int8Layer layer(rows, cols, scale);
        file.read(reinterpret_cast<char*>(layer.weights.data), static_cast<size_t>(rows) * cols);
        int bs;
        file.read(reinterpret_cast<char*>(&bs), sizeof(bs));
        file.read(reinterpret_cast<char*>(layer.bias.data()), static_cast<size_t>(bs) * sizeof(float));
        layers.push_back(std::move(layer));
    }
    return layers;
}

// ============================================================
// Inplace arithmetic kernels
// ============================================================

void quantize_int8_inplace(const Tensor& input, Int8Tensor& out) {
    float max_abs = 0.0f;
    for (int i = 0; i < input.size(); i++)
        max_abs = std::max(max_abs, std::abs(input.data[i]));
    out.scale = (max_abs == 0.0f) ? 1.0f : max_abs / 127.0f;
    for (int i = 0; i < input.size(); i++) {
        float s = input.data[i] / out.scale;
        out.data[i] = static_cast<int8_t>(std::clamp(std::round(s), -128.0f, 127.0f));
    }
}

void matmul_int8_scalar_inplace(const Int8Tensor& A, const Int8Tensor& B, Tensor& C, std::vector<int32_t>& acc) {
    int m = A.rows, n = B.cols, k = A.cols;
    float cs = A.scale * B.scale;
    std::fill(acc.begin(), acc.end(), 0);

    for (int i = 0; i < m; i++) {
        for (int q = 0; q < k; q++) {
            int8_t a = A.data[i * k + q];
            for (int j = 0; j < n; j++) {
                acc[i * n + j] += static_cast<int32_t>(a) * static_cast<int32_t>(B.data[q * n + j]);
            }
        }
    }

    for (int i = 0; i < m * n; i++)
        C.data[i] = static_cast<float>(acc[i]) * cs;
}

void matmul_int8_simd_inplace(const Int8Tensor& A, const Int8Tensor& B, Tensor& C, std::vector<int32_t>& acc) {
    int m = A.rows, n = B.cols, k = A.cols;
    float cs = A.scale * B.scale;
    std::fill(acc.begin(), acc.end(), 0);

    for (int i = 0; i < m; i++) {
        for (int q = 0; q < k; q++) {
            int8_t a_val = A.data[i * k + q];
            __m256i a16 = _mm256_set1_epi16(static_cast<int16_t>(a_val));

            int j = 0;
            for (; j + 32 <= n; j += 32) {
                // Corrected offset: load weights from B row q
                __m256i b8 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(B.data + q * n + j));
                __m256i b16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b8));
                __m256i b16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b8, 1));
                __m256i p16_lo = _mm256_mullo_epi16(a16, b16_lo);
                __m256i p16_hi = _mm256_mullo_epi16(a16, b16_hi);
                __m256i p32_0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(p16_lo));
                __m256i p32_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p16_lo, 1));
                __m256i p32_2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(p16_hi));
                __m256i p32_3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p16_hi, 1));

                __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + i*n + j));
                __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + i*n + j + 8));
                __m256i a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + i*n + j + 16));
                __m256i a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + i*n + j + 24));

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + i*n + j),     _mm256_add_epi32(a0, p32_0));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + i*n + j + 8),  _mm256_add_epi32(a1, p32_1));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + i*n + j + 16), _mm256_add_epi32(a2, p32_2));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + i*n + j + 24), _mm256_add_epi32(a3, p32_3));
            }
            // Corrected offset: load scalar weights from B row q
            for (; j < n; j++)
                acc[i * n + j] += static_cast<int32_t>(a_val) * static_cast<int32_t>(B.data[q * n + j]);
        }
    }
    for (int i = 0; i < m * n; i++)
        C.data[i] = static_cast<float>(acc[i]) * cs;
}

// ============================================================
// Inplace forward passes
// ============================================================

void float32_forward_inplace(const std::vector<Layer>& layers, const Tensor& input, Float32Context& ctx) {
    std::fill(ctx.act0.data, ctx.act0.data + 64, 0.0f);
    for (int q = 0; q < 128; q++) { float a = input.data[q]; for (int j = 0; j < 64; j++) ctx.act0.data[j] += a * layers[0].weights.data[q*64+j]; }
    for (int i = 0; i < 64; i++) ctx.act0.data[i] = std::max(0.0f, ctx.act0.data[i] + layers[0].bias.data[i]);

    std::fill(ctx.act1.data, ctx.act1.data + 32, 0.0f);
    for (int q = 0; q < 64; q++) { float a = ctx.act0.data[q]; for (int j = 0; j < 32; j++) ctx.act1.data[j] += a * layers[1].weights.data[q*32+j]; }
    for (int i = 0; i < 32; i++) ctx.act1.data[i] = std::max(0.0f, ctx.act1.data[i] + layers[1].bias.data[i]);

    std::fill(ctx.act2.data, ctx.act2.data + 10, 0.0f);
    for (int q = 0; q < 32; q++) { float a = ctx.act1.data[q]; for (int j = 0; j < 10; j++) ctx.act2.data[j] += a * layers[2].weights.data[q*10+j]; }
    for (int i = 0; i < 10; i++) ctx.act2.data[i] = std::max(0.0f, ctx.act2.data[i] + layers[2].bias.data[i]);
}

void forward_scalar_inplace(const std::vector<Int8Layer>& layers, const Tensor& input, Int8Context& ctx) {
    quantize_int8_inplace(input, ctx.q_in);
    matmul_int8_scalar_inplace(ctx.q_in, layers[0].weights, ctx.act0, ctx.acc0);
    for (int i = 0; i < 64; i++) ctx.act0.data[i] = std::max(0.0f, ctx.act0.data[i] + layers[0].bias[i]);

    quantize_int8_inplace(ctx.act0, ctx.q_act0);
    matmul_int8_scalar_inplace(ctx.q_act0, layers[1].weights, ctx.act1, ctx.acc1);
    for (int i = 0; i < 32; i++) ctx.act1.data[i] = std::max(0.0f, ctx.act1.data[i] + layers[1].bias[i]);

    quantize_int8_inplace(ctx.act1, ctx.q_act1);
    matmul_int8_scalar_inplace(ctx.q_act1, layers[2].weights, ctx.act2, ctx.acc2);
    for (int i = 0; i < 10; i++) ctx.act2.data[i] = std::max(0.0f, ctx.act2.data[i] + layers[2].bias[i]);
}

void forward_simd_inplace(const std::vector<Int8Layer>& layers, const Tensor& input, Int8Context& ctx) {
    quantize_int8_inplace(input, ctx.q_in);
    matmul_int8_simd_inplace(ctx.q_in, layers[0].weights, ctx.act0, ctx.acc0);
    for (int i = 0; i < 64; i++) ctx.act0.data[i] = std::max(0.0f, ctx.act0.data[i] + layers[0].bias[i]);

    quantize_int8_inplace(ctx.act0, ctx.q_act0);
    matmul_int8_simd_inplace(ctx.q_act0, layers[1].weights, ctx.act1, ctx.acc1);
    for (int i = 0; i < 32; i++) ctx.act1.data[i] = std::max(0.0f, ctx.act1.data[i] + layers[1].bias[i]);

    quantize_int8_inplace(ctx.act1, ctx.q_act1);
    matmul_int8_simd_inplace(ctx.q_act1, layers[2].weights, ctx.act2, ctx.acc2);
    for (int i = 0; i < 10; i++) ctx.act2.data[i] = std::max(0.0f, ctx.act2.data[i] + layers[2].bias[i]);
}

double percentile(const std::vector<double>& sorted, double p) {
    size_t idx = static_cast<size_t>(sorted.size() * p / 100.0);
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return sorted[idx];
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== SIMD Benchmark: scalar int8 vs SIMD int8 vs float32 ===\n\n";

    auto f32_layers = load_float32_weights("weights.bin");
    auto i8_layers  = load_int8_weights("v2_int8/weights_int8.bin");

    Tensor input(1, 128);
    for (int i = 0; i < input.size(); i++) input.data[i] = 1.0f;

    Float32Context f32_ctx;
    Int8Context i8_ctx;

    // Warmup
    for (int w = 0; w < 500; w++) {
        float32_forward_inplace(f32_layers, input, f32_ctx);
        forward_simd_inplace(i8_layers, input, i8_ctx);
    }

    volatile float sink = 0.0f;

    // ══ CROSS-CHECK 1: Batch=100 (10,000 samples) ══
    int batch = 100;
    constexpr int SAMPLES = 10000;
    std::vector<double> f32_lat(SAMPLES), scalar_lat(SAMPLES), simd_lat(SAMPLES);

    std::cout << "--- Cross-check 1: batch=100 (" << SAMPLES << " samples) ---\n";

    for (int s = 0; s < SAMPLES; s++) {
        auto t0 = Clock::now();
        for (int b = 0; b < batch; b++) float32_forward_inplace(f32_layers, input, f32_ctx);
        auto t1 = Clock::now();
        sink += f32_ctx.act2.data[0];
        f32_lat[s] = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()) / batch / 1000.0;
    }
    for (int s = 0; s < SAMPLES; s++) {
        auto t0 = Clock::now();
        for (int b = 0; b < batch; b++) forward_scalar_inplace(i8_layers, input, i8_ctx);
        auto t1 = Clock::now();
        sink += i8_ctx.act2.data[0];
        scalar_lat[s] = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()) / batch / 1000.0;
    }
    for (int s = 0; s < SAMPLES; s++) {
        auto t0 = Clock::now();
        for (int b = 0; b < batch; b++) forward_simd_inplace(i8_layers, input, i8_ctx);
        auto t1 = Clock::now();
        sink += i8_ctx.act2.data[0];
        simd_lat[s] = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()) / batch / 1000.0;
    }

    auto f32s = f32_lat; std::sort(f32s.begin(), f32s.end());
    auto scs  = scalar_lat; std::sort(scs.begin(), scs.end());
    auto sis  = simd_lat; std::sort(sis.begin(), sis.end());

    double f32_p50=percentile(f32s,50), f32_p95=percentile(f32s,95), f32_p99=percentile(f32s,99);
    double sc_p50=percentile(scs,50),  sc_p95=percentile(scs,95),   sc_p99=percentile(scs,99);
    double si_p50=percentile(sis,50),  si_p95=percentile(sis,95),   si_p99=percentile(sis,99);

    std::cout << "\n  Batch=100 results (per-iteration latency):\n";
    std::cout << "    float32     p50: " << std::fixed << std::setprecision(3) << f32_p50 << " us  p95: " << f32_p95 << "  p99: " << f32_p99 << "\n";
    std::cout << "    int8 scalar p50: " << sc_p50  << " us  p95: " << sc_p95  << "  p99: " << sc_p99  << "\n";
    std::cout << "    int8 SIMD   p50: " << si_p50  << " us  p95: " << si_p95  << "  p99: " << si_p99  << "\n";
    std::cout << "    SIMD vs scalar speedup (p50): " << std::setprecision(2) << sc_p50/si_p50 << "x\n";
    std::cout << "    SIMD vs float32 speedup (p50): " << f32_p50/si_p50 << "x\n";

    std::ofstream csv("benchmarks/latency_results_v3.csv");
    csv << "iteration,float32_us,scalar_int8_us,simd_int8_us\n";
    for (int i = 0; i < SAMPLES; i++)
        csv << (i+1) << "," << std::fixed << std::setprecision(4) << f32_lat[i] << "," << scalar_lat[i] << "," << simd_lat[i] << "\n";
    csv.close();
    std::cout << "\n  ✓ Wrote benchmarks/latency_results_v3.csv\n";

    // ══ CROSS-CHECK 2: Batch=1 (1,000 samples) ══
    constexpr int SINGLE = 1000;
    std::vector<double> sc1(SINGLE), si1(SINGLE);

    std::cout << "\n--- Cross-check 2: batch=1 (" << SINGLE << " samples) ---\n";

    for (int s = 0; s < SINGLE; s++) {
        auto t0 = Clock::now(); forward_scalar_inplace(i8_layers, input, i8_ctx); auto t1 = Clock::now();
        sink += i8_ctx.act2.data[0];
        sc1[s] = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()) / 1000.0;
    }
    for (int s = 0; s < SINGLE; s++) {
        auto t0 = Clock::now(); forward_simd_inplace(i8_layers, input, i8_ctx); auto t1 = Clock::now();
        sink += i8_ctx.act2.data[0];
        si1[s] = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()) / 1000.0;
    }

    auto sc1s = sc1; std::sort(sc1s.begin(), sc1s.end());
    auto si1s = si1; std::sort(si1s.begin(), si1s.end());
    double sc1_p50=percentile(sc1s,50), sc1_p95=percentile(sc1s,95), sc1_p99=percentile(sc1s,99);
    double si1_p50=percentile(si1s,50), si1_p95=percentile(si1s,95), si1_p99=percentile(si1s,99);

    int sc_d=1, si_d=1;
    for (size_t i=1;i<sc1s.size();i++) if(sc1s[i]!=sc1s[i-1]) sc_d++;
    for (size_t i=1;i<si1s.size();i++) if(si1s[i]!=si1s[i-1]) si_d++;

    std::cout << "\n  Batch=1 results (true single-call latency):\n";
    std::cout << "    int8 scalar p50: " << std::fixed << std::setprecision(3) << sc1_p50 << " us  p95: " << sc1_p95 << "  p99: " << sc1_p99 << "\n";
    std::cout << "    int8 SIMD   p50: " << si1_p50 << " us  p95: " << si1_p95 << "  p99: " << si1_p99 << "\n";
    std::cout << "    SIMD vs scalar speedup (p50): " << std::setprecision(2) << sc1_p50/si1_p50 << "x\n";
    std::cout << "    Timer resolution: " << sc_d << " distinct scalar values, " << si_d << " distinct SIMD values out of " << SINGLE << "\n";
    if (sc_d < 20 || si_d < 20)
        std::cout << "    ⚠ Low distinct count — timer resolution may be affecting measurements.\n";
    else
        std::cout << "    ✓ Sufficient distinct values — measurements appear above timer resolution.\n";

    // ══ CROSS-CHECK 3: batch=100 vs batch=1 consistency ══
    std::cout << "\n--- Cross-check 3: batch=100 vs batch=1 consistency ---\n";
    double sr = sc_p50/sc1_p50, sir = si_p50/si1_p50;
    std::cout << "  Scalar: batch=100 p50=" << sc_p50 << "  batch=1 p50=" << sc1_p50 << "  ratio=" << std::setprecision(2) << sr << "x\n";
    std::cout << "  SIMD:   batch=100 p50=" << si_p50 << "  batch=1 p50=" << si1_p50 << "  ratio=" << sir << "x\n";
    if (sr > 0.7 && sr < 1.5 && sir > 0.7 && sir < 1.5)
        std::cout << "  ✓ Both ratios within 0.7x–1.5x — batch averaging is not hiding anomalies.\n";
    else
        std::cout << "  ⚠ Ratio outside 0.7x–1.5x range — may indicate timer or scheduling noise.\n";

    // ══ Final summary ══
    std::cout << "\n=== Summary ===\n";
    std::cout << "  Actual SIMD vs scalar speedup (batch=100, p50): " << std::setprecision(2) << sc_p50/si_p50 << "x\n";
    if (si_p50 >= sc_p50) {
        std::cout << "  SIMD is NOT faster than scalar at this model size.\n";
    } else {
        std::cout << "  SIMD IS faster than scalar at this model size.\n";
    }

    (void)sink;
    return 0;
}
