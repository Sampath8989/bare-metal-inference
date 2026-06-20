/*
 * benchmarks/bench_both.cpp
 *
 * Benchmark harness: runs float32 and int8 forward passes.
 * - Batch=100 timing (10,000 samples) for stable latency distribution
 * - Batch=1 timing (1,000 samples) for true single-call latency
 * - Multi-input accuracy: 10 random inputs, argmax match + error stats
 * Allocation-free timed execution loops using pre-allocated contexts.
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
#include <random>

// Import baseline structures
#include "../src/tensor.hpp"
#include "../src/Layer.hpp"
#include "../src/relu.hpp"

using Clock = std::chrono::high_resolution_clock;

// ============================================================
// Pre-allocated Contexts (allocation-free timed runs)
// ============================================================

struct Float32Context {
    Tensor act0;
    Tensor act1;
    Tensor act2;

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
        o.data = nullptr;
        o.rows = 0; o.cols = 0;
    }
    Int8Tensor& operator=(Int8Tensor&& o) noexcept {
        if (this != &o) {
            std::free(data);
            data = o.data;
            rows = o.rows;
            cols = o.cols;
            scale = o.scale;
            o.data = nullptr;
            o.rows = 0; o.cols = 0;
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
    if (!file.is_open())
        throw std::runtime_error("Cannot open float32 weights: " + path);

    std::vector<Layer> layers;
    while (file.peek() != EOF) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        if (file.gcount() == 0) break;
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        Layer layer(rows, cols);
        size_t nw = static_cast<size_t>(rows) * cols;
        file.read(reinterpret_cast<char*>(layer.weights.data), nw * sizeof(float));

        int bs;
        file.read(reinterpret_cast<char*>(&bs), sizeof(bs));
        file.read(reinterpret_cast<char*>(layer.bias.data), static_cast<size_t>(bs) * sizeof(float));

        layers.push_back(std::move(layer));
    }
    return layers;
}

std::vector<Int8Layer> load_int8_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open int8 weights: " + path);

    std::vector<Int8Layer> layers;
    while (file.peek() != EOF) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        if (file.gcount() == 0) break;
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        float scale;
        file.read(reinterpret_cast<char*>(&scale), sizeof(scale));

        Int8Layer layer(rows, cols, scale);
        size_t nw = static_cast<size_t>(rows) * cols;
        file.read(reinterpret_cast<char*>(layer.weights.data), nw);

        int bs;
        file.read(reinterpret_cast<char*>(&bs), sizeof(bs));
        file.read(reinterpret_cast<char*>(layer.bias.data()), static_cast<size_t>(bs) * sizeof(float));

        layers.push_back(std::move(layer));
    }
    return layers;
}

// ============================================================
// Forward Pass Functions
// ============================================================

void float32_forward_inplace(const std::vector<Layer>& layers, const Tensor& input, Float32Context& ctx) {
    // Layer 0: 128 -> 64
    std::fill(ctx.act0.data, ctx.act0.data + 64, 0.0f);
    for (int q = 0; q < 128; q++) {
        float a = input.data[q];
        for (int j = 0; j < 64; j++) {
            ctx.act0.data[j] += a * layers[0].weights.data[q * 64 + j];
        }
    }
    for (int i = 0; i < 64; i++) {
        ctx.act0.data[i] = std::max(0.0f, ctx.act0.data[i] + layers[0].bias.data[i]);
    }

    // Layer 1: 64 -> 32
    std::fill(ctx.act1.data, ctx.act1.data + 32, 0.0f);
    for (int q = 0; q < 64; q++) {
        float a = ctx.act0.data[q];
        for (int j = 0; j < 32; j++) {
            ctx.act1.data[j] += a * layers[1].weights.data[q * 32 + j];
        }
    }
    for (int i = 0; i < 32; i++) {
        ctx.act1.data[i] = std::max(0.0f, ctx.act1.data[i] + layers[1].bias.data[i]);
    }

    // Layer 2: 32 -> 10
    std::fill(ctx.act2.data, ctx.act2.data + 10, 0.0f);
    for (int q = 0; q < 32; q++) {
        float a = ctx.act1.data[q];
        for (int j = 0; j < 10; j++) {
            ctx.act2.data[j] += a * layers[2].weights.data[q * 10 + j];
        }
    }
    for (int i = 0; i < 10; i++) {
        ctx.act2.data[i] = std::max(0.0f, ctx.act2.data[i] + layers[2].bias.data[i]);
    }
}

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

void matmul_int8_inplace(const Int8Tensor& A, const Int8Tensor& B, Tensor& C, std::vector<int32_t>& acc) {
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

void int8_forward_inplace(const std::vector<Int8Layer>& layers, const Tensor& input, Int8Context& ctx) {
    // Layer 0: 128 -> 64
    quantize_int8_inplace(input, ctx.q_in);
    matmul_int8_inplace(ctx.q_in, layers[0].weights, ctx.act0, ctx.acc0);
    for (int i = 0; i < 64; i++) {
        ctx.act0.data[i] = std::max(0.0f, ctx.act0.data[i] + layers[0].bias[i]);
    }

    // Layer 1: 64 -> 32
    quantize_int8_inplace(ctx.act0, ctx.q_act0);
    matmul_int8_inplace(ctx.q_act0, layers[1].weights, ctx.act1, ctx.acc1);
    for (int i = 0; i < 32; i++) {
        ctx.act1.data[i] = std::max(0.0f, ctx.act1.data[i] + layers[1].bias[i]);
    }

    // Layer 2: 32 -> 10
    quantize_int8_inplace(ctx.act1, ctx.q_act1);
    matmul_int8_inplace(ctx.q_act1, layers[2].weights, ctx.act2, ctx.acc2);
    for (int i = 0; i < 10; i++) {
        ctx.act2.data[i] = std::max(0.0f, ctx.act2.data[i] + layers[2].bias[i]);
    }
}

// ============================================================
// Argmax helper
// ============================================================

int argmax(const float* data, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (data[i] > data[best]) best = i;
    }
    return best;
}

// ============================================================
// Percentile helper
// ============================================================

double percentile(const std::vector<double>& sorted, double p) {
    size_t idx = static_cast<size_t>(sorted.size() * p / 100.0);
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return sorted[idx];
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "=== Benchmark: float32 vs int8 ===\n\n";

    auto f32_layers = load_float32_weights("weights.bin");
    auto i8_layers = load_int8_weights("v2_int8/weights_int8.bin");

    Float32Context f32_ctx;
    Int8Context i8_ctx;

    // ── Warmup ────────────────────────────────────────────────
    Tensor warmup_input(1, 128);
    for (int i = 0; i < warmup_input.size(); i++) warmup_input.data[i] = 1.0f;
    for (int w = 0; w < 500; w++) {
        float32_forward_inplace(f32_layers, warmup_input, f32_ctx);
        int8_forward_inplace(i8_layers, warmup_input, i8_ctx);
    }

    // ══════════════════════════════════════════════════════════
    // PART A: Batch=100 timing (10,000 samples)
    // ══════════════════════════════════════════════════════════
    int batch = 100;
    constexpr int SAMPLES = 10000;
    std::vector<double> f32_latencies(SAMPLES);
    std::vector<double> i8_latencies(SAMPLES);
    volatile float sink = 0.0f;

    Tensor input(1, 128);
    for (int i = 0; i < input.size(); i++) input.data[i] = 1.0f;

    std::cout << "--- Batch=100 timing (" << SAMPLES << " samples) ---\n";
    std::cout << "Benchmarking float32 ...\n";
    for (int s = 0; s < SAMPLES; s++) {
        auto t0 = Clock::now();
        for (int b = 0; b < batch; b++)
            float32_forward_inplace(f32_layers, input, f32_ctx);
        auto t1 = Clock::now();
        sink += f32_ctx.act2.data[0];
        f32_latencies[s] = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count())
            / batch / 1000.0;
    }

    std::cout << "Benchmarking int8 ...\n";
    for (int s = 0; s < SAMPLES; s++) {
        auto t0 = Clock::now();
        for (int b = 0; b < batch; b++)
            int8_forward_inplace(i8_layers, input, i8_ctx);
        auto t1 = Clock::now();
        sink += i8_ctx.act2.data[0];
        i8_latencies[s] = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count())
            / batch / 1000.0;
    }

    auto f32_sorted = f32_latencies;
    auto i8_sorted  = i8_latencies;
    std::sort(f32_sorted.begin(), f32_sorted.end());
    std::sort(i8_sorted.begin(),  i8_sorted.end());

    double f32_p50 = percentile(f32_sorted, 50);
    double f32_p95 = percentile(f32_sorted, 95);
    double f32_p99 = percentile(f32_sorted, 99);
    double i8_p50  = percentile(i8_sorted, 50);
    double i8_p95  = percentile(i8_sorted, 95);
    double i8_p99  = percentile(i8_sorted, 99);

    // Write CSV
    std::ofstream csv("benchmarks/latency_results.csv");
    csv << "iteration,float32_us,int8_us\n";
    for (int i = 0; i < SAMPLES; i++) {
        csv << (i + 1) << ","
            << std::fixed << std::setprecision(4)
            << f32_latencies[i] << ","
            << i8_latencies[i] << "\n";
    }
    csv.close();
    std::cout << "✓ Wrote benchmarks/latency_results.csv\n\n";

    std::cout << "Batch=100 results (per-iteration latency):\n";
    std::cout << "  float32  p50: " << std::fixed << std::setprecision(3) << f32_p50
              << " us  p95: " << f32_p95 << " us  p99: " << f32_p99 << " us\n";
    std::cout << "  int8     p50: " << i8_p50
              << " us  p95: " << i8_p95 << " us  p99: " << i8_p99 << " us\n";
    std::cout << "  speedup (p50): " << std::setprecision(1) << f32_p50 / i8_p50 << "x\n";

    // ══════════════════════════════════════════════════════════
    // PART B: Batch=1 timing (1,000 samples) — true single-call
    // ══════════════════════════════════════════════════════════
    constexpr int SINGLE_SAMPLES = 1000;
    std::vector<double> f32_single(SINGLE_SAMPLES);
    std::vector<double> i8_single(SINGLE_SAMPLES);

    std::cout << "\n--- Batch=1 timing (" << SINGLE_SAMPLES << " samples, true single-call) ---\n";
    std::cout << "Benchmarking float32 ...\n";
    for (int s = 0; s < SINGLE_SAMPLES; s++) {
        auto t0 = Clock::now();
        float32_forward_inplace(f32_layers, input, f32_ctx);
        auto t1 = Clock::now();
        sink += f32_ctx.act2.data[0];
        f32_single[s] = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / 1000.0;
    }

    std::cout << "Benchmarking int8 ...\n";
    for (int s = 0; s < SINGLE_SAMPLES; s++) {
        auto t0 = Clock::now();
        int8_forward_inplace(i8_layers, input, i8_ctx);
        auto t1 = Clock::now();
        sink += i8_ctx.act2.data[0];
        i8_single[s] = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / 1000.0;
    }

    auto f32s_sorted = f32_single;
    auto i8s_sorted  = i8_single;
    std::sort(f32s_sorted.begin(), f32s_sorted.end());
    std::sort(i8s_sorted.begin(),  i8s_sorted.end());

    double f32s_p50 = percentile(f32s_sorted, 50);
    double f32s_p95 = percentile(f32s_sorted, 95);
    double f32s_p99 = percentile(f32s_sorted, 99);
    double i8s_p50  = percentile(i8s_sorted, 50);
    double i8s_p95  = percentile(i8s_sorted, 95);
    double i8s_p99  = percentile(i8s_sorted, 99);

    // Check for timer resolution issues: count distinct values
    int f32s_distinct = 1;
    for (size_t i = 1; i < f32s_sorted.size(); i++)
        if (f32s_sorted[i] != f32s_sorted[i - 1]) f32s_distinct++;
    int i8s_distinct = 1;
    for (size_t i = 1; i < i8s_sorted.size(); i++)
        if (i8s_sorted[i] != i8s_sorted[i - 1]) i8s_distinct++;

    std::cout << "\nBatch=1 results (true single-call latency):\n";
    std::cout << "  float32  p50: " << std::fixed << std::setprecision(3) << f32s_p50
              << " us  p95: " << f32s_p95 << " us  p99: " << f32s_p99 << " us\n";
    std::cout << "  int8     p50: " << i8s_p50
              << " us  p95: " << i8s_p95 << " us  p99: " << i8s_p99 << " us\n";
    std::cout << "  speedup (p50): " << std::setprecision(1) << f32s_p50 / i8s_p50 << "x\n";
    std::cout << "  Timer resolution note: " << SINGLE_SAMPLES << " samples produced "
              << f32s_distinct << " distinct float32 values and "
              << i8s_distinct << " distinct int8 values.\n";
    if (f32s_distinct < 20 || i8s_distinct < 20) {
        std::cout << "  ⚠ Low distinct count suggests timer resolution limits are affecting single-call measurements.\n";
        std::cout << "    Many samples land on the same tick; batch=100 numbers above are more reliable for latency comparison.\n";
    } else {
        std::cout << "  ✓ Sufficient distinct values — measurements appear above timer resolution.\n";
    }

    // ══════════════════════════════════════════════════════════
    // PART C: Multi-input accuracy + argmax validation
    // ══════════════════════════════════════════════════════════
    constexpr int NUM_INPUTS = 1000;
    constexpr unsigned INPUT_SEED = 12345;

    std::mt19937 rng(INPUT_SEED);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::cout << "\n--- Multi-input accuracy (" << NUM_INPUTS << " random inputs, seed=" << INPUT_SEED << ") ---\n";
    std::cout << "Input distribution: N(0, 1) normal\n";

    float global_max_abs_err = 0.0f;
    float global_sum_abs_err = 0.0f;
    int total_elements = 0;
    int argmax_matches = 0;

    for (int t = 0; t < NUM_INPUTS; t++) {
        // Generate random input
        Tensor test_input(1, 128);
        for (int i = 0; i < 128; i++)
            test_input.data[i] = dist(rng);

        // Float32 forward pass
        float32_forward_inplace(f32_layers, test_input, f32_ctx);

        // Int8 forward pass
        int8_forward_inplace(i8_layers, test_input, i8_ctx);

        // Compute errors
        float max_err = 0.0f;
        float sum_err = 0.0f;
        for (int i = 0; i < 10; i++) {
            float err = std::abs(f32_ctx.act2.data[i] - i8_ctx.act2.data[i]);
            max_err = std::max(max_err, err);
            sum_err += err;
        }
        global_max_abs_err = std::max(global_max_abs_err, max_err);
        global_sum_abs_err += sum_err;
        total_elements += 10;

        // Argmax comparison
        int f32_am = argmax(f32_ctx.act2.data, 10);
        int i8_am  = argmax(i8_ctx.act2.data, 10);
        bool match = (f32_am == i8_am);
        if (match) argmax_matches++;

        if (t < 5 || !match) {
            std::cout << "  Input " << std::setw(4) << (t + 1)
                      << ": max_err=" << std::scientific << std::setprecision(4) << max_err
                      << "  argmax: f32=" << f32_am << " i8=" << i8_am
                      << (match ? " ✓" : " ✗ MISMATCH")
                      << "\n";
            if (t == 4 && NUM_INPUTS > 5) {
                std::cout << "  ... (skipping intermediate progress prints for speed) ...\n";
            }
        }
    }

    float global_mean_abs_err = global_sum_abs_err / total_elements;

    std::cout << "\nAggregate accuracy across " << NUM_INPUTS << " inputs:\n";
    std::cout << "  Max abs error (across all inputs, all output elements): "
              << std::scientific << std::setprecision(4) << global_max_abs_err << "\n";
    std::cout << "  Mean abs error (across all inputs, all output elements): "
              << std::scientific << global_mean_abs_err << "\n";
    std::cout << "  Argmax match rate: " << argmax_matches << "/" << NUM_INPUTS
              << " (" << std::fixed << std::setprecision(0)
              << (100.0 * argmax_matches / NUM_INPUTS) << "%)\n";

    (void)sink;
    return 0;
}
