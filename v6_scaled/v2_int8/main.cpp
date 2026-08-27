// Scaled Int8 inference engine: quantized MLP forward pass.
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
#include <limits>
#include <numeric>
#include <cstdlib>
#include <random>

using Clock = std::chrono::high_resolution_clock;

struct Tensor {
    float* data;
    int rows, cols;

    Tensor(int r, int c) : rows(r), cols(c) {
        size_t n = static_cast<size_t>(r) * c;
        size_t aligned = (n * sizeof(float) + 63) & ~63ULL;
        data = static_cast<float*>(std::aligned_alloc(64, aligned));
        if (!data) throw std::bad_alloc();
        std::memset(data, 0, n * sizeof(float));
    }
    Tensor(const Tensor& o) : rows(o.rows), cols(o.cols) {
        size_t n = static_cast<size_t>(rows) * cols;
        size_t aligned = (n * sizeof(float) + 63) & ~63ULL;
        data = static_cast<float*>(std::aligned_alloc(64, aligned));
        if (!data) throw std::bad_alloc();
        std::memcpy(data, o.data, n * sizeof(float));
    }
    Tensor& operator=(const Tensor& o) {
        if (this != &o) {
            std::free(data);
            rows = o.rows; cols = o.cols;
            size_t n = static_cast<size_t>(rows) * cols;
            size_t aligned = (n * sizeof(float) + 63) & ~63ULL;
            data = static_cast<float*>(std::aligned_alloc(64, aligned));
            if (!data) throw std::bad_alloc();
            std::memcpy(data, o.data, n * sizeof(float));
        }
        return *this;
    }
    Tensor(Tensor&& o) noexcept : data(o.data), rows(o.rows), cols(o.cols) {
        o.data = nullptr;
        o.rows = 0; o.cols = 0;
    }
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            std::free(data);
            data = o.data;
            rows = o.rows;
            cols = o.cols;
            o.data = nullptr;
            o.rows = 0; o.cols = 0;
        }
        return *this;
    }
    ~Tensor() { if (data) std::free(data); }

    int size() const { return rows * cols; }
    float& operator()(int r, int c) { return data[r * cols + c]; }
    const float& operator()(int r, int c) const { return data[r * cols + c]; }
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

    int size() const { return rows * cols; }
};

inline float relu(float x) { return std::max(0.0f, x); }

Int8Tensor quantize(const Tensor& input, float& out_scale) {
    float max_abs = 0.0f;
    for (int i = 0; i < input.size(); i++)
        max_abs = std::max(max_abs, std::abs(input.data[i]));

    out_scale = (max_abs == 0.0f) ? 1.0f : max_abs / 127.0f;

    Int8Tensor out(input.rows, input.cols, out_scale);
    for (int i = 0; i < input.size(); i++) {
        float scaled = input.data[i] / out_scale;
        out.data[i] = static_cast<int8_t>(std::clamp(std::round(scaled), -128.0f, 127.0f));
    }
    return out;
}

Tensor matmul_int8(const Int8Tensor& A, const Int8Tensor& B) {
    int m = A.rows, n = B.cols, k = A.cols;
    float combined_scale = A.scale * B.scale;

    std::vector<int32_t> acc(static_cast<size_t>(m) * n, 0);

    for (int i = 0; i < m; i++) {
        for (int q = 0; q < k; q++) {
            int8_t a_val = A.data[i * k + q];
            for (int j = 0; j < n; j++) {
                acc[i * n + j] += static_cast<int32_t>(a_val)
                                * static_cast<int32_t>(B.data[q * n + j]);
            }
        }
    }

    Tensor C(m, n);
    for (int i = 0; i < m * n; i++)
        C.data[i] = static_cast<float>(acc[i]) * combined_scale;

    return C;
}

struct Int8Layer {
    Int8Tensor weights;
    std::vector<float> bias;
    int in_size, out_size;

    Int8Layer(int in_sz, int out_sz, float scale)
        : weights(in_sz, out_sz, scale), bias(static_cast<size_t>(out_sz), 0.0f),
          in_size(in_sz), out_size(out_sz) {
        std::mt19937 gen(42 + in_sz * 1000 + out_sz);
        std::uniform_int_distribution<int> dis(-64, 64);
        for (int i = 0; i < weights.size(); i++) {
            weights.data[i] = static_cast<int8_t>(dis(gen));
        }
    }

    Int8Layer(Int8Layer&&) = default;
};

Tensor forward(const std::vector<Int8Layer>& layers, const Tensor& input) {
    Tensor current = input;

    for (size_t l = 0; l < layers.size(); l++) {
        float input_scale;
        Int8Tensor input_q = quantize(current, input_scale);

        Tensor output = matmul_int8(input_q, layers[l].weights);

        int m = output.rows;
        int out_size = layers[l].out_size;
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < out_size; col++) {   // add float32 bias
                output.data[row * out_size + col] += layers[l].bias[col];
            }
        }

        for (int i = 0; i < output.size(); i++)
            output.data[i] = relu(output.data[i]);

        current = std::move(output);
    }

    return current;
}

int main(int argc, char* argv[]) {
    int scale = 512;
    if (argc >= 2) {
        int s = std::atoi(argv[1]);
        if (s > 0) scale = s;
    }

    int in_dim = scale;
    int h1_dim = scale;
    int h2_dim = scale;
    int out_dim = scale;

    std::cout << "=== V2 Int8 Inference Engine (Scaled: " << scale << "x" << scale << ") ===\n";
    std::cout << "Architecture: " << in_dim << " -> " << h1_dim << " -> " << h2_dim << " -> " << out_dim << "\n";

    std::vector<Int8Layer> layers;
    layers.emplace_back(in_dim, h1_dim, 0.001f);
    layers.emplace_back(h1_dim, h2_dim, 0.001f);
    layers.emplace_back(h2_dim, out_dim, 0.001f);

    Tensor input(1, in_dim);
    for (int i = 0; i < input.size(); i++)
        input.data[i] = 1.0f;

    for (int w = 0; w < 200; w++) {
        Tensor out = forward(layers, input);
        (void)out(0, 0);
    }

    constexpr int RUNS = 1000;
    std::vector<double> times;
    times.reserve(RUNS);

    volatile float sink = 0.0f;
    Tensor final_out(1, out_dim);

    for (int r = 0; r < RUNS; r++) {
        auto t0 = Clock::now();
        final_out = forward(layers, input);
        sink += final_out(0, 0);
        auto t1 = Clock::now();
        double elapsed_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
        times.push_back(elapsed_us);
    }

    std::sort(times.begin(), times.end());
    double p50 = times[RUNS * 50 / 100];
    double p95 = times[RUNS * 95 / 100];
    double p99 = times[RUNS * 99 / 100];

    std::cout << "Size: " << scale << "\n";
    std::cout << "p50: " << std::fixed << std::setprecision(3) << p50 << " us\n";
    std::cout << "p95: " << p95 << " us\n";
    std::cout << "p99: " << p99 << " us\n";

    std::cout << "Output sample (first 5): [";
    for (int i = 0; i < std::min(5, final_out.cols); i++) {
        std::cout << std::fixed << std::setprecision(4) << final_out(0, i);
        if (i < std::min(5, final_out.cols) - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";

    (void)sink;
    return 0;
}
