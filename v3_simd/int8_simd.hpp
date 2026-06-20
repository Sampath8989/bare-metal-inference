/*
 * v3_simd/int8_simd.hpp
 *
 * Shared header for AVX2-accelerated int8 inference.
 * Contains both scalar and SIMD matmul implementations for comparison.
 */

#pragma once

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
#include <immintrin.h>

// ============================================================
// Minimal float32 Tensor
// ============================================================

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
        o.data = nullptr; o.rows = 0; o.cols = 0;
    }
    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            std::free(data);
            data = o.data; rows = o.rows; cols = o.cols;
            o.data = nullptr; o.rows = 0; o.cols = 0;
        }
        return *this;
    }
    ~Tensor() { if (data) std::free(data); }

    int size() const { return rows * cols; }
    float& operator()(int r, int c) { return data[r * cols + c]; }
    const float& operator()(int r, int c) const { return data[r * cols + c]; }
};

// ============================================================
// Int8 Tensor (64-byte aligned)
// ============================================================

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

inline float relu(float x) { return std::max(0.0f, x); }

// ============================================================
// Pre-allocated Context for Inplace Execution
// ============================================================

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
// Quantize float32 → int8
// ============================================================

inline Int8Tensor quantize_int8(const Tensor& input, float& out_scale) {
    float max_abs = 0.0f;
    for (int i = 0; i < input.size(); i++)
        max_abs = std::max(max_abs, std::abs(input.data[i]));
    out_scale = (max_abs == 0.0f) ? 1.0f : max_abs / 127.0f;

    Int8Tensor out(input.rows, input.cols, out_scale);
    for (int i = 0; i < input.size(); i++) {
        float s = input.data[i] / out_scale;
        out.data[i] = static_cast<int8_t>(std::clamp(std::round(s), -128.0f, 127.0f));
    }
    return out;
}

inline void quantize_int8_inplace(const Tensor& input, Int8Tensor& out) {
    float max_abs = 0.0f;
    for (int i = 0; i < input.size(); i++)
        max_abs = std::max(max_abs, std::abs(input.data[i]));
    out.scale = (max_abs == 0.0f) ? 1.0f : max_abs / 127.0f;
    for (int i = 0; i < input.size(); i++) {
        float s = input.data[i] / out.scale;
        out.data[i] = static_cast<int8_t>(std::clamp(std::round(s), -128.0f, 127.0f));
    }
}

// ============================================================
// Scalar matmul
// ============================================================

inline Tensor matmul_int8_scalar(const Int8Tensor& A, const Int8Tensor& B) {
    int m = A.rows, n = B.cols, k = A.cols;
    float cs = A.scale * B.scale;

    std::vector<int32_t> acc(static_cast<size_t>(m) * n, 0);

    for (int i = 0; i < m; i++) {
        for (int q = 0; q < k; q++) {
            int8_t a = A.data[i * k + q];
            for (int j = 0; j < n; j++) {
                acc[i * n + j] += static_cast<int32_t>(a)
                                * static_cast<int32_t>(B.data[q * n + j]);
            }
        }
    }

    Tensor C(m, n);
    for (int i = 0; i < m * n; i++)
        C.data[i] = static_cast<float>(acc[i]) * cs;
    return C;
}

inline void matmul_int8_scalar_inplace(const Int8Tensor& A, const Int8Tensor& B, Tensor& C, std::vector<int32_t>& acc) {
    int m = A.rows, n = B.cols, k = A.cols;
    float cs = A.scale * B.scale;
    std::fill(acc.begin(), acc.end(), 0);

    for (int i = 0; i < m; i++) {
        for (int q = 0; q < k; q++) {
            int8_t a = A.data[i * k + q];
            for (int j = 0; j < n; j++) {
                acc[i * n + j] += static_cast<int32_t>(a)
                                * static_cast<int32_t>(B.data[q * n + j]);
            }
        }
    }

    for (int i = 0; i < m * n; i++)
        C.data[i] = static_cast<float>(acc[i]) * cs;
}

// ============================================================
// AVX2 SIMD int8 matmul (Weight offsets corrected)
// ============================================================

inline Tensor matmul_int8_simd(const Int8Tensor& A, const Int8Tensor& B) {
    int m = A.rows, n = B.cols, k = A.cols;
    float cs = A.scale * B.scale;

    std::vector<int32_t> acc(static_cast<size_t>(m) * n, 0);

    for (int i = 0; i < m; i++) {
        for (int q = 0; q < k; q++) {
            int8_t a_val = A.data[i * k + q];
            __m256i a16 = _mm256_set1_epi16(static_cast<int16_t>(a_val));

            int j = 0;
            for (; j + 32 <= n; j += 32) {
                // Load 32 int8 from B row q (offset calculation corrected)
                __m256i b8 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(B.data + q * n + j));

                __m128i b8_lo = _mm256_castsi256_si128(b8);
                __m256i b16_lo = _mm256_cvtepi8_epi16(b8_lo);

                __m128i b8_hi = _mm256_extracti128_si256(b8, 1);
                __m256i b16_hi = _mm256_cvtepi8_epi16(b8_hi);

                __m256i p16_lo = _mm256_mullo_epi16(a16, b16_lo);
                __m256i p16_hi = _mm256_mullo_epi16(a16, b16_hi);

                __m128i p16_lo_lo = _mm256_castsi256_si128(p16_lo);
                __m128i p16_lo_hi = _mm256_extracti128_si256(p16_lo, 1);
                __m128i p16_hi_lo = _mm256_castsi256_si128(p16_hi);
                __m128i p16_hi_hi = _mm256_extracti128_si256(p16_hi, 1);

                __m256i p32_0 = _mm256_cvtepi16_epi32(p16_lo_lo);
                __m256i p32_1 = _mm256_cvtepi16_epi32(p16_lo_hi);
                __m256i p32_2 = _mm256_cvtepi16_epi32(p16_hi_lo);
                __m256i p32_3 = _mm256_cvtepi16_epi32(p16_hi_hi);

                __m256i acc0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(acc.data() + i * n + j));
                __m256i acc1 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(acc.data() + i * n + j + 8));
                __m256i acc2 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(acc.data() + i * n + j + 16));
                __m256i acc3 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(acc.data() + i * n + j + 24));

                acc0 = _mm256_add_epi32(acc0, p32_0);
                acc1 = _mm256_add_epi32(acc1, p32_1);
                acc2 = _mm256_add_epi32(acc2, p32_2);
                acc3 = _mm256_add_epi32(acc3, p32_3);

                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(acc.data() + i * n + j), acc0);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(acc.data() + i * n + j + 8), acc1);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(acc.data() + i * n + j + 16), acc2);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(acc.data() + i * n + j + 24), acc3);
            }

            // Scalar tail fallback (offset calculation corrected)
            for (; j < n; j++) {
                acc[i * n + j] += static_cast<int32_t>(a_val)
                                * static_cast<int32_t>(B.data[q * n + j]);
            }
        }
    }

    Tensor C(m, n);
    for (int i = 0; i < m * n; i++)
        C.data[i] = static_cast<float>(acc[i]) * cs;
    return C;
}

inline void matmul_int8_simd_inplace(const Int8Tensor& A, const Int8Tensor& B, Tensor& C, std::vector<int32_t>& acc) {
    int m = A.rows, n = B.cols, k = A.cols;
    float cs = A.scale * B.scale;
    std::fill(acc.begin(), acc.end(), 0);

    for (int i = 0; i < m; i++) {
        for (int q = 0; q < k; q++) {
            int8_t a_val = A.data[i * k + q];
            __m256i a16 = _mm256_set1_epi16(static_cast<int16_t>(a_val));

            int j = 0;
            for (; j + 32 <= n; j += 32) {
                __m256i b8 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(B.data + q * n + j));

                __m128i b8_lo = _mm256_castsi256_si128(b8);
                __m256i b16_lo = _mm256_cvtepi8_epi16(b8_lo);

                __m128i b8_hi = _mm256_extracti128_si256(b8, 1);
                __m256i b16_hi = _mm256_cvtepi8_epi16(b8_hi);

                __m256i p16_lo = _mm256_mullo_epi16(a16, b16_lo);
                __m256i p16_hi = _mm256_mullo_epi16(a16, b16_hi);

                __m128i p16_lo_lo = _mm256_castsi256_si128(p16_lo);
                __m128i p16_lo_hi = _mm256_extracti128_si256(p16_lo, 1);
                __m128i p16_hi_lo = _mm256_castsi256_si128(p16_hi);
                __m128i p16_hi_hi = _mm256_extracti128_si256(p16_hi, 1);

                __m256i p32_0 = _mm256_cvtepi16_epi32(p16_lo_lo);
                __m256i p32_1 = _mm256_cvtepi16_epi32(p16_lo_hi);
                __m256i p32_2 = _mm256_cvtepi16_epi32(p16_hi_lo);
                __m256i p32_3 = _mm256_cvtepi16_epi32(p16_hi_hi);

                __m256i acc0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(acc.data() + i * n + j));
                __m256i acc1 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(acc.data() + i * n + j + 8));
                __m256i acc2 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(acc.data() + i * n + j + 16));
                __m256i acc3 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(acc.data() + i * n + j + 24));

                acc0 = _mm256_add_epi32(acc0, p32_0);
                acc1 = _mm256_add_epi32(acc1, p32_1);
                acc2 = _mm256_add_epi32(acc2, p32_2);
                acc3 = _mm256_add_epi32(acc3, p32_3);

                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(acc.data() + i * n + j), acc0);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(acc.data() + i * n + j + 8), acc1);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(acc.data() + i * n + j + 16), acc2);
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(acc.data() + i * n + j + 24), acc3);
            }

            for (; j < n; j++) {
                acc[i * n + j] += static_cast<int32_t>(a_val)
                                * static_cast<int32_t>(B.data[q * n + j]);
            }
        }
    }

    for (int i = 0; i < m * n; i++)
        C.data[i] = static_cast<float>(acc[i]) * cs;
}

// ============================================================
// Load int8 weights
// ============================================================

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

inline std::vector<Int8Layer> load_int8_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open weights file: " + path);

    std::vector<Int8Layer> layers;
    int idx = 0;

    while (file.peek() != EOF) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        if (file.gcount() == 0) break;
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        float scale;
        file.read(reinterpret_cast<char*>(&scale), sizeof(scale));

        Int8Layer layer(rows, cols, scale);
        size_t num_weights = static_cast<size_t>(rows) * cols;
        file.read(reinterpret_cast<char*>(layer.weights.data), num_weights);

        int bias_size;
        file.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));
        file.read(reinterpret_cast<char*>(layer.bias.data()),
                  static_cast<size_t>(bias_size) * sizeof(float));

        layers.push_back(std::move(layer));
        idx++;
    }
    return layers;
}

// ============================================================
// Forward passes
// ============================================================

inline Tensor forward_scalar(const std::vector<Int8Layer>& layers,
                             const Tensor& input) {
    Tensor current = input;
    for (int l = 0; l < static_cast<int>(layers.size()); l++) {
        float input_scale;
        Int8Tensor input_q = quantize_int8(current, input_scale);
        Tensor output = matmul_int8_scalar(input_q, layers[l].weights);

        int m = output.rows, out_size = layers[l].out_size;
        for (int row = 0; row < m; row++)
            for (int col = 0; col < out_size; col++)
                output.data[row * out_size + col] += layers[l].bias[col];

        for (int i = 0; i < output.size(); i++)
            output.data[i] = relu(output.data[i]);

        current = std::move(output);
    }
    return current;
}

inline Tensor forward_simd(const std::vector<Int8Layer>& layers,
                           const Tensor& input) {
    Tensor current = input;
    for (int l = 0; l < static_cast<int>(layers.size()); l++) {
        float input_scale;
        Int8Tensor input_q = quantize_int8(current, input_scale);
        Tensor output = matmul_int8_simd(input_q, layers[l].weights);

        int m = output.rows, out_size = layers[l].out_size;
        for (int row = 0; row < m; row++)
            for (int col = 0; col < out_size; col++)
                output.data[row * out_size + col] += layers[l].bias[col];

        for (int i = 0; i < output.size(); i++)
            output.data[i] = relu(output.data[i]);

        current = std::move(output);
    }
    return current;
}

inline void forward_scalar_inplace(const std::vector<Int8Layer>& layers, const Tensor& input, Int8Context& ctx) {
    quantize_int8_inplace(input, ctx.q_in);
    matmul_int8_scalar_inplace(ctx.q_in, layers[0].weights, ctx.act0, ctx.acc0);
    for (int i = 0; i < 64; i++) ctx.act0.data[i] = relu(ctx.act0.data[i] + layers[0].bias[i]);

    quantize_int8_inplace(ctx.act0, ctx.q_act0);
    matmul_int8_scalar_inplace(ctx.q_act0, layers[1].weights, ctx.act1, ctx.acc1);
    for (int i = 0; i < 32; i++) ctx.act1.data[i] = relu(ctx.act1.data[i] + layers[1].bias[i]);

    quantize_int8_inplace(ctx.act1, ctx.q_act1);
    matmul_int8_scalar_inplace(ctx.q_act1, layers[2].weights, ctx.act2, ctx.acc2);
    for (int i = 0; i < 10; i++) ctx.act2.data[i] = relu(ctx.act2.data[i] + layers[2].bias[i]);
}

inline void forward_simd_inplace(const std::vector<Int8Layer>& layers, const Tensor& input, Int8Context& ctx) {
    quantize_int8_inplace(input, ctx.q_in);
    matmul_int8_simd_inplace(ctx.q_in, layers[0].weights, ctx.act0, ctx.acc0);
    for (int i = 0; i < 64; i++) ctx.act0.data[i] = relu(ctx.act0.data[i] + layers[0].bias[i]);

    quantize_int8_inplace(ctx.act0, ctx.q_act0);
    matmul_int8_simd_inplace(ctx.q_act0, layers[1].weights, ctx.act1, ctx.acc1);
    for (int i = 0; i < 32; i++) ctx.act1.data[i] = relu(ctx.act1.data[i] + layers[1].bias[i]);

    quantize_int8_inplace(ctx.act1, ctx.q_act1);
    matmul_int8_simd_inplace(ctx.q_act1, layers[2].weights, ctx.act2, ctx.acc2);
    for (int i = 0; i < 10; i++) ctx.act2.data[i] = relu(ctx.act2.data[i] + layers[2].bias[i]);
}

// ============================================================
// Helper: argmax over output vector
// ============================================================

inline int argmax(const float* data, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (data[i] > data[best]) best = i;
    return best;
}
