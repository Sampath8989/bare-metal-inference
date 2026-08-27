#pragma once
// common.hpp — Shared SIMD types and functions for V3 (Scaled).

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

// ─── Tensor (float32, 64-byte aligned) ──────────────────────────────
struct Tensor {
    float* data;
    int rows, cols;

    Tensor() : data(nullptr), rows(0), cols(0) {}

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
    void zero() { std::memset(data, 0, static_cast<size_t>(rows) * cols * sizeof(float)); }
};

// ─── Int8Tensor (int8, 64-byte aligned) ─────────────────────────────
struct Int8Tensor {
    int8_t* data;
    int rows, cols;
    float scale;

    Int8Tensor() : data(nullptr), rows(0), cols(0), scale(1.0f) {}

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

    int size() const { return rows * cols; }
};

// ─── Quantization ────────────────────────────────────────────────────
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

// ─── Int8 matmul: AVX2 SIMD (32 int8 per iteration) ────────────────
inline void matmul_int8_simd_inplace(const Int8Tensor& A, const Int8Tensor& B,
                                     Tensor& C, std::vector<int32_t>& acc) {
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

                __m256i p32_0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(p16_lo));
                __m256i p32_1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p16_lo, 1));
                __m256i p32_2 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(p16_hi));
                __m256i p32_3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p16_hi, 1));

                __m256i acc0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + i*n + j));
                __m256i acc1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + i*n + j + 8));
                __m256i acc2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + i*n + j + 16));
                __m256i acc3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc.data() + i*n + j + 24));

                acc0 = _mm256_add_epi32(acc0, p32_0);
                acc1 = _mm256_add_epi32(acc1, p32_1);
                acc2 = _mm256_add_epi32(acc2, p32_2);
                acc3 = _mm256_add_epi32(acc3, p32_3);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + i*n + j),     acc0);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + i*n + j + 8),  acc1);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + i*n + j + 16), acc2);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc.data() + i*n + j + 24), acc3);
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

// ─── Int8Layer ───────────────────────────────────────────────────────
struct Int8Layer {
    Int8Tensor weights;
    std::vector<float> bias;
    int in_size, out_size;

    Int8Layer() : in_size(0), out_size(0) {}
    Int8Layer(int in_sz, int out_sz, float sc)
        : weights(in_sz, out_sz, sc),
          bias(static_cast<size_t>(out_sz), 0.0f),
          in_size(in_sz), out_size(out_sz) {
        std::mt19937 gen(42 + in_sz * 1000 + out_sz);
        std::uniform_int_distribution<int> dis(-64, 64);
        for (int i = 0; i < weights.size(); i++) {
            weights.data[i] = static_cast<int8_t>(dis(gen));
        }
    }
    Int8Layer(Int8Layer&&) = default;
};

// ─── Int8Context: pre-allocated buffers for 3-layer inference ─────────
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

    Int8Context(int in_dim, int h1_dim, int h2_dim, int out_dim)
        : q_in(1, in_dim),
          act0(1, h1_dim), q_act0(1, h1_dim),
          act1(1, h2_dim), q_act1(1, h2_dim),
          act2(1, out_dim),
          acc0(h1_dim, 0), acc1(h2_dim, 0), acc2(out_dim, 0) {}
};

// ─── AVX2 SIMD INT8 forward pass ─────────────────────────────────────
inline void forward_simd_inplace(const std::vector<Int8Layer>& layers,
                                 const Tensor& input, Int8Context& ctx) {
    quantize_int8_inplace(input, ctx.q_in);
    matmul_int8_simd_inplace(ctx.q_in, layers[0].weights, ctx.act0, ctx.acc0);
    for (int i = 0; i < ctx.act0.cols; i++)
        ctx.act0.data[i] = std::max(0.0f, ctx.act0.data[i] + layers[0].bias[i]);

    quantize_int8_inplace(ctx.act0, ctx.q_act0);
    matmul_int8_simd_inplace(ctx.q_act0, layers[1].weights, ctx.act1, ctx.acc1);
    for (int i = 0; i < ctx.act1.cols; i++)
        ctx.act1.data[i] = std::max(0.0f, ctx.act1.data[i] + layers[1].bias[i]);

    quantize_int8_inplace(ctx.act1, ctx.q_act1);
    matmul_int8_simd_inplace(ctx.q_act1, layers[2].weights, ctx.act2, ctx.acc2);
    for (int i = 0; i < ctx.act2.cols; i++)
        ctx.act2.data[i] = std::max(0.0f, ctx.act2.data[i] + layers[2].bias[i]);
}
