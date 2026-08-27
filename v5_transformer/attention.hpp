#pragma once

// Single-head self-attention + benchmark.
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "../src/tensor.hpp"

Tensor matmul(const Tensor& A, const Tensor& B);

struct AttentionHead {
    Tensor Wq;   // d_model × d_head
    Tensor Wk;   // d_model × d_head
    Tensor Wv;   // d_model × d_head
    Tensor Wo;   // d_head × d_model
    int d_model;
    int d_head;

    AttentionHead(int d_model, int d_head)
        : Wq(d_model, d_head), Wk(d_model, d_head),
          Wv(d_model, d_head), Wo(d_head, d_model),
          d_model(d_model), d_head(d_head) {}
};

void softmax_stable(float* row, int length) {
    // Subtract row max before exp() to prevent overflow.
    float max_val = *std::max_element(row, row + length);
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        row[i] = std::exp(row[i] - max_val);
        sum += row[i];
    }
    for (int i = 0; i < length; i++) {
        row[i] /= sum;
    }
}

Tensor attention_forward(const Tensor& input, const AttentionHead& head) {
    int seq_len = input.rows;
    int d_model = head.d_model;
    int d_head = head.d_head;

    Tensor Q = matmul(input, head.Wq);
    Tensor K = matmul(input, head.Wk);
    Tensor V = matmul(input, head.Wv);

    // Manual Kᵀ: matmul() expects row-major.
    Tensor K_T(d_head, seq_len);
    for (int i = 0; i < seq_len; i++)
        #pragma unroll
        for (int j = 0; j < d_head; j++)
            K_T(j, i) = K(i, j);

    // d_head is compile-time 64 → unrollable; seq_len is runtime.
    Tensor scores = matmul(Q, K_T);
    float scale = 1.0f / std::sqrt(static_cast<float>(d_head));
    for (int i = 0; i < scores.size(); i++)
        scores.data[i] *= scale;

    for (int i = 0; i < seq_len; i++) {
        softmax_stable(&scores.data[i * seq_len], seq_len);
    }

    Tensor output = matmul(scores, V);

    Tensor final_output = matmul(output, head.Wo);

    return final_output;
}

void init_attention_weights(AttentionHead& head) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    float scale_qkv = std::sqrt(2.0f / head.d_model);
    float scale_o = std::sqrt(2.0f / head.d_head);

    for (int i = 0; i < head.Wq.size(); i++) head.Wq.data[i] = dis(gen) * scale_qkv;
    for (int i = 0; i < head.Wk.size(); i++) head.Wk.data[i] = dis(gen) * scale_qkv;
    for (int i = 0; i < head.Wv.size(); i++) head.Wv.data[i] = dis(gen) * scale_qkv;
    for (int i = 0; i < head.Wo.size(); i++) head.Wo.data[i] = dis(gen) * scale_o;
}

void benchmark_attention() {
    const int d_model = 128;
    const int d_head = 64;
    const int seq_len = 1;

    AttentionHead head(d_model, d_head);
    init_attention_weights(head);

    Tensor input(seq_len, d_model);
    std::mt19937 gen(12345);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < input.size(); i++) input.data[i] = dis(gen);

    for (int w = 0; w < 200; w++) {
        Tensor out = attention_forward(input, head);
        (void)out(0, 0);
    }

    constexpr int RUNS = 1000;
    std::vector<double> times;
    times.reserve(RUNS);
    volatile float sink = 0.0f;

    for (int r = 0; r < RUNS; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        Tensor out = attention_forward(input, head);
        sink += out(0, 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
        times.push_back(us);
    }

    std::sort(times.begin(), times.end());
    double p50 = times[RUNS * 50 / 100];
    double p95 = times[RUNS * 95 / 100];
    double p99 = times[RUNS * 99 / 100];

    std::cout << "\n=== ATTENTION BENCHMARK (d_model=" << d_model
              << ", d_head=" << d_head << ", seq_len=" << seq_len << ")" << std::endl;
    std::cout << "p50=" << std::fixed << std::setprecision(3) << p50
              << " µs  p95=" << p95 << "  p99=" << p99 << std::endl;

    (void)sink;
}
