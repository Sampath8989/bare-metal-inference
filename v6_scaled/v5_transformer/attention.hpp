#pragma once

// Single-head self-attention + benchmark.
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "tensor.hpp"

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

inline void softmax_stable(float* row, int length) {
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

inline Tensor attention_forward(const Tensor& input, const AttentionHead& head) {
    int seq_len = input.rows;
    int d_model = head.d_model;
    int d_head = head.d_head;

    Tensor Q = matmul(input, head.Wq);
    Tensor K = matmul(input, head.Wk);
    Tensor V = matmul(input, head.Wv);

    // Manual Kᵀ: matmul() expects row-major.
    Tensor K_T(d_head, seq_len);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_head; j++)
            K_T(j, i) = K(i, j);

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

inline void init_attention_weights(AttentionHead& head) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    float scale_qkv = std::sqrt(2.0f / head.d_model);
    float scale_o = std::sqrt(2.0f / head.d_head);

    for (int i = 0; i < head.Wq.size(); i++) head.Wq.data[i] = dis(gen) * scale_qkv;
    for (int i = 0; i < head.Wk.size(); i++) head.Wk.data[i] = dis(gen) * scale_qkv;
    for (int i = 0; i < head.Wv.size(); i++) head.Wv.data[i] = dis(gen) * scale_qkv;
    for (int i = 0; i < head.Wo.size(); i++) head.Wo.data[i] = dis(gen) * scale_o;
}
