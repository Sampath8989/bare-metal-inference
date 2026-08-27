#pragma once

// Transformer block: attention → residual + LayerNorm → MLP.
#include "attention.hpp"
#include "../src/Layer.hpp"
#include "../src/relu.hpp"
#include <cmath>
#include <numeric>

void layer_norm_inplace(Tensor& t) {
    for (int row = 0; row < t.rows; row++) {
        float* data = &t.data[row * t.cols];
        int n = t.cols;

        float mean = 0.0f;
        for (int i = 0; i < n; i++) mean += data[i];
        mean /= n;

        float var = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = data[i] - mean;
            var += diff * diff;
        }
        var /= n;

        float inv_std = 1.0f / std::sqrt(var + 1e-5f);   // no learned gamma/beta
        for (int i = 0; i < n; i++) {
            data[i] = (data[i] - mean) * inv_std;
        }
    }
}

struct TransformerBlock {
    AttentionHead attn;
    Layer mlp1;   // d_model → mlp_hidden
    Layer mlp2;   // mlp_hidden → d_model
    int d_model;

    TransformerBlock(int d_model, int d_head, int mlp_hidden)
        : attn(d_model, d_head),

          mlp1(d_model, mlp_hidden),
          mlp2(mlp_hidden, d_model),
          d_model(d_model) {}

    Tensor forward(const Tensor& input) {

        Tensor attn_out = attention_forward(input, attn);

        Tensor residual(input.rows, input.cols);
        for (int i = 0; i < residual.size(); i++) {
            residual.data[i] = input.data[i] + attn_out.data[i];
        }
        layer_norm_inplace(residual);

        Tensor hidden = mlp1.forward(residual, true);
        Tensor mlp_out = mlp2.forward(hidden, true);

        return mlp_out;
    }
};

void init_transformer_block(TransformerBlock& block) {
    init_attention_weights(block.attn);

    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    float scale1 = std::sqrt(2.0f / block.mlp1.in_size);
    for (int i = 0; i < block.mlp1.weights.size(); i++)
        block.mlp1.weights.data[i] = dis(gen) * scale1;
    float scale2 = std::sqrt(2.0f / block.mlp2.in_size);
    for (int i = 0; i < block.mlp2.weights.size(); i++)
        block.mlp2.weights.data[i] = dis(gen) * scale2;
}
