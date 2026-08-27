#pragma once
#include <vector>
#include <cassert>
#include <algorithm>
#include <numeric>
#include "tensor.hpp"
#include "relu.hpp"

struct Layer {
    Tensor weights;
    Tensor bias;

    int in_size;
    int out_size;

    Layer(int in_sz, int out_sz) :
        weights(in_sz, out_sz),
        bias(1, out_sz),
        in_size(in_sz),
        out_size(out_sz)
    {
        for (int i = 0; i < weights.size(); i++) {
            weights.data[i] = 0.01f;
        }
        for (int i = 0; i < bias.size(); i++) {
            bias.data[i] = 0.0f;
        }
    }

    Tensor forward(const Tensor& x, bool apply_relu = true) const {
        if (x.cols != in_size) {
            throw std::invalid_argument(
                "Layer::forward size mismatch: expected cols="
                + std::to_string(in_size)
                + ", got cols="
                + std::to_string(x.cols)
            );
        }

        Tensor out = matmul(x, weights);

        for (int i = 0; i < out_size; i++) {
            out(0, i) += bias(0, i);
        }

        if (apply_relu) {
            for (int i = 0; i < out.size(); i++) {
                out.data[i] = relu(out.data[i]);
            }
        }
        return out;
    }
};
