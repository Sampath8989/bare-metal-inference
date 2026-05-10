#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <stdexcept>

struct Tensor {
    std::vector<float> data;
    int rows, cols;

    Tensor(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {
        if (r <= 0 || c <= 0)
            throw std::invalid_argument("Tensor dimensions must be positive");
    }

    float& operator()(int r, int c) {
        return data[r * cols + c];
    }

    const float& operator()(int r, int c) const {
        return data[r * cols + c];
    }

    int size() const { return rows * cols; }
};

Tensor matmul(const Tensor& A, const Tensor& B);

#endif
