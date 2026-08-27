#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

struct Tensor {
    float* data;
    int rows;
    int cols;

    Tensor(int r, int c) : rows(r), cols(c) {
        if (r <= 0 || c <= 0) {
            throw std::invalid_argument("Tensor dimensions must be positive");
        }

        size_t bytes = (size_t)r * c * sizeof(float);
        size_t aligned_bytes = (bytes + 63) & ~63ULL;
        data = static_cast<float*>(std::aligned_alloc(64, aligned_bytes));
        if (!data) throw std::bad_alloc();
        std::memset(data, 0, bytes);
    }

    Tensor(const Tensor& other) : rows(other.rows), cols(other.cols) {
        size_t bytes = (size_t)rows * cols * sizeof(float);
        size_t aligned_bytes = (bytes + 63) & ~63ULL;
        data = static_cast<float*>(std::aligned_alloc(64, aligned_bytes));
        if (!data) throw std::bad_alloc();
        std::memcpy(data, other.data, bytes);
    }

    Tensor& operator=(const Tensor& other) {
        if (this == &other) return *this;
        std::free(data);
        rows = other.rows; cols = other.cols;
        size_t bytes = (size_t)rows * cols * sizeof(float);
        size_t aligned_bytes = (bytes + 63) & ~63ULL;
        data = static_cast<float*>(std::aligned_alloc(64, aligned_bytes));
        if (!data) throw std::bad_alloc();
        std::memcpy(data, other.data, bytes);
        return *this;
    }

    Tensor(Tensor&& other) noexcept 
        : data(other.data), rows(other.rows), cols(other.cols) {
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) return *this;
        std::free(data);
        data = other.data;
        rows = other.rows;
        cols = other.cols;
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
        return *this;
    }

    ~Tensor() {
        std::free(data);
        data = nullptr;
    }

    float& operator() (int r, int c) {
        return data[r * cols + c];
    }

    const float& operator() (int r, int c) const {
        return data[r * cols + c];
    }

    int size() const { return rows * cols; }

    void zero() {
        std::memset(data, 0, rows * cols * sizeof(float));
    }
};

Tensor matmul(const Tensor& A, const Tensor& B);

#endif
