#include "tensor.hpp"

Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows)
        throw std::invalid_argument("matmul: A.cols must equal B.rows");

    int m = A.rows, n = B.cols, k = A.cols;
    Tensor C(m, n);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int q = 0; q < k; q++)
                C(i, j) += A(i, q) * B(q, j);

    return C;
}
