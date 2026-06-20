#include "tensor.hpp"

Tensor matmul(const Tensor& A, const Tensor& B) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("A.cols must equal B.rows");
    }

    int m = A.rows;
    int n = B.cols;
    int k = A.cols;
    
    Tensor C(m, n);
    C.zero();
    // Loop order: i-k-j (cache-friendly)
    // Row-major traversal: hoisting a_iq allows sequential access to B and C,
    // maximizing cache-line utilization and hardware prefetching.

    const float* __restrict__ a = A.data;
    const float* __restrict__ b = B.data;
          float* __restrict__ c = C.data;

    for (int i = 0; i < m; i++) {
        for (int q = 0; q < k; q++) {
            float a_iq = a[i * k + q];
            for (int j = 0; j < n; j++) {
                c[i * n + j] += a_iq * b[q * n + j];
            }
        }
    }
    
    return C;
}