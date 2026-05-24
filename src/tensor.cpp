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
    // Initialize to zero using the struct's method
    // ── LOOP ORDER: i-k-j (NOT i-j-k) ─────────────────────────────────────
    //
    // Memory layout: both A and B are row-major. Element (r,c) = data[r*cols+c].
    //
    // i-j-k (WRONG — cache-unfriendly):
    //   Inner loop: for j: C[i][j] += A[i][k] * B[k][j]
    //   B[k][j] accesses column j — stride of N floats between accesses.
    //   For N=128: each B access jumps 512 bytes. L1 cache line = 64 bytes.
    //   Result: cache miss on every single inner-loop iteration. Brutal.
    //
    // i-k-j (CORRECT — cache-friendly):
    //   Hoist A[i][k] out of inner loop (it doesn't change when j varies).
    //   Inner loop: for j: C[i][j] += a_ik * B[k][j]
    //   Both C[i][j] and B[k][j] advance sequentially. Prefetcher is happy.
    //   Result: ~3-5x fewer cache misses on large matrices.

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