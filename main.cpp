#include <iostream>
#include <chrono>
#include "tensor.hpp"

using Clock = std::chrono::high_resolution_clock;
using Us    = std::chrono::microseconds;

int main() {
    // sanity check
    Tensor A(1, 2), B(2, 1);
    A(0,0) = 4; A(0,1) = 6;
    B(0,0) = 6; B(1,0) = 9;
    Tensor C = matmul(A, B);
    std::cout << "matmul check (expect 78): " << C(0,0) << "\n\n";

    // 128x128 baseline
    Tensor X(128, 128), Y(128, 128);
    for (int i = 0; i < 128*128; i++) { X.data[i] = 0.01f; Y.data[i] = 0.01f; }

    auto t0 = Clock::now();
    Tensor Z = matmul(X, Y);
    auto t1 = Clock::now();

    std::cout << "128x128 matmul: "
              << std::chrono::duration_cast<Us>(t1-t0).count() << " us\n";
    return 0;
}
