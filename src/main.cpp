#include <iostream>
#include <chrono>
#include "tensor.hpp"

using Clock = std::chrono::high_resolution_clock;
using Us = std::chrono::microseconds;

int main() {
    std::cout << "Start small. Ship something.\n\n";

    // Test Tensor
    Tensor w(4, 3);
    for (int i = 0; i < w.rows; i++)
        for (int j = 0; j < w.cols; j++)
            w(i, j) = static_cast<float>(i * w.cols + j);

    std::cout << "Value at w(1,2): " << w(1, 2) << "\n\n";

    // Test matmul
    Tensor A(1, 2);
    A(0, 0) = 4; A(0, 1) = 6;

    Tensor B(2, 1);
    B(0, 0) = 6;
    B(1, 0) = 9;

    auto t0 = Clock::now();
    Tensor C = matmul(A, B);
    auto t1 = Clock::now();

    std::cout << "Matmul Result:\n";
    for (int i = 0; i < C.rows; i++) {
        for (int j = 0; j < C.cols; j++)
            std::cout << C(i, j) << "  ";
        std::cout << "\n";
    }

    auto elapsed = std::chrono::duration_cast<Us>(t1 - t0).count();
    std::cout << "matmul = " << elapsed << " us\n";
    std::cout << "C(0,0) = " << C(0, 0) << "\n";

    return 0;
}