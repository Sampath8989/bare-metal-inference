#include <iostream>
#include <chrono>
#include "tensor.hpp"
#include "Layer.hpp"

using Clock = std::chrono::high_resolution_clock;
using Us    = std::chrono::microseconds;

int main() {
    Layer layer1(128, 64);
    Layer layer2(64,  32);
    Layer layer3(32,  10);

    Tensor input(1, 128);
    for (int i = 0; i < input.size(); i++) input.data[i] = 1.0f;

    auto t0 = Clock::now();
    Tensor o1 = layer1.forward(input);
    Tensor o2 = layer2.forward(o1);
    Tensor o3 = layer3.forward(o2);
    auto t1 = Clock::now();

    for (int i = 0; i < o3.cols; i++)
        std::cout << "out[" << i << "] = " << o3(0,i) << "\n";

    std::cout << "\nforward pass: "
              << std::chrono::duration_cast<Us>(t1-t0).count() << " us\n";
    return 0;
}
