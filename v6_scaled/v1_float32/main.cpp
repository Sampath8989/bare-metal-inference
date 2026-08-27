// Scaled Float32 baseline inference engine.
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <string>
#include <cstdlib>
#include "tensor.hpp"
#include "Layer.hpp"

using Clock = std::chrono::high_resolution_clock;
using Us    = std::chrono::microseconds;

int main(int argc, char* argv[]) {
    int scale = 512;
    if (argc >= 2) {
        int s = std::atoi(argv[1]);
        if (s > 0) scale = s;
    }

    int in_dim = scale;
    int h1_dim = scale;
    int h2_dim = scale;
    int out_dim = scale;

    std::cout << "=== V1 Float32 Baseline (Scaled: " << scale << "x" << scale << ") ===\n";
    std::cout << "Architecture: " << in_dim << " -> " << h1_dim << " -> " << h2_dim << " -> " << out_dim << "\n";

    std::vector<Layer> layers;
    layers.emplace_back(in_dim, h1_dim);
    layers.emplace_back(h1_dim, h2_dim);
    layers.emplace_back(h2_dim, out_dim);

    // All-ones input
    Tensor input(1, in_dim);
    for (int i = 0; i < input.size(); i++) {
        input.data[i] = 1.0f;
    }

    // Warmup (200 runs)
    for (int w = 0; w < 200; w++) {
        Tensor o1 = layers[0].forward(input);
        Tensor o2 = layers[1].forward(o1);
        Tensor o3 = layers[2].forward(o2);
        (void)o3(0, 0);
    }

    constexpr int RUNS = 1000;
    std::vector<long long> times;
    times.reserve(RUNS);

    Tensor final_out(1, out_dim);
    volatile float sink = 0.0f;

    // Timed runs
    for (int r = 0; r < RUNS; r++) {
        auto t0 = Clock::now();

        Tensor o1 = layers[0].forward(input);
        Tensor o2 = layers[1].forward(o1);
        final_out  = layers[2].forward(o2);

        sink += final_out(0, 0);

        auto t1 = Clock::now();
        times.push_back(std::chrono::duration_cast<Us>(t1 - t0).count());
    }

    std::sort(times.begin(), times.end());
    long long p50 = times[RUNS * 50 / 100];
    long long p95 = times[RUNS * 95 / 100];
    long long p99 = times[RUNS * 99 / 100];

    std::cout << "Size: " << scale << "\n";
    std::cout << "p50: " << p50 << " us\n";
    std::cout << "p95: " << p95 << " us\n";
    std::cout << "p99: " << p99 << " us\n";

    std::cout << "Output sample (first 5): [";
    for (int i = 0; i < std::min(5, final_out.cols); i++) {
        std::cout << std::fixed << std::setprecision(4) << final_out(0, i);
        if (i < std::min(5, final_out.cols) - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";

    (void)sink;
    return 0;
}
