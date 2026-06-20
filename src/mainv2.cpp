#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "tensor.hpp"
#include "Layer.hpp"

using Clock = std::chrono::high_resolution_clock;
using Us    = std::chrono::microseconds;

int main() {
    std::cout << "--- Initializing 3-Layer Neural Network ---\n\n";

    Layer layer1(128, 64);
    Layer layer2(64,  32);
    Layer layer3(32,  10);

    Tensor input(1, 128);
    for (int i = 0; i < input.size(); i++)
        input.data[i] = 1.0f;

    // Warmup (200 runs, discard timing)
    for (int w = 0; w < 200; w++) {
        Tensor o1 = layer1.forward(input);
        Tensor o2 = layer2.forward(o1);
        Tensor o3 = layer3.forward(o2);
        (void)o3(0, 0);
    }

    // Benchmark (1000 runs)
    constexpr int RUNS = 1000;
    std::vector<long long> times;
    times.reserve(RUNS);

    Tensor final_out(1, 10); // declared outside so it's accessible later

    for (int r = 0; r < RUNS; r++) {
        auto t0 = Clock::now();

        Tensor o1 = layer1.forward(input);
        Tensor o2 = layer2.forward(o1);
        final_out  = layer3.forward(o2); // store last run's output

        (void)final_out(0, 0);

        auto t1 = Clock::now();
        times.push_back(std::chrono::duration_cast<Us>(t1 - t0).count());
    }

    std::sort(times.begin(), times.end());
    long long p50 = times[RUNS * 50 / 100];
    long long p95 = times[RUNS * 95 / 100];
    long long p99 = times[RUNS * 99 / 100];

    std::cout << "p50 : " << p50 << " us\n"
              << "p95 : " << p95 << " us\n"
              << "p99 : " << p99 << " us\n";
    std::cout << "Baseline execution complete.\n";

    // Output vs Golden comparison
    std::cout << "\n=== C++ OUTPUT ===\n";
    for (int i = 0; i < final_out.cols; i++) {
        std::cout << "   cpp[" << i << "] = "
                  << std::fixed << std::setprecision(6)
                  << final_out(0, i) << "\n";
    }

    std::ifstream golden_file("golden_output.txt");
    if (!golden_file.is_open()) {
        std::cerr << "ERROR: golden_output.txt not found!\n";
        return 1;
    }

    float max_err = 0.0f;
    for (int i = 0; i < final_out.cols; i++) {
        float ref;
        golden_file >> ref;
        float err  = std::abs(final_out(0, i) - ref);
        max_err    = std::max(max_err, err);
    }
    std::cout << "Max error vs golden : " << std::scientific << max_err << "\n";

    return 0;
}