#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include "tensor.hpp"

using Clock = std::chrono::high_resolution_clock;
using µs    = std::chrono::microseconds;

int main() {
    constexpr int N    = 128;
    constexpr int RUNS = 20;   // more runs = more stable

    Tensor A(N, N), B(N, N);
    for (int i = 0; i < N*N; i++) {
        A.data[i] = static_cast<float>(i % 7);
        B.data[i] = static_cast<float>(i % 5);
    }

    // Extended warm-up: force CPU to full frequency
    // Run 5 times before timing anything
    for (int w = 0; w < 5; w++) {
        Tensor tmp = matmul(A, B); (void)tmp;
    }

    // Timed runs
    std::vector<long long> times;
    times.reserve(RUNS);

    for (int run = 0; run < RUNS; run++) {
        auto t0 = Clock::now();
        Tensor C = matmul(A, B);
        auto t1 = Clock::now();
        auto elapsed = std::chrono::duration_cast<µs>(t1-t0).count();
        times.push_back(elapsed);
        (void)C(0,0);
    }

    // Statistics
    std::sort(times.begin(), times.end());

    long long p50 = times[RUNS * 50 / 100];
    long long p95 = times[RUNS * 95 / 100];
    long long p99 = times[RUNS * 99 / 100];
    long long best = times.front();
    long long worst = times.back();

    // Print all runs so you can see the ramp-up
    std::cout << "All runs:\n";
    for (int i = 0; i < RUNS; i++) {
        std::cout << "  run " << i+1 << ": " << times[i] << " µs";
        if (i < 5) std::cout << "  ← warm-up spike (ignore)";
        std::cout << "\n";
    }

    std::cout << "\n";
    std::cout << "=== FLOAT32 BASELINE RESULTS ===\n";
    std::cout << "  Best  : " << best  << " µs\n";
    std::cout << "  p50   : " << p50   << " µs  ← your real number\n";
    std::cout << "  p95   : " << p95   << " µs\n";
    std::cout << "  p99   : " << p99   << " µs\n";
    std::cout << "  Worst : " << worst << " µs\n";
    std::cout << "\nWRITE DOWN YOUR p50: " << p50 << " µs\n";
    std::cout << "Baseline execution complete.\n";

    return 0;
}