/*
 * v3_simd/main.cpp
 *
 * AVX2-accelerated int8 inference engine.
 * Standalone — loads weights_int8.bin, runs 3-layer forward pass with SIMD matmul.
 */

#include "int8_simd.hpp"

using Clock = std::chrono::high_resolution_clock;

int main(int argc, char* argv[]) {
    std::string weights_path =
        (argc >= 2) ? argv[1] : "v2_int8/weights_int8.bin";

    std::cout << "=== Int8 SIMD Inference Engine (AVX2) ===\n";

    auto layers = load_int8_weights(weights_path);
    std::cout << "Loaded " << layers.size() << " layers from " << weights_path << "\n";

    Tensor input(1, 128);
    for (int i = 0; i < input.size(); i++) input.data[i] = 1.0f;

    Int8Context ctx;

    // Warmup
    for (int w = 0; w < 200; w++) {
        forward_simd_inplace(layers, input, ctx);
    }

    constexpr int RUNS = 1000;
    std::vector<double> times;
    times.reserve(RUNS);
    volatile float sink = 0.0f;

    for (int r = 0; r < RUNS; r++) {
        auto t0 = Clock::now();
        forward_simd_inplace(layers, input, ctx);
        sink += ctx.act2.data[0];
        auto t1 = Clock::now();
        double us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
        times.push_back(us);
    }

    std::sort(times.begin(), times.end());
    double p50 = times[RUNS * 50 / 100];
    double p95 = times[RUNS * 95 / 100];
    double p99 = times[RUNS * 99 / 100];

    std::cout << "\nForward pass: p50=" << std::fixed << std::setprecision(3)
              << p50 << " us  p95=" << p95 << "  p99=" << p99 << "\n";

    std::cout << "\nOutput: [";
    for (int i = 0; i < ctx.act2.cols; i++) {
        std::cout << std::fixed << std::setprecision(4) << ctx.act2(0, i);
        if (i < ctx.act2.cols - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    (void)sink;
    return 0;
}
