#include "int8_simd.hpp"

using Clock = std::chrono::high_resolution_clock;

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

    std::cout << "=== V3 Int8 SIMD Inference Engine (AVX2, Scaled: " << scale << "x" << scale << ") ===\n";
    std::cout << "Architecture: " << in_dim << " -> " << h1_dim << " -> " << h2_dim << " -> " << out_dim << "\n";

    std::vector<Int8Layer> layers;
    layers.emplace_back(in_dim, h1_dim, 0.001f);
    layers.emplace_back(h1_dim, h2_dim, 0.001f);
    layers.emplace_back(h2_dim, out_dim, 0.001f);

    Tensor input(1, in_dim);
    for (int i = 0; i < input.size(); i++) input.data[i] = 1.0f;

    Int8Context ctx(in_dim, h1_dim, h2_dim, out_dim);

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

    std::cout << "Size: " << scale << "\n";
    std::cout << "p50: " << std::fixed << std::setprecision(3) << p50 << " us\n";
    std::cout << "p95: " << p95 << " us\n";
    std::cout << "p99: " << p99 << " us\n";

    std::cout << "Output sample (first 5): [";
    for (int i = 0; i < std::min(5, ctx.act2.cols); i++) {
        std::cout << std::fixed << std::setprecision(4) << ctx.act2(0, i);
        if (i < std::min(5, ctx.act2.cols) - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";

    (void)sink;
    return 0;
}
