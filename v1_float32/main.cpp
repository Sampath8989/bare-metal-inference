/*
 * v1_float32/main.cpp - Float32 inference baseline
 *
 * Baselines float32 implementation for verification and comparison.
 *
 * Build:  make float32_v1
 * Run:    ./float32_v1 weights.bin
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <string>
#include "../src/tensor.hpp"
#include "../src/Layer.hpp"

using Clock = std::chrono::high_resolution_clock;
using Us    = std::chrono::microseconds;

/*
 * Binary weight file format (per layer):
 *   [int32 rows][int32 cols][rows*cols float32 weights]
 *   [int32 bias_size][bias_size float32 bias]
 */
std::vector<Layer> load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open weights file: " + path);
    }

    std::vector<Layer> layers;
    int layer_idx = 0;

    while (file.peek() != EOF) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        if (file.gcount() == 0) break;
        file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        if (!file) {
            throw std::runtime_error("Failed to read layer header");
        }

        Layer layer(rows, cols);

        size_t num_weights = static_cast<size_t>(rows) * cols;
        file.read(reinterpret_cast<char*>(layer.weights.data),
                  num_weights * sizeof(float));

        int bias_size;
        file.read(reinterpret_cast<char*>(&bias_size), sizeof(bias_size));
        if (bias_size != cols) {
            std::cerr << "Warning: bias_size=" << bias_size
                      << " != cols=" << cols << "\n";
        }
        file.read(reinterpret_cast<char*>(layer.bias.data),
                  static_cast<size_t>(bias_size) * sizeof(float));

        if (!file) {
            throw std::runtime_error("Failed to read layer " +
                                     std::to_string(layer_idx) + " data");
        }

        std::cout << "  Loaded layer " << (layer_idx + 1)
                  << ": " << rows << "x" << cols
                  << " weights + " << bias_size << " biases\n";

        layers.push_back(std::move(layer));
        layer_idx++;
    }

    return layers;
}

int main(int argc, char* argv[]) {
    std::string weights_path = (argc >= 2) ? argv[1] : "weights.bin";

    std::cout << "=== Float32 Inference Engine Baseline ===\n";
    std::cout << "Loading weights from: " << weights_path << "\n\n";

    std::vector<Layer> layers;
    try {
        layers = load_weights(weights_path);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    if (layers.size() != 3) {
        std::cerr << "ERROR: Expected 3 layers, got " << layers.size() << "\n";
        return 1;
    }

    // Create input tensor (all ones)
    Tensor input(1, 128);
    for (int i = 0; i < input.size(); i++) {
        input.data[i] = 1.0f;
    }

    // Warmup (200 runs, discard timing)
    for (int w = 0; w < 200; w++) {
        Tensor o1 = layers[0].forward(input);
        Tensor o2 = layers[1].forward(o1);
        Tensor o3 = layers[2].forward(o2);
        (void)o3(0, 0);
    }

    // Benchmark (1000 runs)
    constexpr int RUNS = 1000;
    std::vector<long long> times;
    times.reserve(RUNS);

    Tensor final_out(1, 10);
    volatile float sink = 0.0f;

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

    std::cout << "\nForward pass: p50=" << p50 << " us"
              << "  p95=" << p95 << "  p99=" << p99 << "\n";

    // Print output vector
    std::cout << "\nOutput: [";
    for (int i = 0; i < final_out.cols; i++) {
        std::cout << std::fixed << std::setprecision(4) << final_out(0, i);
        if (i < final_out.cols - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    (void)sink;
    return 0;
}
