// Scaled Transformer block + KV cache latency comparison.
#include "transformer_block.hpp"
#include "kv_cache.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

using Clock = std::chrono::steady_clock;

struct LatencyStats {
    double p50, p95, p99;
};

LatencyStats compute_stats(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    int n = times.size();
    return {
        times[n * 50 / 100],
        times[n * 95 / 100],
        times[n * 99 / 100]
    };
}

void print_stats(const char* label, std::vector<double>& times) {
    LatencyStats stats = compute_stats(times);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << label << " — p50=" << stats.p50
              << " us  p95=" << stats.p95
              << " us  p99=" << stats.p99 << " us" << std::endl;
}

int main(int argc, char* argv[]) {
    int scale = 512;
    if (argc >= 2) {
        int s = std::atoi(argv[1]);
        if (s > 0) scale = s;
    }

    const int d_model = scale;
    const int d_head = (scale >= 1024) ? 128 : 64;
    const int mlp_hidden = scale * 4;
    const int max_tokens = 50;

    std::cout << "=== V5 Transformer Block + KV Cache (Scaled: " << scale << ") ===" << std::endl;
    std::cout << "Dimensions: d_model=" << d_model << ", d_head=" << d_head
              << ", mlp_hidden=" << mlp_hidden << ", max_tokens=" << max_tokens << std::endl;

    TransformerBlock block(d_model, d_head, mlp_hidden);
    init_transformer_block(block);

    Tensor input(1, d_model);
    std::mt19937 gen(12345);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < input.size(); i++) input.data[i] = dis(gen);

    std::cout << "\n=== KV CACHE SPEEDUP COMPARISON ===" << std::endl;

    std::vector<double> times_no_cache;
    times_no_cache.reserve(max_tokens);
    {
        std::vector<float> seq_buffer;
        seq_buffer.reserve(max_tokens * d_model);
        for (int i = 0; i < input.size(); i++)
            seq_buffer.push_back(input.data[i]);

        for (int t = 0; t < max_tokens; t++) {
            int current_len = t + 1;

            Tensor cur(current_len, d_model);
            std::memcpy(cur.data, seq_buffer.data(), current_len * d_model * sizeof(float));

            auto t0 = Clock::now();
            Tensor out = block.forward(cur);
            auto t1 = Clock::now();

            times_no_cache.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0);

            for (int i = 0; i < d_model; i++)
                seq_buffer.push_back(out.data[(out.rows - 1) * d_model + i]);
        }
    }

    std::vector<double> times_cached;
    times_cached.reserve(max_tokens);
    {
        KVCache cache(d_head, max_tokens);
        Tensor cur = input;

        for (int t = 0; t < max_tokens; t++) {
            auto t0 = Clock::now();

            Tensor attn_out = attention_forward_cached(cur, block.attn, &cache);

            Tensor residual(cur.rows, cur.cols);
            for (int i = 0; i < residual.size(); i++) residual.data[i] = cur.data[i] + attn_out.data[i];
            layer_norm_inplace(residual);
            Tensor hidden = block.mlp1.forward(residual, true);
            Tensor out = block.mlp2.forward(hidden, true);
            auto t1 = Clock::now();

            times_cached.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0);
            cur = std::move(out);
        }
    }

    double nc_10 = times_no_cache[9];
    double nc_25 = times_no_cache[24];
    double nc_50 = times_no_cache[49];
    double ck_10 = times_cached[9];
    double ck_25 = times_cached[24];
    double ck_50 = times_cached[49];

    std::cout << "\n--- No cache (full recomputation) ---" << std::endl;
    print_stats("No cache", times_no_cache);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  At seq_len=10: " << nc_10 << " us" << std::endl;
    std::cout << "  At seq_len=25: " << nc_25 << " us" << std::endl;
    std::cout << "  At seq_len=50: " << nc_50 << " us" << std::endl;

    std::cout << "\n--- With KV cache (pre-allocated, zero-alloc loop) ---" << std::endl;
    print_stats("With cache", times_cached);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  At seq_len=10: " << ck_10 << " us" << std::endl;
    std::cout << "  At seq_len=25: " << ck_25 << " us" << std::endl;
    std::cout << "  At seq_len=50: " << ck_50 << " us" << std::endl;

    std::cout << "\n=== KV CACHE SPEEDUP SUMMARY ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Speedup (no-cache / cached):" << std::endl;
    std::cout << "  seq_len=10: " << nc_10 / ck_10 << "x" << std::endl;
    std::cout << "  seq_len=25: " << nc_25 / ck_25 << "x" << std::endl;
    std::cout << "  seq_len=50: " << nc_50 / ck_50 << "x" << std::endl;

    std::cout << "\nKV Cache memory footprint at seq_len=50, d_head=" << d_head << ": "
              << 50 * d_head * 2 * sizeof(float) << " bytes\n\n";

    return 0;
}
