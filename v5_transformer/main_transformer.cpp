// Transformer block + KV cache latency comparison.
#include "transformer_block.hpp"
#include "kv_cache.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

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
              << " µs  p95=" << stats.p95
              << " µs  p99=" << stats.p99 << " µs" << std::endl;
}

int main() {
    std::cout << "=== TRANSFORMER BLOCK + KV CACHE ===" << std::endl;
    std::cout << "(Using std::chrono::steady_clock for all timing)" << std::endl;

    const int d_model = 128;
    const int d_head = 64;
    const int mlp_hidden = 256;
    const int max_tokens = 50;

    TransformerBlock block(d_model, d_head, mlp_hidden);
    init_transformer_block(block);

    Tensor input(1, d_model);
    std::mt19937 gen(12345);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < input.size(); i++) input.data[i] = dis(gen);

    std::cout << "\n=== KV CACHE SPEEDUP COMPARISON (single unified timing pass) ===" << std::endl;
    std::cout << "Timing: std::chrono::steady_clock (monotonic, consistent)" << std::endl;

    std::vector<double> times_no_cache;
    times_no_cache.reserve(max_tokens);
    {
        std::vector<float> seq_buffer;
        seq_buffer.reserve(max_tokens * d_model);
        for (int i = 0; i < input.size(); i++)
            seq_buffer.push_back(input.data[i]);

        for (int t = 0; t < max_tokens; t++) {
            int current_len = t + 1;

            // Alloc + memcpy outside the timed section (fair comparison).
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
        // Pre-allocated cache: zero heap alloc during generation.
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
            cur = std::move(out);   // pointer swap, zero heap alloc
        }
    }

    // print_stats() sorts in place — snapshot raw values first.
    double nc_10 = times_no_cache[9];
    double nc_25 = times_no_cache[24];
    double nc_50 = times_no_cache[49];
    double ck_10 = times_cached[9];
    double ck_25 = times_cached[24];
    double ck_50 = times_cached[49];

    std::cout << "\n--- No cache (full recomputation) ---" << std::endl;
    print_stats("No cache", times_no_cache);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  At seq_len=10: " << nc_10 << " µs" << std::endl;
    std::cout << "  At seq_len=25: " << nc_25 << " µs" << std::endl;
    std::cout << "  At seq_len=50: " << nc_50 << " µs" << std::endl;

    std::cout << "\n--- With KV cache (pre-allocated, zero-alloc loop) ---" << std::endl;
    print_stats("With cache", times_cached);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  At seq_len=10: " << ck_10 << " µs" << std::endl;
    std::cout << "  At seq_len=25: " << ck_25 << " µs" << std::endl;
    std::cout << "  At seq_len=50: " << ck_50 << " µs" << std::endl;

    std::cout << "\n=== KV CACHE SPEEDUP SUMMARY ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "No cache — per-token at seq_len 10: " << nc_10
              << "  µs  25: " << nc_25
              << "  µs  50: " << nc_50 << " µs" << std::endl;
    std::cout << "With cache — per-token at seq_len 10: " << ck_10
              << "  µs  25: " << ck_25
              << "  µs  50: " << ck_50 << " µs" << std::endl;

    std::cout << "\nSpeedup (no-cache / cached):" << std::endl;
    std::cout << "  seq_len=10: " << nc_10 / ck_10 << "x" << std::endl;
    std::cout << "  seq_len=25: " << nc_25 / ck_25 << "x" << std::endl;
    std::cout << "  seq_len=50: " << nc_50 / ck_50 << "x" << std::endl;

    std::cout << "\nKV Cache memory footprint at seq_len=50, d_head=64: "
              << 50 * 64 * 2 * sizeof(float) << " bytes"
              << " (50 × 64 × 2 × 4 — 2=K and V, 4=bytes per float)" << std::endl;

    return 0;
}
