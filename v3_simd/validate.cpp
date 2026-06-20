/*
 * v3_simd/validate.cpp
 *
 * Correctness validation: compares SIMD int8 output vs scalar int8 output
 * across 10 random inputs (seed=12345, N(0,1) normal distribution).
 */

#include "int8_simd.hpp"

int main() {
    std::cout << "=== SIMD Correctness Validation (scalar vs AVX2) ===\n\n";

    auto layers = load_int8_weights("v2_int8/weights_int8.bin");
    std::cout << "Loaded " << layers.size() << " layers\n\n";

    constexpr int NUM_INPUTS = 10;
    constexpr unsigned SEED = 12345;
    std::mt19937 rng(SEED);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::cout << "Input distribution: N(0, 1) normal, seed=" << SEED << "\n\n";

    float global_max_err = 0.0f;
    float global_sum_err = 0.0f;
    int total_elements = 0;
    int argmax_matches = 0;

    std::cout << std::setw(5) << "Input"
              << "  " << std::setw(14) << "max_err"
              << "  " << std::setw(14) << "mean_err"
              << "  " << std::setw(12) << "f32_am"
              << "  " << std::setw(12) << "simd_am"
              << "  " << std::setw(6) << "match" << "\n";
    std::cout << std::string(70, '-') << "\n";

    Int8Context scalar_ctx;
    Int8Context simd_ctx;

    for (int t = 0; t < NUM_INPUTS; t++) {
        Tensor test_input(1, 128);
        for (int i = 0; i < 128; i++)
            test_input.data[i] = dist(rng);

        forward_scalar_inplace(layers, test_input, scalar_ctx);
        forward_simd_inplace(layers, test_input, simd_ctx);

        float max_err = 0.0f;
        float sum_err = 0.0f;
        for (int i = 0; i < 10; i++) {
            float err = std::abs(scalar_ctx.act2.data[i] - simd_ctx.act2.data[i]);
            max_err = std::max(max_err, err);
            sum_err += err;
        }
        float mean_err = sum_err / 10.0f;

        global_max_err = std::max(global_max_err, max_err);
        global_sum_err += sum_err;
        total_elements += 10;

        int am_scalar = argmax(scalar_ctx.act2.data, 10);
        int am_simd   = argmax(simd_ctx.act2.data, 10);
        bool match = (am_scalar == am_simd);
        if (match) argmax_matches++;

        std::cout << std::setw(5) << (t + 1)
                  << "  " << std::scientific << std::setprecision(4) << std::setw(14) << max_err
                  << "  " << std::setw(14) << mean_err
                  << "  " << std::setw(12) << am_scalar
                  << "  " << std::setw(12) << am_simd
                  << "  " << std::setw(6) << (match ? "✓" : "✗") << "\n";
    }

    float global_mean_err = global_sum_err / total_elements;

    std::cout << "\n=== Aggregate Results ===\n";
    std::cout << "  Max abs error (SIMD vs scalar, across all inputs):  "
              << std::scientific << global_max_err << "\n";
    std::cout << "  Mean abs error (SIMD vs scalar, across all inputs): "
              << std::scientific << global_mean_err << "\n";
    std::cout << "  Argmax match rate: " << argmax_matches << "/" << NUM_INPUTS
              << " (" << std::fixed << (100.0 * argmax_matches / NUM_INPUTS) << "%)\n";

    if (global_max_err > 1e-3) {
        std::cout << "\n⚠ WARNING: Max error > 1e-3 — SIMD implementation may have a bug!\n";
    } else {
        std::cout << "\n✓ SIMD output matches scalar to within floating-point tolerance.\n";
    }

    return 0;
}
