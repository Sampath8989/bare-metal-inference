#!/usr/bin/env python3
"""
plot_histogram_v3.py
Reads benchmarks/latency_results_v3.csv and plots three overlapping histograms
(float32, int8 scalar, int8 SIMD) on the same chart.

Usage:
  python3 scripts/plot_histogram_v3.py
  → produces benchmarks/latency_histogram_v2.png
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "benchmarks/latency_results_v3.csv"
OUT_PATH = "benchmarks/latency_histogram_v2.png"


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {CSV_PATH}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        df["float32_us"],
        bins=80, alpha=0.5, color="#2196F3", edgecolor="white", linewidth=0.3,
        label=f"float32  (p50={df['float32_us'].median():.3f} µs)",
    )
    ax.hist(
        df["scalar_int8_us"],
        bins=80, alpha=0.5, color="#FF5722", edgecolor="white", linewidth=0.3,
        label=f"int8 scalar  (p50={df['scalar_int8_us'].median():.3f} µs)",
    )
    ax.hist(
        df["simd_int8_us"],
        bins=80, alpha=0.5, color="#4CAF50", edgecolor="white", linewidth=0.3,
        label=f"int8 SIMD  (p50={df['simd_int8_us'].median():.3f} µs)",
    )

    ax.set_xlabel("Latency (µs)", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title("Forward Pass Latency: float32 vs int8 scalar vs int8 SIMD (batch=100)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)

    print(f"✓ Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
