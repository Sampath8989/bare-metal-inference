#!/usr/bin/env python3
"""
plot_histogram.py

Reads benchmarks/latency_results.csv and plots two overlapping histograms
(float32 vs int8 latency distribution) on the same chart.

Usage:
  python3 scripts/plot_histogram.py
  → produces benchmarks/latency_histogram.png
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

CSV_PATH = "benchmarks/latency_results.csv"
OUT_PATH = "benchmarks/latency_histogram.png"


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} samples from {CSV_PATH}")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        df["float32_us"],
        bins=80,
        alpha=0.65,
        color="#2196F3",
        edgecolor="white",
        linewidth=0.3,
        label=f"float32  (p50={df['float32_us'].median():.3f} µs)",
    )
    ax.hist(
        df["int8_us"],
        bins=80,
        alpha=0.65,
        color="#FF5722",
        edgecolor="white",
        linewidth=0.3,
        label=f"int8  (p50={df['int8_us'].median():.3f} µs)",
    )

    ax.set_xlabel("Latency (µs)", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title("Forward Pass Latency Distribution: float32 vs int8", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)

    print(f"✓ Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
