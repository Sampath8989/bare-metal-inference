#!/usr/bin/env python3
"""
gpu_benchmark/plot_results.py

Dynamically reads gpu_results.csv and generates three accurate, publication-ready charts:
  1. gpu_all_optimizations.png    - Bar chart of all 6 GPU configurations across 128 to 2048 (Log scale)
  2. gpu_baseline_vs_combined.png - Direct comparison of Baseline FP32 vs Combined with speedup annotations
  3. gpu_speedup.png              - Speedup bars relative to Baseline FP32 with 1.0x reference line
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_gpu_results(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Result file '{filepath}' not found.", file=sys.stderr)
        print("Please compile and run the GPU benchmark first (./benchmark_gpu) to generate it.", file=sys.stderr)
        sys.exit(1)

    data = {}
    with open(filepath, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                size = int(parts[0])
                opt = parts[1]
                p50 = float(parts[2])
                p95 = float(parts[3])
                p99 = float(parts[4])
                speedup = float(parts[5])
                err = float(parts[6]) if len(parts) > 6 else 0.0
            except ValueError as e:
                print(f"Error parsing row '{line}': {e}", file=sys.stderr)
                continue

            if size not in data:
                data[size] = {}
            data[size][opt] = {
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "speedup": speedup,
                "max_error": err
            }

    if not data:
        print(f"Error: No valid data found in '{filepath}'.", file=sys.stderr)
        sys.exit(1)

    return data

def plot_all_configurations(data, output_dir):
    """Graph 1: GPU Optimization Benchmark — All Configurations"""
    sizes = sorted(data.keys())
    opts = ["Baseline", "FP16", "Shared Memory", "WMMA Tensor Cores", "Kernel Fusion", "Combined"]
    colors = ["#E74C3C", "#E67E22", "#1ABC9C", "#F1C40F", "#9B59B6", "#2ECC71"]

    x = np.arange(len(sizes))
    n_opts = len(opts)
    width = 0.13

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    for idx, opt in enumerate(opts):
        y_vals = []
        for s in sizes:
            if opt in data[s]:
                y_vals.append(data[s][opt]["p50"])
            else:
                y_vals.append(np.nan)
        offset = (idx - n_opts / 2 + 0.5) * width
        ax.bar(x + offset, y_vals, width, label=opt, color=colors[idx], edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Matrix Size (N × N)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (µs) — Log Scale", fontsize=12, fontweight="bold")
    ax.set_title("GPU Optimization Benchmark — All Configurations", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes], fontsize=11)
    ax.set_yscale("log")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "gpu_all_optimizations.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✓ Saved {out_path}")

def plot_baseline_vs_combined(data, output_dir):
    """Graph 2: GPU Baseline vs Final Combined Performance"""
    sizes = sorted(data.keys())
    x = np.arange(len(sizes))
    width = 0.35

    base_vals = [data[s]["Baseline"]["p50"] if "Baseline" in data[s] else np.nan for s in sizes]
    comb_vals = [data[s]["Combined"]["p50"] if "Combined" in data[s] else np.nan for s in sizes]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    b1 = ax.bar(x - width/2, base_vals, width, label="Baseline (Naive FP32 CUDA)", color="#E74C3C", edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, comb_vals, width, label="Combined (FP16 + Shared Memory + WMMA + Fusion)", color="#2ECC71", edgecolor="black", linewidth=0.8)

    for bar1, bar2, s in zip(b1, b2, sizes):
        h1 = bar1.get_height()
        h2 = bar2.get_height()
        speedup = h1 / h2 if (h2 and not np.isnan(h2) and h2 > 0) else 0.0
        if not np.isnan(h1):
            ax.annotate(f"{h1:.1f} µs", xy=(bar1.get_x() + bar1.get_width()/2, h1),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
        if not np.isnan(h2):
            ax.annotate(f"{h2:.1f} µs\n({speedup:.2f}×)", xy=(bar2.get_x() + bar2.get_width()/2, h2),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Matrix Size (N × N)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency (µs) — Log Scale", fontsize=12, fontweight="bold")
    ax.set_title("GPU Baseline vs Final Combined Performance", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes], fontsize=11)
    ax.set_yscale("log")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "gpu_baseline_vs_combined.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✓ Saved {out_path}")

def plot_speedup(data, output_dir):
    """Graph 3: GPU Optimization Speedup Relative to Baseline"""
    sizes = sorted(data.keys())
    opts = ["FP16", "Shared Memory", "WMMA Tensor Cores", "Kernel Fusion", "Combined"]
    colors = ["#E67E22", "#1ABC9C", "#F1C40F", "#9B59B6", "#2ECC71"]

    x = np.arange(len(sizes))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    for idx, opt in enumerate(opts):
        speedups = []
        for s in sizes:
            if opt in data[s] and "Baseline" in data[s]:
                base_p50 = data[s]["Baseline"]["p50"]
                opt_p50 = data[s][opt]["p50"]
                speedups.append(base_p50 / opt_p50 if opt_p50 > 0 else np.nan)
            elif opt in data[s]:
                speedups.append(data[s][opt]["speedup"])
            else:
                speedups.append(np.nan)

        offset = (idx - len(opts) / 2 + 0.5) * width
        rects = ax.bar(x + offset, speedups, width, label=opt, color=colors[idx], edgecolor="black", linewidth=0.6)

        for rect in rects:
            h = rect.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{h:.1f}×", xy=(rect.get_x() + rect.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="Baseline FP32 (1.0×)")
    ax.set_xlabel("Matrix Size (N × N)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup (×)", fontsize=12, fontweight="bold")
    ax.set_title("GPU Optimization Speedup Relative to Baseline", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}×{s}" for s in sizes], fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "gpu_speedup.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✓ Saved {out_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "gpu_results.csv")

    data = load_gpu_results(csv_path)
    plot_all_configurations(data, script_dir)
    plot_baseline_vs_combined(data, script_dir)
    plot_speedup(data, script_dir)

if __name__ == "__main__":
    main()
