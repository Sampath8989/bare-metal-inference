#!/bin/bash
set -e

echo "============================================================"
echo "Step 1: Check GPU and CUDA Environment"
echo "============================================================"
nvidia-smi
nvcc --version

echo ""
echo "============================================================"
echo "Step 2: Compile Master GPU Benchmark (NVIDIA T4 / sm_75)"
echo "============================================================"
nvcc -O3 -arch=sm_75 -std=c++17 main.cu -o benchmark_gpu
echo "✓ Compilation successful: ./benchmark_gpu"

echo ""
echo "============================================================"
echo "Step 3: Run Master GPU Benchmark (128x128 -> 2048x2048)"
echo "============================================================"
./benchmark_gpu

echo ""
echo "============================================================"
echo "Step 4: Generate Benchmark Plots"
echo "============================================================"
python3 plot_results.py

echo ""
echo "============================================================"
echo "Step 5: Benchmark Completed Successfully"
echo "============================================================"
echo "Generated Results:"
ls -lh gpu_results.csv gpu_results.txt *.png
