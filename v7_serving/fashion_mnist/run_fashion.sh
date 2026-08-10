#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo " Fashion MNIST Validation Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "[Step 1/3] Compiling inference_fashion.cpp..."
g++ -O2 -std=c++17 -o inference_fashion inference_fashion.cpp -lm
echo "  ✓ Compiled: inference_fashion"
echo ""

echo "[Step 2/3] Training Fashion MNIST MLP..."
python3 train_fashion_mnist.py
echo ""

echo "[Step 3/3] Running C++ inference..."
./inference_fashion
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo " Pipeline complete!"
echo "═══════════════════════════════════════════════════════════════"
