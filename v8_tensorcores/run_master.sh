#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  Compiling cuda_wmma_master.cu..."
echo "═══════════════════════════════════════════════════════════════"
nvcc -O2 -arch=sm_75 -std=c++17 cuda_wmma_master.cu -o wmma_master
echo "  ✓ Compiled: wmma_master"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  Running benchmark..."
echo "═══════════════════════════════════════════════════════════════"
./wmma_master 2>&1 | tee wmma_master_results.txt
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Results saved to: wmma_master_results.txt"
echo "═══════════════════════════════════════════════════════════════"
