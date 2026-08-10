#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NVCC_FLAGS="-O2 -arch=sm_75 -std=c++17"

echo "═══════════════════════════════════════════════════════════════"
echo "  1/2  Compiling cuda_fusion.cu  →  fusion_demo"
echo "═══════════════════════════════════════════════════════════════"
nvcc $NVCC_FLAGS cuda_fusion.cu -o fusion_demo
echo "  ✓ Compiled"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  2/2  Compiling cuda_streams.cu  →  streams_demo"
echo "═══════════════════════════════════════════════════════════════"
nvcc $NVCC_FLAGS cuda_streams.cu -o streams_demo
echo "  ✓ Compiled"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  Running Kernel Fusion demo (saving to fusion_results.txt)"
echo "═══════════════════════════════════════════════════════════════"
./fusion_demo 2>&1 | tee fusion_results.txt
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Running CUDA Streams demo (saving to streams_results.txt)"
echo "═══════════════════════════════════════════════════════════════"
./streams_demo 2>&1 | tee streams_results.txt
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Done. Results saved to:"
echo "    v9_fusion_streams/fusion_results.txt"
echo "    v9_fusion_streams/streams_results.txt"
echo "═══════════════════════════════════════════════════════════════"
