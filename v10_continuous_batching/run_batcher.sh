#!/bin/bash
# Build & run the continuous batching simulator. Usage: run_batcher.sh [num_requests] [max_batch] [mean_arrival_us] [seed]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CXX="${CXX:-g++}"
BIN="continuous_batcher"
OUT="batcher_results.txt"

echo "═══════════════════════════════════════════════════════════════"
echo "  Compiling continuous_batcher.cpp  →  ${BIN}"
echo "═══════════════════════════════════════════════════════════════"
${CXX} -O3 -std=c++17 -Wall -Wextra continuous_batcher.cpp -o ${BIN}
echo "  ✓ Compiled"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  Running simulator (saving to ${OUT})"
echo "═══════════════════════════════════════════════════════════════"
./${BIN} "$@" 2>&1 | tee ${OUT}
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  Done. Results saved to: ${OUT}"
echo "═══════════════════════════════════════════════════════════════"
