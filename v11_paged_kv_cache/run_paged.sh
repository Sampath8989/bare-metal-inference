#!/bin/bash
# Build & run the Paged KV Cache simulator. Usage: run_paged.sh [num_requests] [arrivals_per_step] [seed]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CXX="${CXX:-g++}"
BIN="paged_kv_cache"
OUT="paged_kv_results.txt"

echo "═══════════════════════════════════════════════════════════════"
echo "  Compiling paged_kv_cache.cpp  →  ${BIN}"
echo "═══════════════════════════════════════════════════════════════"
${CXX} -O3 -std=c++17 -Wall -Wextra paged_kv_cache.cpp -o ${BIN}
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
