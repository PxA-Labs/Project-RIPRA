#!/usr/bin/env bash
# OpenMP scaling benchmark — run benchmark_openmp at 1/2/4/8 threads.
# Requires: compiled benchmark_openmp in ../build/, synthetic data in rippra/data_raw/
# Usage: cd tools && bash benchmark_scaling.sh

set -euo pipefail

BENCH="${BENCH:-../build/benchmark_openmp}"
DATA="${DATA:-../rippra/data_raw}"

if [ ! -f "$BENCH" ]; then
    echo "ERROR: $BENCH not found — run 'cmake --build build --target benchmark_openmp' first"
    exit 1
fi

echo "=== RIPRA OpenMP Scaling Benchmark ==="
echo "Binary: $BENCH"
echo ""

for threads in 1 2 4 8; do
    echo "----------------------------------------"
    echo "  OMP_NUM_THREADS=$threads"
    echo "----------------------------------------"
    OMP_NUM_THREADS="$threads" "$BENCH" 2>&1 | tail -n +2
    echo ""
done

echo "=== Done ==="
