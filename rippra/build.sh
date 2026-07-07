#!/bin/bash
# Thin wrapper around CMake build
# Usage: ./build.sh [release|debug|openmp]
# All modes enable OpenMP by default; the argument is accepted for compatibility.
set -e
cd "$(dirname "$0")/.."
BUILD_TYPE="${1:-release}"
cmake -B build -S . -DRIPRA_BUILD_TESTS=OFF -DRIPRA_BUILD_BENCHMARKS=OFF \
  -DCMAKE_BUILD_TYPE=$(test "$BUILD_TYPE" = debug && echo Debug || echo Release)
cmake --build build --target ripra
