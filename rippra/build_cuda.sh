#!/bin/bash
# Thin CMake wrapper: build with CUDA support (auto-detected by CMake)
# Usage: bash build_cuda.sh [release|debug]
set -e
cd "$(dirname "$0")/.."
BUILD_TYPE="${1:-release}"
cmake -B build -S . -DRIPRA_BUILD_TESTS=OFF -DRIPRA_BUILD_BENCHMARKS=OFF \
  -DCMAKE_BUILD_TYPE=$(test "$BUILD_TYPE" = debug && echo Debug || echo Release)
cmake --build build --target ripra
