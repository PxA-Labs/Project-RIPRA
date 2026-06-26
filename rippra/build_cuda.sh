#!/bin/bash
# build_cuda.sh - Build RIPPA with CUDA support
# Usage: bash build_cuda.sh
set -e
cd "$(dirname "$0")"

MODE="${1:-release}"
echo "=== RIPPA CUDA build ($MODE) ==="

CFLAGS="-std=c99 -Wall -Wextra -D_POSIX_SOURCE"
CUFLAGS=""
LDFLAGS=""
if [ "$MODE" = "debug" ]; then
    CFLAGS="$CFLAGS -g -O0 -DRIPPA_DEBUG"
    CUFLAGS="-g -G"
else
    CFLAGS="$CFLAGS -O2 -DNDEBUG"
    CUFLAGS="-O2"
fi

mkdir -p bin

echo "Compiling C sources..."
for src in src/*.c; do
    obj="bin/$(basename "$src" .c).o"
    echo "  $src -> $obj"
    gcc $CFLAGS -Iinclude -c "$src" -o "$obj"
done

echo "Compiling CUDA sources..."
CUDA_OBJS=""
for cu in cuda/*.cu; do
    obj="bin/$(basename "$cu" .cu)_cuda.o"
    echo "  $cu -> $obj"
    nvcc $CUFLAGS -Iinclude -dc "$cu" -o "$obj"
    CUDA_OBJS="$CUDA_OBJS $obj"
done

if [ -n "$CUDA_OBJS" ]; then
    echo "Linking CUDA device code..."
    nvcc $CUFLAGS -dlink $CUDA_OBJS -o bin/cuda_link.o -lcudart
    CUDA_OBJS="$CUDA_OBJS bin/cuda_link.o"
fi

echo "Linking librippra.a..."
ar rcs bin/librippra.a bin/*.o
ranlib bin/librippra.a 2>/dev/null || true

echo ""
echo "=== CUDA build successful ==="
echo "Library: bin/librippra.a"
echo ""
echo "To build CUDA test, run:"
echo "  bash build_cuda_test.sh"