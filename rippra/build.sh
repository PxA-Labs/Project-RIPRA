#!/bin/bash
# build.sh — build the RIPPA C pipeline
# Usage: bash build.sh          (release build)
#        bash build.sh debug    (debug build with -g -O0)

set -e
cd "$(dirname "$0")"

MODE="${1:-release}"
echo "=== RIPPA build ($MODE) ==="

CFLAGS="-std=c99 -Wall -Wextra -D_POSIX_SOURCE"
if [ "$MODE" = "debug" ]; then
    CFLAGS="$CFLAGS -g -O0 -DRIPPA_DEBUG"
else
    CFLAGS="$CFLAGS -O2 -DNDEBUG"
fi

echo "CFLAGS: $CFLAGS"

mkdir -p bin

SRCFILES=$(ls src/*.c 2>/dev/null)
if [ -z "$SRCFILES" ]; then
    echo "ERROR: no source files in src/"
    exit 1
fi

echo "Compiling library..."
OBJS=""
for f in $SRCFILES; do
    obj="bin/$(basename "$f" .c).o"
    echo "  $f -> $obj"
    gcc $CFLAGS -Iinclude -c "$f" -o "$obj"
    OBJS="$OBJS $obj"
done

echo "Linking librippra.a..."
ar rcs bin/librippra.a $OBJS
ranlib bin/librippra.a 2>/dev/null || true

echo ""
echo "=== Build successful ==="
echo "Library: bin/librippra.a"
echo ""
echo "To build a test or main program, link with:"
echo "  gcc $CFLAGS -Iinclude your_main.c -Lbin -lrippra -lm -o bin/your_program"
