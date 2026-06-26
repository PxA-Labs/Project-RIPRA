# RIPRA Wavefront Reconstruction - Docker Build Environment
#
# Build:
#   docker build -t rippra:latest .
#
# Run (interactive):
#   docker run --rm -it --gpus all rippra:latest
#
# Run (C reconstructor benchmark):
#   docker run --rm rippra:latest rippra/build_and_test.sh

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

LABEL maintainer="RIPRA Team"
LABEL description="RIPRA Wavefront Reconstruction & Turbulence Characterization"

ENV DEBIAN_FRONTEND=noninteractive

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    make \
    wget \
    git \
    python3 \
    python3-pip \
    python3-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python ML dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    torch \
    onnx \
    onnxruntime-gpu \
    scipy

WORKDIR /workspace

# Copy entire project
COPY rippra/ ./rippra/
COPY docs/    ./docs/
COPY config/  ./config/

# Build C library (static)
WORKDIR /workspace/rippra
RUN mkdir -p bin && \
    gcc -std=c99 -Wall -Wextra -O2 -fopenmp -DNDEBUG \
        -Iinclude -c src/centroid.c -o bin/centroid.o && \
    gcc -std=c99 -Wall -Wextra -O2 -fopenmp -DNDEBUG \
        -Iinclude -c src/io.c -o bin/io.o && \
    gcc -std=c99 -Wall -Wextra -O2 -fopenmp -DNDEBUG \
        -Iinclude -c src/la.c -o bin/la.o && \
    gcc -std=c99 -Wall -Wextra -O2 -fopenmp -DNDEBUG \
        -Iinclude -c src/recon.c -o bin/recon.o && \
    gcc -std=c99 -Wall -Wextra -O2 -fopenmp -DNDEBUG \
        -Iinclude -c src/stream.c -o bin/stream.o && \
    gcc -std=c99 -Wall -Wextra -O2 -fopenmp -DNDEBUG \
        -Iinclude -c src/rippra_api.c -o bin/rippra_api.o && \
    gcc -shared -fopenmp -o bin/librippra.so \
        bin/centroid.o bin/io.o bin/la.o bin/recon.o bin/stream.o bin/rippra_api.o && \
    echo "Build complete"

# Build test programs
RUN gcc -std=c99 -Wall -Wextra -O2 -fopenmp -DNDEBUG \
        -Iinclude -c tests/test_recon.c -o bin/test_recon.o && \
    gcc -fopenmp -o bin/test_recon \
        bin/test_recon.o bin/centroid.o bin/io.o bin/la.o bin/recon.o bin/stream.o && \
    echo "Test binaries built"

# Export ONNX models
RUN cd ml && python3 export_onnx.py --output_dir /workspace/rippra/onnx_models && \
    echo "ONNX export complete"

# Run baseline tests
RUN echo "=== Testing C reconstructor ===" && \
    ./bin/test_recon && \
    echo "=== C tests passed ==="

# Default command
CMD ["/bin/bash"]
