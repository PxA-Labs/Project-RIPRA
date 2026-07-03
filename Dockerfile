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
COPY . .

# Build C library and tests using CMake
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build

# Generate reference centroids and export ONNX models
RUN cd rippra && python3 ml/synthetic_shwfs.py && \
    mkdir -p results && ./build/test_centroid && \
    cd ml && python3 export_onnx.py --output_dir /workspace/rippra/onnx_models && \
    echo "ONNX export complete"

# Run baseline tests
RUN echo "=== Testing C reconstructor ===" && \
    ./build/test_recon && \
    echo "=== C tests passed ==="

# Default command
CMD ["/bin/bash"]

