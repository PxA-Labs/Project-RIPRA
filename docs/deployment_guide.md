# Deployment Guide

## Shared Library

### Build DLL

```bash
cd rippra
build_dll.bat
```

Output:
- `rippra/bin/rippra.dll` — the shared library
- `rippra/bin/librippra.dll.a` — import library for linking

### API Headers

A single header provides the full public API:

```c
#include "rippra/rippra_api.h"
```

Compile with `-DBUILD_RIPRA_DLL` when building the DLL, or link against the import library for consumers.

## Python Bindings

### Setup

```python
from bindings.rippra import Rippra
r = Rippra(dll_path="bin/rippra.dll")
```

### Requirements

- NumPy
- ctypes (stdlib)

### Test

```bash
cd rippra
python bindings/test_bindings.py
```

## ONNX Model Deployment

### Exported Models

| Model | File | Size | Input Shape | Output Shape |
|---|---|---|---|---|
| MLP | `onnx_models/wavefront_mlp.onnx` | 1170 KB | (1, 254) | (1, 20) |
| CNN | `onnx_models/wavefront_cnn.onnx` | 810 KB | (1, 1, 12, 12) | (1, 20) |
| LSTM | `onnx_models/wavefront_lstm.onnx` | 830 KB | (1, 10, 20) | (1, 20) |

### Usage with ONNX Runtime

```python
from bindings.onnx_inference import ONNXInference
import numpy as np

model = ONNXInference("onnx_models/wavefront_cnn.onnx")
displacements = np.random.randn(1, 254).astype(np.float32)
coeffs = model.predict(displacements)
```

### CPU/GPU Selection

ONNX Runtime automatically selects CUDA if available:

```python
model = ONNXInference("model.onnx", device="cuda")  # force GPU
model = ONNXInference("model.onnx", device="cpu")    # force CPU
```

## Docker

### Build

```bash
docker build -t ripra .
```

### Run

```bash
# CPU only
docker run --rm ripra

# With GPU
docker run --gpus all --rm ripra
```

The container:
1. Compiles the C library with OpenMP
2. Builds and runs all tests
3. Exports ONNX models
4. Runs validation checks

### Base Image

`nvidia/cuda:12.8.0-devel-ubuntu22.04` — includes CUDA Toolkit, GCC, CMake, Python 3.10.

## Performance Notes

- **Classical path:** 0.041 ms (modal) / ~0.9 ms (full pipeline) — suitable for real-time AO on CPU
- **ML path:** 0.67 ms (MLP) / 2.26 ms (CNN) on RTX 2050 — requires GPU for real-time
- **ONNX Runtime:** Comparable latency to PyTorch, no Python dependency in production
- All methods meet the 10 ms AO cycle budget

## Directory Structure for Deployment

```
deploy/
├── rippra.dll           # C shared library
├── rippra_api.h         # Public API header
├── rippra.py            # Python bindings
├── onnx_inference.py    # ONNX runtime wrapper
├── wavefront_mlp.onnx   # ML model
├── wavefront_cnn.onnx
├── wavefront_lstm.onnx
├── config/
│   └── system.conf
└── README.md
```
