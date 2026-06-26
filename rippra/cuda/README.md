# CUDA Acceleration — `rippra/cuda/`

GPU kernels for centroiding, matrix operations, and DM mapping.

## Files

| File | Purpose |
|---|---|
| `centroid_kernels.cu` | Parallel TCoG centroiding on GPU |
| `matrix_kernels.cu` | Matrix-vector multiply, transpose, SVD |
| `dm_kernels.cu` | DM actuator command computation |
| `rippra_cuda.h` | Header for CUDA pipeline entry points |

## Pipeline

```
Frame (GPU) → centroid_kernels → matrix_kernels → dm_kernels → Result (GPU)
```

## Requirements

- NVIDIA GPU with Compute Capability 6.0+
- CUDA Toolkit 12.x
- Tested on RTX 2050 (4.3 GB VRAM)

## Build

```bash
# Linux
nvcc -o ripra_cuda centroid_kernels.cu matrix_kernels.cu dm_kernels.cu -lcublas

# Windows (build_cuda_test.bat)
nvcc -o bin/test_cuda.exe tests/test_cuda.c cuda/centroid_kernels.cu ...
```

## ML Inference

ML models (MLP, CNN, LSTM) run on the same GPU via PyTorch CUDA:

| Model | GPU FPS | GPU Latency |
|---|---|---|
| MLP | 93,659 | 0.67 ms |
| CNN | 26,866 | 2.26 ms |
