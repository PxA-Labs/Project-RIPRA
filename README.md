# Project RIPRA (ऋप्र)

<p align="center">
  <img src="./visualizations/logo.png" width="300" alt="Project RIPRA Logo"/>
</p>

**Real-time wavefront reconstruction, turbulence characterization, and deformable mirror control** using a Shack-Hartmann Wavefront Sensor (SH-WFS), with classical C reconstruction, GPU-accelerated pipelines, and deep learning inference.

[![Pipeline Dashboard](./visualizations/pipeline_dashboard.html)](visualizations/pipeline_dashboard.html)
[![Full Dashboard](./visualizations/index.html)](visualizations/index.html)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SH-WFS Camera (648×492 px)                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  sh_flat.raw (reference)          img.raw (aberrated)    │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 1: C Library (rippra/src/)  ───  < 0.9 ms/frame         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  centroid.c  →  io.c  →  la.c  →  recon.c  →  stream.c  │  │
│  │  TCoG centroid   raw I/O   linear alg  zonal/modal  ring │  │
│  │  (0.805 ms)                (SVD)      recon + DM    buf  │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  rippra_api.c  →  DLL/SO  →  Python bindings (ctypes)    │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 2: GPU Acceleration (rippra/cuda/)                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  centroid_kernels.cu   matrix_kernels.cu   dm_kernels.cu  │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│  Layer 3: Python ML & Visualization                             │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  ML Models        │  │  Dashboards   │  │  Tools           │  │
│  │  MLP / CNN / LSTM │  │  wavefront   │  │  dataset gen.    │  │
│  │  PyTorch + ONNX   │  │  Zernike     │  │  time-series     │  │
│  │  MSE 0.01056      │  │  turbulence  │  │  raw conversion  │  │
│  │  (CNN)            │  │  pipeline    │  │                  │  │
│  └──────────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

### Classical C Reconstruction
- **Centroiding**: Thresholded Center of Gravity (TCoG) — fast single-pass (0.805 ms/frame) and iterative refined (1.18 ms/frame)
- **Zonal Reconstruction**: Fried geometry wavefront nodes, truncated SVD pseudo-inverse
- **Modal Reconstruction**: Zernike derivative matrix via numerical area integration (15×15 quadrature), 20 modes (Noll j=2..21)
- **Turbulence Characterization**: Fried parameter r₀ from slope variance, coherence time τ₀ from temporal auto-covariance
- **Deformable Mirror Mapping**: Actuator command inversion with inter-actuator coupling compensation

### Real-Time Pipeline
- **Full pipeline latency**: ~0.9 ms/frame (fast centroid 0.805 ms + recon 0.05 ms + DM 0.05 ms)
- **Throughput**: ~1100 fps theoretical, exceeding 500 fps requirement
- **Stability**: σ < 0.00005λ on static frames (threshold 0.05λ)
- **Ping-pong buffers** + SPSC ring buffer for streaming acquisition
- **OpenMP parallelization** on centroiding, matrix ops, modal integration, r₀ computation

### GPU Acceleration
- CUDA C kernels for centroiding, matrix operations, DM mapping
- Full GPU pipeline (`rippra_cuda_full_pipeline`)
- ML inference on CUDA via PyTorch: MLP 93,659 fps, CNN 26,866 fps

### Machine Learning
| Model | Test MSE | Pearson r | Parameters | Latency (ms) |
|-------|----------|-----------|------------|-------------|
| Classical Modal | 0.0003 | 1.0000 | — | 0.04 |
| CNN | 0.01056 | 0.9907 | 206K | 2.26 |
| MLP | 0.752 | 0.9077 | 35K | 0.67 |
| LSTM (predict) | — | — | 211K | 0.62 |

- **CNN with ResNet-style spatial mapping**: Arranges irregular sub-aperture coordinates onto a dense 2D physical grid
- **LSTM sequence models**: 1/5/10 ms lookahead prediction, turbulence regime classification (99.64% accuracy), D/r₀ estimation (R²=0.693)
- **ONNX export**: All models exported to ONNX for deployment

### Visualization Dashboard
- **Pipeline Dashboard** (`pipeline_dashboard.html`): Three-node layout — SH-WFS input (animated spot traces) → processing pipeline → reconstructed wavefront (matplotlib-animated 3D surface)
- **Wavefront Visualizations**: 2D phase map, 3D surface with circular pupil (RdBu_r colormap), spot centroid offset vectors
- **Zernike Dashboard**: Modal weight bar chart, low-order time-series tracking (500 frames from real data)
- **Turbulence Analytics**: r₀/τ₀ telemetry readouts, D/r₀ regime classification (Weak/Moderate/Strong)
- **Performance Monitor**: 6-panel latency, FPS, CPU/GPU/memory metrics
- Dark-themed, self-contained HTML (no CDN dependencies)
- All visualizations rendered to `visualizations/` via `dashboard.py`

---

## Repository Structure

```
Project-RIPRA/
├── rippra/
│   ├── src/               C source (centroid.c, io.c, la.c, recon.c, stream.c, rippra_api.c)
│   ├── include/rippra/    C headers (centroid.h, io.h, la.h, recon.h, stream.h, rippra_api.h)
│   ├── tests/             C test programs (test_full_pipeline.c, test_recon.c, etc.)
│   ├── cuda/              CUDA kernels (centroid_kernels.cu, matrix_kernels.cu, dm_kernels.cu)
│   ├── ml/                PyTorch models, training, evaluation, ONNX export
│   ├── viz/               Python visualization dashboards
│   ├── tools/             Dataset generation utilities
│   ├── bindings/          Python ctypes bindings to the C library
│   ├── config/            System configuration files
│   ├── onnx_models/       Exported ONNX models (MLP, CNN, LSTM)
│   ├── build*.bat         Windows build scripts
│   └── build*.sh          Linux build scripts
├── docs/                  Documentation (math foundation, algorithms, roadmap)
├── visualizations/        Generated plots, HTML dashboards, animated GIFs
├── RAW_DATA/              Original experimental data + MATLAB reference toolbox
├── Dockerfile             CUDA 12.8 / Ubuntu 22.04 build container
└── README.md
```

---

## Build & Run

### Windows (MinGW)
```bash
# Build and run the full pipeline test
cd rippra
build_dll.bat                          # Build shared library
build_test_pipeline.bat                # Build integration test
run_test_pipeline.bat                  # Run 23 tests
```

### Linux (Docker)
```bash
docker build -t ripra .
docker run --gpus all ripra
```

### Python (visualizations, ML)
```bash
# Generate all visualizations
cd rippra/viz
python dashboard.py

# Generate pipeline dashboard
python pipeline_dashboard.py
python generate_3d_animation.py

# Generate realistic time-series (500 frames, Kolmogorov AR(1))
python tools/generate_realistic_ts.py --frames 500

# Run ML evaluation
cd rippra/ml
python evaluate_inference.py
python performance_profile.py
```

---

## Results

### Performance Benchmarking (500 iterations)

| Method | Mean (ms) | σ (ms) | P50 (ms) | P95 (ms) | Memory |
|--------|-----------|--------|----------|----------|--------|
| Modal | 0.041 | 0.012 | 0.037 | 0.059 | 0.08 MB |
| MLP | 0.666 | 0.332 | 0.529 | 1.386 | 1.14 MB |
| CNN | 2.257 | 1.159 | 2.091 | 3.587 | 0.79 MB |

### Noise Robustness
- **Gaussian noise** (σ 0.01→3.16 px): CNN robust (RMSE 0.106→0.107), Modal graceful (0.0003→0.011)
- **Photon shot noise** (γ 0.01→31.6): Modal 0.328→0.006, CNN 0.611→0.107
- **Spot occlusion** (0→50%): MLP most robust at high occlusion (RMSE 1.063 vs Modal 1.284)

### Pipeline Dashboard
Open `visualizations/pipeline_dashboard.html` in any browser for the interactive three-node pipeline view with animated wavefront reconstruction.

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/mathematical_foundation.md` | Zernike polynomials, Noll indexing, Fried geometry, modal reconstruction |
| `docs/wavefront_reconstruction.md` | Zonal and modal reconstruction algorithms |
| `docs/turbulence_characterization.md` | r₀, τ₀, Kolmogorov model, Noll covariance |
| `docs/dm_mapping.md` | DM actuator mapping with coupling compensation |
| `docs/future_phases.md` | Full roadmap and completion status (Phases 1-11) |

---

## Acknowledgments

- Original SH-WFS MATLAB reference toolbox: [mshwfs](https://github.com/skurow/mshwfs)
- Experimental SH-WFS data provided by laboratory turbulence simulations
- Built with MinGW GCC, CUDA Toolkit, PyTorch, ONNX Runtime

---

## License

MIT License — see [LICENSE](RAW_DATA/mshwfs-master/LICENSE.txt) for details.
