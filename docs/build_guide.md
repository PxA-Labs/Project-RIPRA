# Build & Usage Guide

## Requirements

### Windows
- MinGW GCC (tested with 6.3.0, 32-bit)
- CUDA Toolkit 12.x (optional, for GPU kernels)
- Python 3.10+ (for ML and visualization)
- PyTorch 2.x (for ML inference/training)
- ONNX Runtime (for model deployment)

### Linux
- GCC 9+
- CUDA Toolkit 12.x
- CMake 3.20+
- Python 3.10+

## Build

### C Library

#### Windows (Batch scripts)
```bash
cd rippra

# Static library (all tests)
build.bat

# Shared library (DLL for Python bindings)
build_dll.bat

# With OpenMP
build_openmp.bat
```
Output: `rippra/bin/librippra.a` (static) or `rippra/bin/rippra.dll` (DLL).

#### Linux / macOS (GCC manual build)
```bash
cd rippra
mkdir -p build

# Compile objects
gcc -O2 -fopenmp -c src/io.c -o build/io.o -Iinclude
gcc -O2 -fopenmp -c src/la.c -o build/la.o -Iinclude
gcc -O2 -fopenmp -c src/centroid.c -o build/centroid.o -Iinclude
gcc -O2 -fopenmp -c src/recon.c -o build/recon.o -Iinclude
gcc -O2 -fopenmp -c src/rippra_api.c -o build/rippra_api.o -Iinclude

# Link static archive
ar rcs build/librippra.a build/io.o build/la.o build/centroid.o build/recon.o build/rippra_api.o
```
Output: `rippra/build/librippra.a` (static library).

#### Cross-Platform CMake Build (Recommended)
From the repository root:
```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```
Outputs (shared library and test executables) will be located in the build directory.

### Tests

```bash
build_test_pipeline.bat   # Full integration test
run_test_pipeline.bat     # Run 35 tests

build_test_recon.bat      # Reconstruction tests
run_test_recon.bat

build_test_stream.bat     # Streaming pipeline test
run_test_stream.bat
```

### CUDA

```bash
# Linux
bash build_cuda.sh

# Windows
build_cuda_test.bat
```

### Docker

```bash
docker build -t ripra .
docker run --gpus all ripra
```

## Run

### Unified ML & C Reproducibility Suite
Verify the entire end-to-end WFS calibration, dataset generation, model training, and simulation checks in a single script:
```bash
python tools/reproduce_all.py
```

### Full Pipeline

```bash
cd rippra
run_test_pipeline.bat
```

Expected output: 35/35 tests passed. Pipeline latency ~0.76 ms (761 µs).

### Real-Time Time-Series Generation

```bash
python tools/generate_realistic_ts.py --frames 500
```

Output: `results/time_series.csv`, `results/zernike_time_series.csv`

### Visualizations

```bash
cd rippra/viz
python dashboard.py
```

Output: all plots in `visualizations/`.

### Pipeline Dashboard

```bash
python viz/pipeline_dashboard.py
python viz/generate_3d_animation.py
```

Open `visualizations/pipeline_dashboard.html` in a browser.

### ML Evaluation

```bash
cd rippra/ml
python evaluate_inference.py       # Classical vs ML comparison
python baseline_comparison.py      # Phase 8.1
python noise_robustness.py         # Phase 8.2
python ablation_study.py           # Phase 8.3
python performance_profile.py      # Phase 8.4
```

### ONNX Export

```bash
cd rippra/ml
python export_onnx.py
```

Output: `rippra/onnx_models/wavefront_mlp.onnx`, `wavefront_cnn.onnx`, `wavefront_lstm.onnx`.

## Configuration

All system parameters in `rippra/config/system.conf`:

| Parameter | Description | Typical Value |
|---|---|---|
| `camera_pixsize` | Pixel size (m) | 7.4e-6 |
| `frame_width` | Frame width (px) | 648 |
| `frame_height` | Frame height (px) | 492 |
| `flength` | Lenslet focal length (m) | 18e-3 |
| `pitch` | Lenslet pitch (m) | 300e-6 |
| `pupil_radius` | Pupil radius (m) | 2e-3 |
| `wavelength` | Operating wavelength (m) | 632.8e-9 |
| `zernike_nmax` | Max Zernike radial order | 5 |
| `dm_nact_x` | DM actuators across X | 8 |
| `dm_nact_y` | DM actuators across Y | 8 |
| `coupling` | Inter-actuator coupling | 0.2 |

### Overriding for Real Datasets

Before running on real ISRO (or other) hardware data, you **must** update:

1. **`rippra/config/system.conf`** — the C pipeline's authoritative config.  
   Override every parameter to match your sensor/lenslet/DM datasheet.

2. **`config/default.yaml`** — Python/ML template config.  
   Each parameter is annotated with `[PS9]` or `[PLACEHOLDER]` provenance.  
   Replace all `[PLACEHOLDER]` values with real measurements.

Key parameters to verify for real data:
- `camera_pixsize` / `pixel_size` — from the camera sensor datasheet
- `frame_width`, `frame_height` / `resolution` — sensor ROI or full-frame size
- `pitch` — MLA lenslet pitch (from manufacturer or calibration)
- `flength` / `focal_length` — MLA focal length
- `pupil_radius` — diameter of the telescope/beam at the MLA plane
- `wavelength` — centre wavelength of the source or filter
- `dm_nact_x`, `dm_nact_y` / `n_actuators` — DM actuator grid dimensions
- `coupling` — inter-actuator influence, measured from DM surface metrology

## Data Files

| File | Size | Description |
|---|---|---|
| `data_raw/sh_flat.raw` | 2.5 MB | Reference flat-field frame |
| `data_raw/sh_flat_bg.raw` | 2.5 MB | Background frame |
| `data_raw/img.raw` | 2.5 MB | Aberrated frame |
| `results/time_series.csv` | 1.6 MB | 500-frame time series (generated) |
| `results/reference_centroids_c.csv` | 4.8 KB | Reference spot coordinates |
| `results/spot_deviations_c.csv` | 2.9 KB | Spot deviations from reference |
