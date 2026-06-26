# API Reference

## C API — `rippra/include/rippra/rippra_api.h`

The shared library exposes 10 functions. All functions return `int` (0 = success, nonzero = error) except where noted.

### Config

#### `int rippa_config_load(rippa_config *cfg, const char *path)`

Parse system configuration from file.

| Parameter | Description |
|---|---|
| `cfg` | Output config struct |
| `path` | Path to `system.conf` |

```c
rippa_config cfg;
rippa_config_load(&cfg, "config/system.conf");
```

### Calibration

#### `int rippa_calibrate(const double *flat, int w, int h, const rippa_config *cfg, rippra_calibration *cal)`

Detect reference spot grid from flat-field frame.

| Parameter | Description |
|---|---|
| `flat` | Flat-field frame (w×h doubles) |
| `w, h` | Frame dimensions |
| `cfg` | System config |
| `cal` | Output calibration (grid, reference centroids) |

### Centroiding

#### `int rippa_centroid(const double *img, int w, int h, const rippra_calibration *cal, const rippa_config *cfg, double *out_cx, double *out_cy)`

Compute spot centroids via TCoG.

| Parameter | Description |
|---|---|
| `img` | Aberrated frame (w×h doubles) |
| `w, h` | Frame dimensions |
| `cal` | Calibration data |
| `cfg` | System config |
| `out_cx, out_cy` | Output centroid coordinates (nspots doubles each) |

#### `int rippa_centroid_refined(const double *img, int w, int h, const rippra_calibration *cal, const rippa_config *cfg, double *out_cx, double *out_cy)`

Iterative refined centroiding (higher accuracy, ~1.5× slower).

### Deltas

#### `int rippa_compute_deltas(const double *cx, const double *cy, const rippra_calibration *cal, double *out_dx, double *out_dy)`

Compute spot deviations from reference positions.

### Reconstruction

#### `int rippa_reconstruct_zonal(const double *dx, const double *dy, const rippra_calibration *cal, const rippa_config *cfg, double *out_phase, rippra_zonal_mesh *mesh)`

Fried geometry zonal wavefront reconstruction.

| Parameter | Description |
|---|---|
| `dx, dy` | Spot displacements (nspots each) |
| `cal` | Calibration data |
| `cfg` | System config |
| `out_phase` | Output phase heights (nnodes) |
| `mesh` | Zonal mesh (pre-initialized via setup) |

#### `int rippa_reconstruct_modal(const double *dx, const double *dy, const rippra_modal_model *model, const rippa_config *cfg, double *out_coeffs)`

Zernike modal reconstruction.

| Parameter | Description |
|---|---|
| `dx, dy` | Spot displacements (nspots each) |
| `model` | Modal model (pre-initialized via setup) |
| `cfg` | System config |
| `out_coeffs` | Zernike coefficients in radians (nmodes) |

### Turbulence

#### `double rippa_compute_r0(const double *dx_series, const double *dy_series, int nframes, int nspots, const rippa_config *cfg)`

Fried parameter from time-series displacement variance.

**Returns:** r₀ in meters.

#### `double rippa_compute_tau0(const double *dx_series, const double *dy_series, int nframes, int nspots, double frame_rate)`

Coherence time from temporal auto-covariance 1/e decay.

**Returns:** τ₀ in seconds.

### DM Mapping

#### `int rippa_dm_map(const double *phase, int nnodes, const rippra_zonal_mesh *mesh, const rippa_config *cfg, double *out_commands)`

Compute DM actuator commands from reconstructed phase.

### Pipeline

#### `int rippa_process_frame(const double *img, int w, int h, rippa_result *result)`

End-to-end pipeline: centroid → deltas → zonal → modal → DM.

### Utility

#### `double rippa_wavefront_rms_lambda(const double *phase, int nnodes)`

Wavefront RMS in units of wavelength (λ).

---

## Python Bindings — `rippra/bindings/rippra.py`

```python
from rippra import Rippra

r = Rippra(dll_path="bin/rippra.dll")
```

### Methods

| Method | Returns | Description |
|---|---|---|
| `load_config(path)` | `None` | Parse system configuration |
| `calibrate(flat, w, h)` | `dict` | Detect reference spot grid |
| `centroid(img, w, h)` | `tuple(cx, cy)` | Compute spot centroids |
| `centroid_refined(img, w, h)` | `tuple(cx, cy)` | Iterative refined centroids |
| `compute_deltas(cx, cy)` | `tuple(dx, dy)` | Spot deviations |
| `reconstruct_zonal(dx, dy)` | `tuple(phase, nnodes)` | Zonal reconstruction |
| `reconstruct_modal(dx, dy)` | `ndarray coeffs` | Zernike coefficients |
| `compute_r0(dx_series, dy_series)` | `float r0` | Fried parameter |
| `compute_tau0(dx_series, dy_series, fps)` | `float tau0` | Coherence time |
| `dm_map(phase)` | `ndarray commands` | DM actuator commands |
| `process_frame(frame)` | `dict result` | Full pipeline |
| `wavefront_rms_lambda()` | `float rms` | Wavefront RMS in λ |

### ONNX Inference

```python
from onnx_inference import ONNXInference

model = ONNXInference("onnx_models/wavefront_cnn.onnx")
coeffs = model.predict(displacements)  # shape: (1, nmodes)
```

## Error Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| -1 | General error |
| -2 | File not found |
| -3 | Invalid config |
| -4 | Calibration failed |
| -5 | Reconstruction failed |
