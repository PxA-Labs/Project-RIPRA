# Python Bindings — `rippra/bindings/`

ctypes interface to the C library (`rippra.dll` / `librippra.so`).

## Files

| File | Purpose |
|---|---|
| `rippra.py` | Main binding — wraps the C API with NumPy integration and DM safety checks |
| `onnx_inference.py` | ONNX Runtime wrapper for exported models |
| `test_bindings.py` | Self-documenting test for every API function |
| `../tests/test_dm_safety_bindings.py` | DM safety integration tests |

## API Surface

| Function | Description |
|---|---|
| `Rippra.load_config(path)` | Parse system configuration |
| `Rippra.calibrate(flat, w, h)` | Detect reference spot grid |
| `Rippra.centroid(img, w, h)` | Compute spot centroids via TCoG |
| `Rippra.reconstruct_zonal(dx, dy)` | Fried geometry phase reconstruction |
| `Rippra.reconstruct_modal(dx, dy)` | Zernike coefficient estimation |
| `Rippra.process_frame(frame)` | End-to-end: centroid → recon → DM |
| `Rippra.compute_r0(dx_series, dy_series)` | Fried parameter from time series |
| `Rippra.compute_tau0(dx_series, dy_series)` | Coherence time from auto-correlation |
| `Rippra.dm_map(phase)` | Actuator command inversion |
| `Rippra.wavefront_rms_lambda()` | Wavefront RMS in units of λ |

## DM safety API

The DM safety bindings require the version 3.x C ABI.  The Python binding
queries `rippra_version()` before installing the `RippraConfig` struct-return
signature, so a pre-3.0.0 library fails with a clear rebuild message instead
of being called with the incompatible configuration layout.

`Rippra()` starts with `rippra_default_config()`, so DM operations can be used
without first loading `config/system.conf`.  `load_config()` replaces that
configuration for subsequent operations.  The status constants are:

```python
from rippra.bindings.rippra import DM_OK, DM_SATURATED, DM_PARKED, DM_ERROR
```

| Method | Return value | Behavior |
|---|---|---|
| `dm_saturate(commands)` | `(commands, count, status)` | Copies the vector, applies the configured stroke limit, and reports the number of over-bound/nonfinite entries. `count == -1` and `status == DM_ERROR` indicate an invalid C argument or configuration. |
| `dm_map(phase, nnodes)` | `commands` | Preserves the existing ndarray return value while applying the configured DM safety layer. |
| `dm_map(phase, nnodes, return_status=True)` | `(commands, status)` | Uses the extended ABI mapping entry point and returns the safety status as well. |
| `dm_park(commands)` | `commands` | Returns a zeroed copy of the vector. |

DM vectors must be non-empty, one-dimensional, and numeric.  Inputs may be
read-only or strided: the binding makes an aligned, writable, C-contiguous
`float64` copy before calling C, and never mutates the caller's array.  A
finite `dm_max_stroke <= 0` disables stroke limiting.  Nonfinite commands are
parked even when limiting is disabled; nonfinite enabled safety configuration
is reported with `DM_ERROR` after parking.  The park threshold is clamped by
the C safety layer to `[0, 1]`.

## Usage

```python
from rippra import Rippra

r = Rippra()
r.load_config("config/system.conf")
r.calibrate(flat_frame, 648, 492)
result = r.process_frame(img_frame)
print("Phase RMSE:", result["rms"])
```

## ONNX Inference

```python
from onnx_inference import ONNXInference

model = ONNXInference("onnx_models/wavefront_cnn.onnx")
coeffs = model.predict(displacements)
```

## DM binding tests

The integration tests use the library named by `RIPRA_TEST_LIBRARY`.  They
skip when that variable is absent; when it is set, a load or ABI failure is a
test failure.

```bash
RIPRA_TEST_LIBRARY=/path/to/librippra.dylib \
  python -m unittest rippra.tests.test_dm_safety_bindings
```
