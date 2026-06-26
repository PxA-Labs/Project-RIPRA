# Python Bindings — `rippra/bindings/`

ctypes interface to the C library (`rippra.dll` / `librippra.so`).

## Files

| File | Purpose |
|---|---|
| `rippra.py` | Main binding — wraps all 10 C API functions with NumPy integration |
| `onnx_inference.py` | ONNX Runtime wrapper for exported models |
| `test_bindings.py` | Self-documenting test for every API function |

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
