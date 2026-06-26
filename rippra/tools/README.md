# Tools — `rippra/tools/`

Dataset generation and data conversion utilities.

## Files

| File | Purpose |
|---|---|
| `generate_dataset.py` | Synthetic Kolmogorov turbulence dataset for ML training |
| `generate_realistic_ts.py` | Physically-realistic time-series from real centroid measurements |
| `npy_to_raw.py` | Convert .npy files to .raw binary format |

## generate_dataset.py

Generates training data with:
- Kolmogorov spatial correlations (Noll covariance matrix)
- Taylor frozen-flow temporal evolution (AR(1) process)
- Configurable D/r₀, τ₀, and noise levels
- Output: `.npz` with displacements, coefficients, metadata

```bash
python generate_dataset.py --samples 10000 --noise 0.1 --out data_ai/dataset.npz
```

## generate_realistic_ts.py

Generates time-series from real measurements:
- Real Zernike coefficients from `results/spot_deviations_c.csv`
- AR(1) evolution with correct Noll covariance
- Per-frame r₀ and τ₀ estimates
- Output: `results/time_series.csv` (500 frames, 20 modes, 127 spots)

```bash
python generate_realistic_ts.py --frames 500
```
