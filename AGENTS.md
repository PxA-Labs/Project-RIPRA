# AGENTS.md — Project Memory

## CI Pipeline
- **3 jobs** (Linux C, Windows C, Python) — all green
- Linux/Windows: compile C library (`io`, `la`, `centroid`, `recon`, `rippra_api`), build tests
- Python: test_onnx_models.py, predictive_ao.py (skips torch if unavailable)
- Synthetic data generated in CI before C tests (via `synthetic_shwfs.generate_test_data`)
- CI does NOT commit synthetic data — regenerates every run

## Critical Bug Fix (2026-06-27)
- `rippra_zonal_reconstruct` and `rippra_modal_reconstruct` used `cfg->totlenses` instead of actual detected spot count → out-of-bounds reads
- Fix: added `nspots` field to `rippra_zonal_mesh` and `rippra_modal_model` structs, stored during `_setup()`, used in `_reconstruct()`
- Without this fix, `rippra_zonal_reconstruct` trashes memory when `totlenses > cal->nspots` and misses data when `totlenses < cal->nspots`

## Synthetic SHWFS Data
- Module: `rippra/ml/synthetic_shwfs.py`
- Generates 137 spots on rectangular grid (40.5 px pitch) within circular pupil
- Flat frame: Gaussian PSF (sigma=1.5 px, amplitude=600, bg=20)
- Shift computation matches recon.c formula: `dx = a * dzdx * FL * WL / (2π * pupil_R * pixsize)`
- Expected max shifts: ~5.5 px for typical coefficients (~0.84 rad RMS)
- Notebook: `notebook/kaggle_synthetic_shwfs_generator.py` (Kaggle-compatible)

## C Pipeline with Synthetic Data
- 35/35 tests pass with totlenses=140
- Config defaults in io.c: default_config() sets totlenses=140, rippra_api.c matches
- Centroid detection: 137 spots detected, realistic shifts
- Authoritative configuration: The C pipeline compiles using physical settings defined in `rippra/config/system.conf` (e.g. wavelength = 632.8 nm, camera pixel size = 7.4 µm, totlenses = 140). High-level templates or python ML modules use `config/default.yaml`.

## Key Config Values (must match C defaults exactly)
- camera_pixsize = 7.4e-6 m
- frame_width = 648, frame_height = 492
- pitch = 300e-6 m, pitch_px = 40.5 px
- flength = 18e-3 m
- pupil_radius = 2e-3 m (diameter 4 mm)
- wavelength = 632.8e-9 m (HeNe)
- totlenses = 140 (must be >= 137 detected spots)

## Build System
- CI builds use gcc, output to `build/` directory
- `BUILD_RIPRA_DLL` NOT defined for CI builds (static linking)
- Local build scripts in `rippra/` match CI structure
- `build_test_pipeline.bat` updated to include `rippra_api.c` (was missing)

## Known Issues
- Fried geometry zonal reconstruction can be ill-conditioned for synthetic data
- Zonal phase values are physically small (~2e-6 rad PV) because slope→phase integration uses geometric coupling
- Modal reconstruction coefficients (~0.04 rad max) correctly represent the wavefront
