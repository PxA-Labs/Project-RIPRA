# AGENTS.md — Project Memory

## CI Pipeline
- **Multiple jobs** (Linux C, Windows C, Python, CUDA, benchmarks) — all green
- Linux/Windows: compile C library (`io`, `la`, `centroid`, `recon`, `rippra_api`, `simd`), build tests
- Python: test_onnx_models.py, predictive_ao.py, test_split_leakage.py (skips torch if unavailable)
- Synthetic data generated in CI before C tests (via `synthetic_shwfs.generate_test_data`)
- CI does NOT commit synthetic data — regenerates every run

## Temporal Leakage Fix (2026-07-08)
- `train_sequence.py` and `evaluate_sequence.py` previously used `random_split` on flat sample list → adjacent sliding windows leaked across train/val/test
- Fix: `SHSequenceDataset.split_by_sequence()` splits at the *sequence* level (contiguous blocks), then `check_split_leakage()` asserts no sequence ID appears in >1 split
- `test_split_leakage.py` programmatically verifies the invariant
- Documented in class docstring and split method

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

## Changelog Auto-Update
- `.github/workflows/auto-changelog.yml` runs on every PR merge to main
- Parses conventional-commit prefix to categorize entry in Keep-a-Changelog format
- `.github/scripts/update_changelog.py` handles the insertion logic

## Labels Taxonomy
- Defined in `docs/labels.md`: three-tier system (category, cross-cutting, tech-specific)
- Created via `gh label create` for any missing labels (e.g. `critical`)

## Performance Documentation
- `docs/performance.md` documents OpenMP scaling behavior (centroiding benefits most, diminishing returns after 4 threads on 2-vCPU runner)
- `tools/benchmark_scaling.sh` runs benchmark_openmp at 1/2/4/8 threads
- CI benchmarks job now runs OpenMP scaling across thread counts

## Added Visualizations
- `rippra/viz/dm_heatmap.py` — DM actuator stroke heatmap over actuator grid (node_u/node_v)
- Integrated into `dashboard.py` render_all() and HTML index
- Saved as `visualizations/dm_actuator_heatmap.png`

## C Pipeline with Synthetic Data
- 38/38 tests pass with totlenses=140 (including doc_example)
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

## TCoG SIMD Coverage (2026-07-08)
- Main TCoG inner loop (`tcog_window_fast` → `rippra_simd_tcog_window_fast`) AVX2-vectorized in PR #76
- 4.30× speedup on Linux GCC (AVX2 vs scalar); disabled on Windows (MSVC/MinGW) due to codegen issues
- `simd_avx2.c` uses `_mm256_loadu_pd`, `_mm256_cmp_pd`/`_mm256_blendv_pd` for thresholding, `_mm256_fmadd_pd` for weighted sums
- Horizontal reductions (`hmin256_pd`, `hmax256_pd`, `hsum256_pd`) via extract128 + unpackhi pattern
- Refined centroiding path (`rippa_compute_centroids_refined`) was still scalar — PR #81 replaced manual minmax+tcog_window with `tcog_window_fast()` call, completing SIMD coverage for both fast and refined paths
- Calibration path (`rippa_calibrate_grid`) remains scalar — called once, not performance-critical
- Tests: `test_simd.c` verifies bit-exact match between scalar and AVX2 paths; `benchmark_simd.c` measures speedup

## Build System
- CI builds use gcc, output to `build/` directory
- `BUILD_RIPRA_DLL` NOT defined for CI builds (static linking)
- Local build scripts in `rippra/` match CI structure
- `build_test_pipeline.bat` updated to include `rippra_api.c` (was missing)

## CUDA CI Build (2026-07-08)
- `test_cuda.c` compiled with nvcc via CMake (`.c` file marked LANGUAGE CUDA)
- nvcc (C++ mode) rejects `goto` crossing variable initializations — fixed by splitting cleanup into `cleanup:` and `gpu_cleanup:` labels
- GPU skip path uses early return (not `goto`) to avoid crossing GPU alloc declarations
- Loop variables declared at function scope to avoid C++ scoping issues
- `.cu` kernel files added to `ripra` library target so `test_cuda` can link them
- Missing function declarations added to `rippra_cuda.h`
- CI runs `ctest -R test_cuda` with `continue-on-error: true`

## Known Issues
- Fried geometry zonal reconstruction can be ill-conditioned for synthetic data
- Zonal phase values are physically small (~2e-6 rad PV) because slope→phase integration uses geometric coupling
- Modal reconstruction coefficients (~0.04 rad max) correctly represent the wavefront
