# Tests — `rippra/tests/`

C test programs for all pipeline stages.

## Files

| File | What It Tests |
|---|---|
| `test_io.c` | RAW frame loading, config parsing |
| `test_centroid.c` | Calibration, centroiding, delta computation, CSV export |
| `test_la.c` | Matrix ops, SVD, pseudo-inverse |
| `test_recon.c` | Zonal/modal reconstruction, turbulence, DM mapping |
| `test_full_pipeline.c` | End-to-end: config → load → calibrate → centroid → zonal → modal → turbulence → DM |
| `test_stream.c` | Real-time streaming pipeline (ping-pong, ring buffer) |
| `test_cuda.c` | CUDA kernel verification |
| `test_minimal.c` | Smoke test for library compilation |
| `benchmark_centroid.c` | Centroiding timing benchmarks |
| `benchmark_openmp.c` | OpenMP vs single-thread comparison |

## Key Tests

### Full Pipeline (23 tests)
```bash
build_test_pipeline.bat
run_test_pipeline.bat
```
Tests: config load, frame load, calibration, centroid (fast + refined), deltas, zonal recon, modal recon, turbulence (r₀ + τ₀), DM map.

### Reconstruction (14 tests)
```bash
build_test_recon.bat
run_test_recon.bat
```
Tests: zonal setup, zonal recon, modal setup, modal recon, r₀, τ₀, DM map, RMS stability.

### Stream
```bash
build_test_stream.bat
run_test_stream.bat
```
Tests: create/destroy stream, frame enqueue/dequeue, 500-frame throughput.

## Data Dependencies

Tests require raw frames in `rippra/data_raw/`:
- `sh_flat.raw` (reference flat-field)
- `img.raw` (aberrated frame)
