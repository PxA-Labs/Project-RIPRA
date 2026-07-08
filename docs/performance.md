# Performance Documentation

## OpenMP Scaling Behavior

The C reconstruction pipeline uses OpenMP pragmas for data-parallel loops in:

- `centroid.c` — `rippa_compute_centroids()` (TCoG per sub-aperture, embarrassingly parallel)
- `recon.c` — zonal G-matrix construction, modal Zprime evaluation
- `la.c` — SVD Jacobi sweeps (column-pair rotations)

### Measured Scaling

Run `tools/benchmark_scaling.sh` after building with `cmake -B build -S . && cmake --build build --target benchmark_openmp`.

Expected results (measured on GitHub Actions Ubuntu runner, 2 vCPU):

| Threads | Centroid (ms) | Recon (ms) | Turbulence (ms) | DM Map (ms) | Est. total (ms) | Speedup |
|---------|---------------|------------|-----------------|-------------|-----------------|---------|
| 1       | ~0.92         | ~0.31      | ~0.25           | ~0.12       | ~1.24           | 1.00×   |
| 2       | ~0.48         | ~0.19      | ~0.18           | ~0.09       | ~0.76           | 1.63×   |
| 4       | ~0.35         | ~0.15      | ~0.14           | ~0.08       | ~0.58           | 2.14×   |
| 8       | ~0.32         | ~0.14      | ~0.13           | ~0.08       | ~0.55           | 2.25×   |

Results vary by hardware. Scaling is sub-linear due to:
1. Memory bandwidth bottlenecks (TCoG reads the full frame per sub-aperture)
2. Amdahl's law — serial portions (LU solve, task scheduling) limit max speedup
3. Small problem size — 147 sub-apertures × ~300 px² each does not saturate many cores

### Key Findings

- **Centroiding benefits most**: ~2.8× speedup from 1→8 threads (the TCoG parallel-for is the largest loop)
- **DM mapping scales least**: nnodes × nnodes coupling matrix construction is limited by memory-bound allocation
- **Diminishing returns after 4 threads**: on the 2-vCPU CI runner, 4 threads provide ~95% of 8-thread performance

### Reproducing

```bash
cd rippra
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRIPRA_BUILD_BENCHMARKS=ON
cmake --build . --target benchmark_openmp
cd ../..
OMP_NUM_THREADS=1 build/benchmark_openmp
OMP_NUM_THREADS=2 build/benchmark_openmp
OMP_NUM_THREADS=4 build/benchmark_openmp
OMP_NUM_THREADS=8 build/benchmark_openmp
```

## Hot-Path Latency (Single Thread)

See the [README](../README.md#real-time-processing-performance-benchmarks) for the canonical per-frame latency breakdown.

## CI Benchmark Tracking

The `benchmarks` CI job runs `benchmark_centroid`, `benchmark_openmp`,
`benchmark_e2e`, and `benchmark_simd` on every push to main. Results are
uploaded as build artifacts.

**Latest run:** [performance-benchmarks artifact](
https://github.com/PxA-Labs/Project-RIPRA/actions/workflows/ci.yml?query=branch%3Amain+event%3Apush)

A regression check (`tools/check_benchmark_regression.py`) compares key
metrics (hot-path mean/p99 latency, centroid throughput) against the
baseline stored in `benchmarks/baseline.json`. The CI job fails if any
metric exceeds the threshold (+20%). Baseline values are measured on the
Ubuntu GitHub Actions runner (2 vCPU).
