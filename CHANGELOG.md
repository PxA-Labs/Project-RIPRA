# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file is auto-updated by a GitHub Action on every merged PR.

## [Unreleased]

### Documentation
- add Known Assumptions and Simplifications section to README and paper ([#84](https://github.com/PxA-Labs/Project-RIPRA/pull/84))
- annotate config/default.yaml with parameter provenance and add override docs ([#83](https://github.com/PxA-Labs/Project-RIPRA/pull/83))
- fix SECURITY.md with verified contact and GitHub security advisories ([#82](https://github.com/PxA-Labs/Project-RIPRA/pull/82))
- add architecture choice justification (#37) ([#77](https://github.com/PxA-Labs/Project-RIPRA/pull/77))
- add architecture diagram, API worked example, and auto-changelog (closes #38, #39, #41) ([#75](https://github.com/PxA-Labs/Project-RIPRA/pull/75))
- Add architecture diagram (Mermaid data-flow) to README ([#38](https://github.com/PxA-Labs/Project-RIPRA/issues/38))
- Refresh CHANGELOG.md with backfilled entries; auto-update on PR merge ([#39](https://github.com/PxA-Labs/Project-RIPRA/issues/39))
- Add API reference worked example with compilable doc_example.c ([#41](https://github.com/PxA-Labs/Project-RIPRA/issues/41))


### Performance
- use SIMD-accelerated tcog_window_fast in refined centroiding path ([#81](https://github.com/PxA-Labs/Project-RIPRA/pull/81))
- add AVX2-accelerated TCoG centroiding with runtime dispatch (#49) ([#76](https://github.com/PxA-Labs/Project-RIPRA/pull/76))

### CI/CD
- enable CUDA test compilation in CI (#22) ([#80](https://github.com/PxA-Labs/Project-RIPRA/pull/80))
- add benchmark regression tracking with baseline comparison (#23) ([#79](https://github.com/PxA-Labs/Project-RIPRA/pull/79))
## [0.4.0] — 2026-07-07

### Added
- add temporal leakage audit for sequence model splits (#35) ([#78](https://github.com/PxA-Labs/Project-RIPRA/pull/78))
- Public C API (`rippra_api.h`) with opaque-handle interface
- Golden-value regression tests for r₀, τ₀, and zonal reconstruction ([#16](https://github.com/PxA-Labs/Project-RIPRA/issues/16))
- Closed-loop convergence test (`test_closed_loop.c`, [#24](https://github.com/PxA-Labs/Project-RIPRA/issues/24))
- Malformed-input fuzz tests for BMP/raw readers ([#26](https://github.com/PxA-Labs/Project-RIPRA/issues/26))
- CMake build system with `RIPRA_BUILD_SHARED`, `RIPRA_BUILD_TESTS`, `RIPRA_BUILD_BENCHMARKS` options
- CodeQL security analysis in CI
- Clang-format CI check (scoped to PR-changed files)
- Apache 2.0 License
- SECURITY.md security policy

### Changed
- Consolidated 9 build scripts into CMake; remaining 6 are thin CMake wrappers
- BLAS dependency removed; la.c vendored as dependency-free
- `.clang-format` added matching project style
- Reorganized repository: moved audit docs to `docs/audit/`, ML visualizations to `visualizations/ml/`, simulation plots to `visualizations/simulation/`
- Set explicit deprecation notice on stale audit report

### Fixed
- `rippra_zonal_reconstruct` / `rippra_modal_reconstruct` out-of-bounds read when `totlenses != nspots`
- Integer overflow vulnerability in centroid accumulators (double precision)
- TCoG zero-denominator guards and NaN propagation masks in `centroid.c`
- BMP config validation rejects mismatched dimensions and non-BMP files
- Removed `fprintf(stderr, "DEBUG ...")` from recon.c

## [0.3.0] — 2026-06-20

### Added
- CUDA kernel stubs for centroiding and reconstruction
- Python ML pipeline: synthetic SHWFS generator, ONNX model training/inference
- LSTM predictor for closed-loop lag compensation
- CNN reconstructor (4.6× accuracy gain over MLP baseline)
- Real-time processing performance benchmarks (hot-path: 761 µs)
- Docker image with CUDA, GCC, and full Python ML stack

### Changed
- Default configuration values tuned for HeNe wavelength (632.8 nm)
- Synthetic data model: 137 spots on rectangular grid, 40.5 px pitch

### Fixed
- Singular-value handling in modal reconstruction for ill-conditioned Zprime matrices
- GPU memory management in CUDA kernels

## [0.2.0] — 2026-06-10

### Added
- Fried-geometry zonal wavefront reconstruction
- Zernike modal reconstruction
- Centroiding via Thresholded Center of Gravity (TCoG)
- DM influence matrix computation and actuator mapping
- Closed-loop AO control (step and run modes)
- Turbulence characterization (r₀, τ₀)
- Strehl ratio via Marechal approximation
- System configuration parser (`system.conf`)
- BMP and raw frame I/O
- Bundled linear algebra (LU, pseudo-inverse via Jacobi SVD)

### Changed
- Standardized function naming: `rippra_` prefix across all public symbols

## [0.1.0] — 2026-05-29

### Added
- Initial prototype: basic wavefront sensor simulation
- Plot scripts and Jupyter notebooks for visualisation
- Project documentation and mathematical foundation

[Unreleased]: https://github.com/PxA-Labs/Project-RIPRA/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/PxA-Labs/Project-RIPRA/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/PxA-Labs/Project-RIPRA/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/PxA-Labs/Project-RIPRA/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/PxA-Labs/Project-RIPRA/releases/tag/v0.1.0
