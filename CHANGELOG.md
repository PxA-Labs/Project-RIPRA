# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Standard GitHub issue templates for bug reports, feature requests, and pull requests.
- Root-level `.editorconfig` to enforce consistent file indentation, endings, and formats across development environments.
- Directory hierarchy documentation (`config/README.md`) explaining configuration roles.
- Comprehensive README with architecture diagrams and mathematical background.
- CMake build system to replace hardcoded Dockerfile gcc invocations.
- Default configuration YAML file (`config/default.yaml`).
- TCoG zero-denominator guards and NaN propagation masks in `centroid.c` to prevent silent corruption.
- Apache 2.0 License.
- SECURITY.md security policy.

### Changed
- Reorganized repository structure to improve hygiene: moved `Audit.tex` and `RIPRA_Audit_and_Issue_Backlog.md` into `docs/audit/`, moved all ML visualizations to `visualizations/ml/`, and consolidated simulation plots into `visualizations/simulation/`.
- Deleted deprecated `rippra/scratch/` files from tracking.
- Set explicit deprecation notice on stale audit report `docs/RIPRA_Audit_Report_BAH2026.md`.
- Replaced missing configuration imports in Docker build with actual configuration file.
- Updated project license from MIT to Apache 2.0.

### Fixed
- Fixed integer overflow vulnerability by ensuring centroid accumulators utilize double precision.
