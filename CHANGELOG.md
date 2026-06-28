# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Comprehensive README with architecture diagrams and mathematical background.
- CMake build system to replace hardcoded Dockerfile gcc invocations.
- Default configuration YAML file (`config/default.yaml`).
- TCoG zero-denominator guards and NaN propagation masks in `centroid.c` to prevent silent corruption.
- MIT License.

### Changed
- Replaced missing configuration imports in Docker build with actual configuration file.

### Fixed
- Fixed integer overflow vulnerability by ensuring centroid accumulators utilize double precision.
