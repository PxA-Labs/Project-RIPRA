# Project RIPRA — Documentation

Welcome to the documentation directory for **Project RIPRA** (Wavefront Reconstruction & Turbulence Characterization).

## Document Index

### Technical Reference
* [Mathematical Foundation](./mathematical_foundation.md) — Centroiding algorithms, wavefront slope estimation, and Zernike polynomials.
* [Wavefront Reconstruction](./wavefront_reconstruction.md) — Zonal (Fried geometry) and modal reconstruction techniques.
* [Turbulence Characterization](./turbulence_characterization.md) — Fried parameter ($r_0$) and coherence time ($\tau_0$) derivations.
* [Deformable Mirror Mapping](./dm_mapping.md) — Actuator mapping, influence functions, and coupling compensation.

### API & Guides
* [API Reference](./api_reference.md) — Full C API and Python bindings reference with types and examples.
* [Build & Usage Guide](./build_guide.md) — Build instructions, configuration, data files, and how to run each component.
* [Deployment Guide](./deployment_guide.md) — DLL packaging, ONNX deployment, Python bindings, Docker.

### Project Management
* [Project Planning](./planning.md) — Timeline and checkpoints for Phase 0 to Phase 11.
* [Future Phases Roadmap](./future_phases.md) — Detailed objectives and completion status for Phases 1–11.
* [Problem Statement Alignment](./problem_statement_alignment.md) — Compliance matrix mapping C routines against requirements.

### Research & Publications
* [Research Material](./research-material.md) — Bibliography of SH-WFS and Adaptive Optics references.
* [Technical Paper (LaTeX)](./paper/rippra_paper.tex) — IEEE-formatted academic paper.
* [Presentation (HTML)](./paper/presentation.html) — Interactive slide deck with keyboard navigation.
* [Presentation (Markdown)](./paper/presentation.md) — Plain-text version of the slide deck.

### Source Module READMEs
* [C Library](../rippra/src/README.md) — Core pipeline: centroid, recon, turbulence, DM mapping.
* [CUDA Kernels](../rippra/cuda/README.md) — GPU acceleration for centroiding and matrix ops.
* [ML Models](../rippra/ml/README.md) — PyTorch training, evaluation, ONNX export.
* [Python Bindings](../rippra/bindings/README.md) — ctypes interface to the C library.
* [Visualization](../rippra/viz/README.md) — matplotlib dashboards and HTML pipeline monitor.
* [Tools](../rippra/tools/README.md) — Dataset generation and data conversion utilities.
* [Tests](../rippra/tests/README.md) — C test programs for all pipeline stages.
