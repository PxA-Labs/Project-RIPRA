# Project RIPRA (ऋप्र) — Project Overview

**Real-time Image Processing for Reconstruction of the Aperture** — Algorithms for
Wavefront Reconstruction, Turbulence Characterization, and Deformable Mirror
Actuator Mapping using Shack–Hartmann Wavefront Sensor (SH-WFS) time-series data.

Repository: [PxA-Labs/Project-RIPRA](https://github.com/PxA-Labs/Project-RIPRA)

---

## 1. Background

Atmospheric turbulence distorts an otherwise plane-parallel wavefront as it
propagates from a distant source (e.g., a star or satellite) down to a
ground-based telescope. In an adaptive optics (AO) system, this distortion is
measured and corrected in real time so that the final image remains sharp.

A **Shack–Hartmann Wavefront Sensor (SH-WFS)** samples the distorted wavefront
using an array of small lenslets (a Microlens Array, or MLA). Each lenslet
focuses a small patch of the incoming wavefront onto a camera detector,
producing a grid of spots. Local tilts in the wavefront cause each spot to
shift from its calibrated reference position — measuring these shifts allows
the original wavefront shape to be reconstructed.

The reconstructed wavefront (or its conjugate) is then converted into a
command map that drives the actuators of a **Deformable Mirror (DM)**,
physically correcting the distortion before it degrades the final image.

## 2. Problem Statement

Project RIPRA addresses the full processing pipeline required to turn raw
SH-WFS camera frames into actionable DM commands, fast enough to keep pace
with the atmosphere:

- **Process SH-WFS frames** collected during turbulence simulated in the
  laboratory.
- **Develop fast image-processing algorithms** for wavefront reconstruction,
  turbulence characterization, and actuator map determination.
- **Optimize for real-time execution** — atmospheric turbulence changes on a
  millisecond timescale, so the full reconstruction pipeline must run in
  under ~10 ms per frame.
- **Derive statistical turbulence parameters** — specifically the Fried
  parameter (r₀) and the coherence time (τ₀) — from the same time-series
  data.

## 3. Expected Outcomes

1. **Reconstructed wavefront phase maps**, W(xᵢ, yᵢ), for each SH-WFS frame.
2. **Turbulence characterization**, expressed as:
   - Fried parameter (r₀)
   - Coherence time (τ₀)
3. **Deformable mirror actuator maps**, A(xᵢ, yᵢ), for each reconstructed
   wavefront, including:
   - Actuator stroke-length mapping
   - Inter-actuator coupling compensation

## 4. Processing Pipeline

| Stage | Description |
|---|---|
| 1. Centroid detection | Locate the centroid of each sub-aperture spot in the WFS frame using a robust centroiding method (e.g., thresholded center-of-gravity). |
| 2. Displacement calculation | Compute each spot's deviation from its calibrated reference position. |
| 3. Wavefront reconstruction | Reconstruct the phase map using zonal or modal techniques, with the MLA and DM actuator grids arranged in a Fried geometry. |
| 4. Turbulence characterization | Derive r₀ and τ₀ from the reconstructed maps or Zernike coefficients. |
| 5. Actuator mapping | Convert the conjugate wavefront into DM actuator commands, accounting for inter-actuator mechanical coupling. |

## 5. Data Requirements

- **Time-series of SH-WFS frames** — sequential `.bmp` images captured at
  millisecond intervals by a science-grade camera.
- **Frame metadata** — pixel size and frame resolution.
- **MLA parameters** — lenslet pitch, lenslet count, and focal length.
- **Pupil diameter** — size of the turbulated beam.
- **DM parameters** — actuator grid geometry and inter-actuator coupling
  characteristics.

## 6. Technology Stack

- **Core language**: C, for the computational efficiency needed for
  sub-10-ms real-time correction.
- **Supporting languages**: Python and Jupyter Notebooks for prototyping,
  analysis, and visualization; CUDA for GPU-accelerated components.
- **Methods**: Zonal and modal reconstruction using orthogonal (Zernike)
  polynomials, or direct integration methods.
- **Libraries**: Linear algebra and optimization libraries as needed for the
  underlying reconstruction math.

## 7. Evaluation Criteria

- **Accuracy** — faithful reconstruction of wavefront phase maps consistent
  with the simulated turbulence.
- **Validation** — correct estimation of the physical/statistical turbulence
  parameters (r₀, τ₀).
- **Performance** — real-time suitability, i.e., total processing latency
  well under the atmospheric coherence timescale.

## 8. Repository Structure

```
Project-RIPRA/
├── .github/workflows/   # CI configuration
├── docs/                 # Documentation
├── notebook/             # Jupyter notebooks (analysis, prototyping)
├── rippra/               # Core library / processing code
├── visualizations/       # Reference images, logos, schematics
├── Dockerfile            # Containerized build/runtime
└── README.md
```

## 9. Context

Project RIPRA was developed as a submission to **Problem Statement 9** of the
**Bharatiya Antariksh Hackathon (BAH) 2026**, targeting real-time wavefront
reconstruction for ground-based adaptive optics systems. The associated
research has been written up in an IEEE-conference-style paper with an
accompanying Zenodo archive for reproducibility.

## 10. License & Contribution

See the repository's `CODE_OF_CONDUCT.md` for community guidelines.
Contributions, issues, and pull requests are welcome via GitHub.
*(Note: a `LICENSE` file has not yet been added to the repository — add one
to clarify usage and contribution terms.)*
