# Project RIPRA (ऋप्र)

<p align="center">
  <img src="./visualizations/logo.png" width="300" alt="Project RIPRA Logo"/>
</p>

Developing and optimizing algorithms for **Wavefront Reconstruction** and **Turbulence Characterization** using Shack-Hartmann Wavefront Sensor (SH-WFS) time-series data. Target: **ISRO BAH 2026 Problem Statement 9**.

## Overview
Project RIPRA processes high-speed SH-WFS frames to perform real-time Adaptive Optics (AO) correction. The system estimates turbulence parameters ($r_0$, $\tau_0$) and computes Deformable Mirror (DM) actuator commands in under 10 ms.

## Architecture
The pipeline is a C/CUDA + ONNX hybrid, optimizing for real-time inference and classical linear algebra (OpenBLAS).

```text
[ BMP Frames ] --> [ TCoG Centroiding ] --> [ Displacement Calculation ] --> [ Zonal/Modal Recon ]
                                                                                   |
                                                                       [ Actuator Mapping ] --> [ DM Control ]
```

## Installation & Quick Start
Dependencies: `cmake`, `make`, `gcc`/`clang`, `OpenMP`, `OpenBLAS`. Optional: `CUDA Toolkit`.

```bash
# 1. Clone repository
git clone https://github.com/PxA-Labs/Project-RIPRA.git
cd Project-RIPRA

# 2. Build via CMake
mkdir build && cd build
cmake .. && make

# 3. Run Tests
ctest

# 4. Execute reconstruction (synthetic data)
./test_recon
```

## Key Equations

### Thresholded Centre-of-Gravity (TCoG)
Computes the sub-aperture spot centroid:
$$ \bar{x} = \frac{\sum_{i} x_i I_i \mathcal{T}_i}{\sum_{i} I_i \mathcal{T}_i} $$
where $\mathcal{T}_i$ is the threshold mask.

### Least-Squares Reconstruction
$$ \boldsymbol{\phi} = \mathbf{D}^+ \mathbf{s} $$
where $\mathbf{D}^+$ is the Moore-Penrose pseudo-inverse of the geometry matrix.

### Turbulence Characterization
Fried parameter $r_0$ is derived from slope variance:
$$ \sigma_s^2 = 0.358 \left(\frac{d}{r_0}\right)^{5/3} \frac{\lambda^2}{4\pi^2 d^2} $$

Coherence time $\tau_0$:
$$ \tau_0 = 0.314\,\frac{r_0}{v_\perp} $$

## Benchmarks (Target)
| Stage | CPU (naive) | GPU (CUDA) | Budget |
|---|---|---|---|
| I/O + Decode | 2-5 ms | 0.1 ms | < 1 ms |
| Centroiding | 0.8 ms | 0.05 ms | < 0.5 ms |
| Wavefront Recon (MVM) | 3-8 ms | 0.1 ms | < 2 ms |
| **Total** | **~20 ms** | **< 0.5 ms** | **< 10 ms** |

## Outputs
*(Output visualizations like Zernike bar plots and wavefront phase maps will be published here upon dataset testing)*

## References
1. Hardy, J. W. (1998). *Adaptive Optics for Astronomical Telescopes*. Oxford University Press.
2. Noll, R. J. (1976). Zernike polynomials and atmospheric turbulence. *JOSA*, 66(3), 207-211.
3. Fried, D. L. (1966). Optical resolution through a randomly inhomogeneous medium. *JOSA*, 56(10), 1372-1379.
4. Southwell, W. H. (1980). Wave-front estimation from wave-front slope measurements. *JOSA*, 70(8), 998-1006.
5. Roddier, F. (1999). *Adaptive Optics in Astronomy*. Cambridge University Press.
