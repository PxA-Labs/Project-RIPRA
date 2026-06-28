# RIPRA Algorithms Handbook

This document describes every algorithm in the RIPRA adaptive optics (AO)
pipeline — from raw camera frames to closed-loop correction — with physics
motivation, key formulas, and links to C source lines.

---

## Table of Contents

1. [Image Acquisition & Preprocessing](#1-image-acquisition--preprocessing)
2. [Spot Detection](#2-spot-detection)
3. [Thresholded Center of Gravity (TCoG)](#3-thresholded-center-of-gravity-tcog)
4. [Grid Calibration](#4-grid-calibration)
5. [Zernike Polynomials](#5-zernike-polynomials)
6. [Modal Reconstruction (Zernike-based)](#6-modal-reconstruction-zernike-based)
7. [Zonal Reconstruction (Fried Geometry)](#7-zonal-reconstruction-fried-geometry)
8. [Fried Parameter r₀](#8-fried-parameter-r)
9. [Coherence Time τ₀](#9-coherence-time)
10. [DM Mapping](#10-dm-mapping)
11. [Closed-Loop AO Control](#11-closed-loop-ao-control)
12. [Linear Algebra Utilities](#12-linear-algebra-utilities)
13. [Synthetic Data Generation](#13-synthetic-data-generation)
14. [References](#14-references)

---

## 1. Image Acquisition & Preprocessing

### Frame Format

Raw Shack-Hartmann frames are stored as contiguous arrays of IEEE-754
doubles (8 bytes each) in row-major order:

```
data[ row * width + col ]   for row ∈ [0, height), col ∈ [0, width)
```

For the current system: 648 × 492 = 318,816 elements = 2,550,528 bytes.

### Configuration

System parameters are loaded from a key-value text file (`system.conf`)
into a `rippa_config` struct ([io.c:44–82](../../rippra/src/io.c)):

| Field | Typical value | Description |
|---|---|---|
| `camera_pixsize` | 7.4 × 10⁻⁶ m | Pixel pitch on the detector |
| `frame_width` | 648 px | Detector columns |
| `frame_height` | 492 px | Detector rows |
| `totlenses` | 140 | Maximum lenslet count (≥ detected) |
| `flength` | 18 × 10⁻³ m | Focal length of lenslets |
| `pitch` | 300 × 10⁻⁶ m | Physical lenslet pitch |
| `pupil_radius` | 2 × 10⁻³ m | Telescope pupil radius (4 mm diam.) |
| `wavelength` | 632.8 × 10⁻⁹ m | HeNe laser wavelength |
| `thresh_binary` | 0.08 | Relative threshold for binarization |
| `centroid_percent` | 0.2 | Relative threshold for TCoG |
| `zernike_nmax` | 5 | Max radial order (20 modes) |

---

## 2. Spot Detection

### 2.1 Binarization

Given the global minimum `fmin` and maximum `fmax` of the reference flat
frame, a binary threshold is computed:

```
level = fmin + thresh_binary · (fmax − fmin)
```

Every pixel `frame[i] ≥ level` is set to 1; all others to 0
([centroid.c:274–280](../../rippra/src/centroid.c)).

### 2.2 Connected-Component Labeling

A two-pass union-find algorithm with 4-connectivity assigns a unique label
to each connected region of foreground pixels
([centroid.c:40–104](../../rippra/src/centroid.c)).

- **First pass**: scan left-to-right, top-to-bottom. If the pixel above or to
  the left shares a label, propagate it. When both exist with different labels,
  record an equivalence (union).
- **Second pass**: resolve equivalence chains via path compression, then remap
  to a compact label sequence (1, 2, 3, …).

### 2.3 Spot Filtering

Components with area < 8 pixels are discarded (mirroring the MATLAB
original's `npixsmall = 8`). For each surviving component, the geometric
centroid is computed:

```
cx_k = (∑ x_i) / area_k      cy_k = (∑ y_i) / area_k
```

([centroid.c:106–182](../../rippra/src/centroid.c)).

---

## 3. Thresholded Center of Gravity (TCoG)

TCoG computes a sub-pixel spot centroid within a rectangular search window,
ignoring pixels below a local threshold.

### 3.1 Window-Level Threshold

For a window of size `W × H` pixels, the local minimum `wmin` and maximum
`wmax` are found. The CoG threshold is:

```
level = wmin + centroid_percent · (wmax − wmin)
```

where `centroid_percent = 0.2` (20% above background).

### 3.2 Weighted Centroid

The centroid is the intensity-weighted mean of all pixels above the
threshold:

```
cx = (∑ x_i · I_i) / ∑ I_i      cy = (∑ y_i · I_i) / ∑ I_i
```

where `I_i` is the pixel intensity and the sums run over pixels with
`I_i ≥ level`.

([centroid.c:188–259](../../rippra/src/centroid.c))

### 3.3 Fallback

If the total mass `∑ I_i = 0` (no pixel above threshold), the centroid
defaults to the geometric center of the window:

```
cx = (col_min + col_max) / 2      cy = (row_min + row_max) / 2
```

### Performance Consideration

The `tcog_window_fast()` variant finds min/max and computes TCoG in a single
pass, halving memory traffic. This is optimal for typical sub-aperture
windows of 5–7 px (≈3k pixels total) vs. scanning the full frame (≈300k
pixels).

---

## 4. Grid Calibration

Calibration is a one-time setup that maps detected spots to a regular
lenslet grid and records their reference positions.

### 4.1 Pupil Center Estimation

The pupil center is the centroid of all detected spot centres:

```
pupil_cx = (1 / N) · ∑ cx_k      pupil_cy = (1 / N) · ∑ cy_k
```

([centroid.c:290–295](../../rippra/src/centroid.c))

### 4.2 Pitch Estimation

The mean nearest-neighbour distance gives the lenslet pitch in pixels:

```
pitch_px = (1 / M) · ∑ min_{j ≠ i} √[(cx_i − cx_j)² + (cy_i − cy_j)²]
```

([centroid.c:299–313](../../rippra/src/centroid.c))

### 4.3 Search Window Size

The TCoG window radius is set to half the nearest-neighbour distance,
scaled for robustness:

```
window_radius = (pitch_px / 2) · (1 / √2)  ≈ 0.35 · pitch_px
```

([centroid.c:317–319](../../rippra/src/centroid.c))

### 4.4 Sub-Aperture Definition

For each detected spot, a square window `[col_min, col_max] × [row_min,
row_max]` is defined. Within this window, TCoG computes the precise
reference centroid `(ref_cx, ref_cy)` via `tcog_window()`. The result is
stored in the `rippra_subap` struct
([centroid.c:325–351](../../rippra/src/centroid.c)).

### 4.5 Per-Frame Centroiding

For subsequent (aberrated) frames, the same search windows are used. The
measured centroid `(cx, cy)` is compared to the reference:

```
dx = cx − ref_cx      dy = cy − ref_cy
```

([centroid.c:376–405](../../rippra/src/centroid.c))

### 4.6 Refined Centroiding (Optional)

A two-pass scheme re-centers the search window on the first-pass centroid:

```
r = pitch_px / (2·√2)              // refined window radius
first pass:  (cx₁, cy₁)            // original windows
second pass: window centred on (round(cx₁), round(cy₁)) with radius r
```

([centroid.c:407–449](../../rippra/src/centroid.c))

---

## 5. Zernike Polynomials

### 5.1 Noll Indexing

Zernike polynomials are ordered by Noll's sequential index `j`
(Noll 1976). The mapping `j → (n, m)` is:

```
j = 1:  (n=0, m=0)   [piston]
j = 2:  (n=1, m=1)   [tip]
j = 3:  (n=1, m=−1)  [tilt]
j = 4:  (n=2, m=0)   [defocus]
j = 5:  (n=2, m=−2)  [astigmatism]
…
```

Modes alternate `cos(mθ)` / `sin(mθ)` within each radial order, with
`m ≥ 0` first. See `noll_to_nm()` in
[recon.c:98–138](../../rippra/src/recon.c).

### 5.2 Radial Polynomial

The radial part `R_n^m(ρ)` is:

```
R_n^m(ρ) = Σ_{s=0}^{(n−|m|)/2}  C_s · ρ^{n−2s}

C_s = (−1)^s · (n−s)! / [ s! · ((n+|m|)/2 − s)! · ((n−|m|)/2 − s)! ]
```

([recon.c:30–36](../../rippra/src/recon.c))

### 5.3 Full Polynomial

The Zernike polynomial at normalised pupil coordinates `(x, y)` with
`ρ = √(x² + y²)`, `θ = atan2(y, x)`:

```
Z_n^m(x, y) = N_n^m · R_n^m(ρ) · cos(m·θ)    for m ≥ 0
Z_n^m(x, y) = N_n^m · R_n^m(ρ) · sin(|m|·θ)   for m < 0

N_n^m = √(n+1)          for m = 0
N_n^m = √(2·(n+1))      for m ≠ 0
```

### 5.4 Analytical Derivatives

The wavefront slope `(∂Z/∂x, ∂Z/∂y)` is computed analytically using the
chain rule in polar coordinates:

```
∂Z/∂x = (∂Z/∂ρ) · cos θ − (∂Z/∂θ) · sin θ / ρ
∂Z/∂y = (∂Z/∂ρ) · sin θ + (∂Z/∂θ) · cos θ / ρ
```

where:

```
∂Z/∂ρ = N · (∂R/∂ρ) · T(θ)
∂Z/∂θ = N · R · (∂T/∂θ)

with T(θ) = cos(|m|·θ) or sin(|m|·θ)
```

At the origin (`ρ < 10⁻⁹`), derivatives are set to 0 for all modes
except tip (n=1, m=1) and tilt (n=1, m=−1), where they equal the
normalisation factor.

See `evaluate_zernike_derivatives()` in
[recon.c:38–96](../../rippra/src/recon.c).

---

## 6. Modal Reconstruction (Zernike-based)

### 6.1 Physics Model

The spot displacement on the detector is related to the wavefront slope by:

```
ϕ(x, y) = Σ a_k · Z_k(x, y)               [wavefront in radians]
∂ϕ/∂x = Σ a_k · ∂Z_k/∂x                   [slope in rad / pupil unit]

Δx_px = (∂ϕ/∂x) · FL · λ / (2π · R_pupil · p)     [shift on detector]
```

where `FL` = lenslet focal length, `λ` = wavelength, `R_pupil` = pupil
radius, `p` = pixel size.

### 6.2 Zprime Matrix

The Zprime matrix maps Zernike coefficients `a_k` to spot displacements:

```
Δx_i = (1/f) · Σ a_m · Z'_ik
Δy_i = (1/f) · Σ a_m · Z'_{(i+N),k}
```

Each element `Z'_ik` is the average Zernike derivative over the sub-aperture
area, computed by numerical quadrature on a 15×15 grid within the
sub-aperture radius `rbar = sa_radius / pupil_radius`:

```
for each spot k:
    for each mode m:
        Z'[k][m] = (pupil_rad / (π · sa_rad²)) · ⟨∂Z_m/∂x⟩_subap
        Z'[k+N][m] = −(pupil_rad / (π · sa_rad²)) · ⟨∂Z_m/∂y⟩_subap
```

([recon.c:284–358](../../rippra/src/recon.c))

### 6.3 Pseudo-Inverse Solution

The coefficient vector `a` is obtained by least squares:

```
a_m = Z'⁺ · s
s = [Δx · p;  Δy · p]        // displacements in metres
a = (1/f) · a_m · (2π / λ)   // convert to radians
```

where `Z'⁺` is the Moore-Penrose pseudo-inverse of Zprime, computed via
one-sided Jacobi SVD ([la.c:128–276](../../rippra/src/la.c)).

### 6.4 Implementation

See `rippra_modal_reconstruct()` in
[recon.c:371–413](../../rippra/src/recon.c). Piston (j=1) is excluded —
only modes 2…21 are fitted.

---

## 7. Zonal Reconstruction (Fried Geometry)

### 7.1 Phase Nodes

Each sub-aperture square has four corner nodes, forming a rectangular grid
on the Fried geometry:

```
node (u,v)   node (u+1,v)
     ●━━━━━━━●
     ┃       ┃
     ┃ subap ┃
     ┃   k   ┃
     ●━━━━━━━●
node (u,v+1) node (u+1,v+1)
```

The integer grid coordinates `(u, v)` are:

```
u = round((ref_cx − pupil_cx) / pitch_px)
v = round((ref_cy − pupil_cy) / pitch_px)
```

([recon.c:160–161](../../rippra/src/recon.c))

### 7.2 Geometry Matrix G

The measured slopes `s = [dx · p/f;  dy · p/f]` (radians) relate to the
phase node heights `W` via:

```
s = G · W
```

For each sub-aperture `k` with nodes `(u,v)`, `(u+1,v)`, `(u,v+1)`,
`(u+1,v+1)`, the X-slope and Y-slope rows of G are:

```
G[k][node]          = [−1, +1, −1, +1] / (2·d)     [X slope row]
G[k + N][node]      = [−1, −1, +1, +1] / (2·d)     [Y slope row]
```

where `d = pitch` (metres). This is the standard Fried geometry finite
difference stencil
([recon.c:218–244](../../rippra/src/recon.c)).

### 7.3 Reconstruction

The phase is recovered from the pseudo-inverse:

```
W = G⁺ · s
```

`G⁺` is computed by `rippa_pinv()` with SVD truncation at `rcond = 10⁻⁴`
to suppress the piston mode (singular value = 0).

See `rippra_zonal_reconstruct()` at
[recon.c:261–279](../../rippra/src/recon.c).

---

## 8. Fried Parameter r₀

The Fried parameter `r₀` quantifies the integrated strength of atmospheric
turbulence. It is the aperture diameter over which the RMS wavefront error
is 1 radian.

### 8.1 Spot Displacement Variance

For a time series of `N` frames with `K` sub-apertures:

```
σ_x²(k) = 1/(N−1) · Σ_t [dx_k(t) − ⟨dx_k⟩]²
σ_y²(k) = 1/(N−1) · Σ_t [dy_k(t) − ⟨dy_k⟩]²

⟨σ²⟩ = (1/K) · Σ_k ½ · (σ_x²(k) + σ_y²(k))
```

Each variance is converted from pixels to radians squared:

```
σ_ϕ² = ⟨σ²⟩ · (p/f)²
```

([recon.c:486–522](../../rippra/src/recon.c))

### 8.2 r₀ Formula

Following the differential tilt variance method (Sasiela 1994):

```
r₀ = [ 0.170 · λ² · d^{−1/3} / σ_ϕ² ]^{3/5}
```

where `d = pitch` is the sub-aperture diameter and `λ` is the wavelength.
The constant `0.170` comes from the Kolmogorov turbulence spectrum
integrated over a square sub-aperture.

See `rippra_compute_r0_impl()` at
[recon.c:475–531](../../rippra/src/recon.c).

---

## 9. Coherence Time τ₀

The coherence time `τ₀` characterises the temporal evolution of turbulence.

### 9.1 Auto-Correlation

For time lags `ℓ = 0, 1, …, N/2`:

```
C(ℓ) = ⟨dx_k(t) · dx_k(t+ℓ) + dy_k(t) · dy_k(t+ℓ)⟩_{k,t}
```

where the average is over all sub-apertures and all frame pairs.

### 9.2 1/e Crossing

`τ₀` is the lag at which the normalised auto-correlation falls to `1/e`:

```
C(τ₀) / C(0) = 1/e    ⟹    τ₀ = lag_at_1/e / frame_rate
```

If the correlation does not cross `1/e`, `τ₀` defaults to `(N/2) /
frame_rate` (max measurable lag).

Linear interpolation between discrete lags gives a fractional-sample
estimate.

See `rippra_compute_tau0_impl()` at
[recon.c:533–592](../../rippra/src/recon.c).

---

## 10. DM Mapping

The deformable mirror (DM) has actuators at the same grid positions as the
zonal phase nodes. The coupling matrix `C` models the influence of each
actuator on its neighbours:

### 10.1 Coupling Matrix

```
C[i][i] = 1
C[i][j] = coupling            for `du=1, dv=0` or `du=0, dv=1`  (edge neighbours)
C[i][j] = coupling²           for `du=1, dv=1`                   (diagonal neighbours)
C[i][j] = 0                   otherwise
```

`coupling = 0.15` (15% influence on adjacent actuators).

([recon.c:596–623](../../rippra/src/recon.c))

### 10.2 Actuator Command Computation

Given a target phase `ϕ_target` to correct, solve:

```
C · v = −ϕ_target    ⟹    v = −C⁻¹ · ϕ_target
```

This is solved via Doolittle LU decomposition with partial pivoting
(`rippa_lusolve()`) at
[la.c:105–189](../../rippra/src/la.c).

The result `v` is the vector of actuator stroke commands.

### 10.3 DM Shape

The actual DM shape produced by commands `v` is:

```
DM_shape = C · v
```

The residual wavefront after DM correction is:

```
residual = ϕ_target + DM_shape = ϕ_target + C · v
```

At convergence (`v = −C⁻¹·ϕ_target`), `residual ≈ 0`.

See `rippra_dm_apply_impl()` at
[recon.c:633–664](../../rippra/src/recon.c).

---

## 11. Closed-Loop AO Control

### 11.1 Single Step

Each closed-loop iteration performs:

1. **Measure residual**: `residual = input + C · dm_commands`
2. **Compute DM delta**: `δ = gain · (−C⁻¹ · residual) = −gain · C⁻¹ · residual`
3. **Accumulate**: `dm_commands += δ`
4. **Return RMS of residual**: `RMS = √( (1/N)·∑ residual_i² )`

The RMS value is reported in microradians (×10⁶).

([recon.c:666–730](../../rippra/src/recon.c))

### 11.2 Full Run

The closed-loop run iterates until either:

- `RMS ≤ target_rms` (converged), or
- `max_iter` iterations reached (not converged).

The default gain is 0.5 (under-relaxed for stability), with `max_iter = 20`
and `target_rms = 10⁻⁶ rad`.

([recon.c:732–805](../../rippra/src/recon.c))

---

## 12. Linear Algebra Utilities

All matrix operations are implemented in pure C99 with OpenMP parallelisation
([la.c](../../rippra/src/la.c)).

### 12.1 Matrix Multiply & Vector

```
C(m×n) = A(m×k) · B(k×n)
y(m)   = A(m×n) · x(n)
```

Standard O(n³) and O(n²) implementations with cache-friendly row-major
access.

### 12.2 SVD Pseudo-Inverse (One-Sided Jacobi)

The pseudo-inverse `A⁺` of an `m × n` matrix `A` is computed via the
one-sided Jacobi SVD algorithm:

1. **Initialize**: `W = A` (working copy), `V = I` (n × n identity)
2. **Sweep**: For each column pair `(p, q)`, compute the Jacobi rotation
   that orthogonalises them:

```
α = W[:,p]·W[:,p]      β = W[:,q]·W[:,q]      γ = W[:,p]·W[:,q]

ζ = (β − α) / (2γ)
t = sign(ζ) / (|ζ| + √(1 + ζ²))
c = 1 / √(1 + t²)      s = c · t

rotate W[:,p], W[:,q] and V[:,p], V[:,q] by [c  −s;  s  c]
```

3. **Convergence**: Repeat sweeps until all column pairs are orthogonal
   (typically 3–5 sweeps for well-conditioned matrices).
4. **Singular values**: `σ_p = ‖W[:,p]‖`
5. **Truncate**: Zero columns where `σ_p < rcond · σ_max`
6. **Assemble**: `A⁺ = V · Σ⁺ · W^T`

`rcond = 10⁻⁴` (singular values below 0.01% of max are discarded).

See `rippa_pinv()` at [la.c:194–276](../../rippra/src/la.c).

### 12.3 LU Solve (Doolittle with Partial Pivoting)

Solves the linear system `A · x = b` in-place:

```
for each column k:
    pivot: swap row k with row max|A[i][k]|
    eliminate: A[i][k] /= A[k][k];  for j>k: A[i][j] −= A[i][k] · A[k][j]

Forward solve L·y = b
Back solve U·x = y
```

Returns 0 on success, 1 if singular.

See `rippa_lusolve()` at [la.c:105–189](../../rippra/src/la.c).

---

## 13. Synthetic Data Generation

The Python module `synthetic_shwfs.py` generates physically accurate
synthetic SHWFS frames for testing and ML training.

### 13.1 Lenslet Grid

Lenslet positions lie on a rectangular grid within a circular pupil:

```
sx = cx + u · pitch_px    for u = −⌊R/p⌋ … ⌊R/p⌋
sy = cy + v · pitch_px    for v = −⌊R/p⌋ … ⌊R/p⌋

retain if (sx − cx)² + (sy − cy)² ≤ R_pupil_px²
```

with `pitch_px = pitch / pixsize ≈ 40.5 px`.

### 13.2 PSF Model

Each lenslet produces a Gaussian PSF:

```
I(x, y) = A · exp[ −(Δx² + Δy²) / (2·σ²) ]
```

with `A = 600` (amplitude), `σ = 1.5 px` (PSF width), and a uniform
background of `20` counts.

### 13.3 Wavefront → Shift Mapping

Given Zernike coefficients `a_j` (in radians), the spot displacement is:

```
dx_k = Σ a_j · (∂Z_j/∂x)(x_k, y_k) · FL · λ / (2π · R_pupil · pixsize)
dy_k = Σ a_j · (∂Z_j/∂y)(x_k, y_k) · FL · λ / (2π · R_pupil · pixsize)
```

([synthetic_shwfs.py:shifts_from_wavefront](../../rippra/ml/synthetic_shwfs.py))

### 13.4 File Format

Frames are saved as raw binary doubles (matching `rippa_load_raw()`):

```
np.array(frame, dtype=np.float64).tofile(path)
```

Row-major, 648 × 492 = 318,816 elements = 2,550,528 bytes.

### 13.5 ML Dataset

The `generate_ml_dataset()` function produces `(N, 2·nspots)` displacement
vectors and `(N, nmodes)` coefficient vectors. Zernike coefficient
distributions follow Noll's Kolmogorov variance
(Noll 1976, Table 1):

```
⟨a_j²⟩ = 0.4874 · (D/r₀)^{5/3} · c_j

c_j = 0.582            for j = 2, 3  (tip/tilt)
c_j = 0.294 · j^{−√3/2} for j ≥ 4    (higher modes)
```

---

## 14. References

1. **Noll, R. J.** (1976). "Zernike polynomials and atmospheric turbulence."
   *J. Opt. Soc. Am.*, 66(3), 207–211.
   — Zernike ordering, Kolmogorov modal variances.

2. **Fried, D. L.** (1965). "Statistics of a Geometric Representation of
   Wavefront Distortion." *J. Opt. Soc. Am.*, 55(11), 1427–1435.
   — Fried parameter r₀, differential tilt variance.

3. **Sasiela, R. J.** (1994). *Electromagnetic Wave Propagation in
   Turbulence*. Springer.
   — r₀ from differential image motion.

4. **Roddier, F.** (1999). *Adaptive Optics in Astronomy*. Cambridge.
   — Shack-Hartmann wavefront sensing, zonal/modal reconstruction,
   closed-loop AO control.

5. **Golub, G. H. & Van Loan, C. F.** (2013). *Matrix Computations* (4th ed.).
   Johns Hopkins.
   — SVD, pseudo-inverse, LU decomposition, Jacobi methods.

6. **Thomas, S. et al.** (2006). "Comparison of centroid computation
   algorithms in a Shack-Hartmann sensor." *MNRAS*, 371(1), 323–336.
   — TCoG performance analysis.

7. **Kogbetliantz, E.** (1955). "Solution of linear equations by
   orthogonalization." *Numerische Mathematik*, 7(1), 39–55.
   — One-sided Jacobi SVD.

---

*Last updated: 2026-06-27*
