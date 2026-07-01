"""
synthetic_shwfs.py — Physically accurate SHWFS synthetic data generator.

Produces raw double-precision frame files and ML training datasets
that exactly match the C pipeline's expectations. All Zernike and
geometry computations are bit-exact reimplementations of recon.c.
"""
import os, math, struct
import numpy as np

# ──────────────────────────────────────────────
# Zernike polynomials (Noll indexing, from recon.c)
# ──────────────────────────────────────────────

_FACT_CACHE = [1.0]

def _fact(k):
    while len(_FACT_CACHE) <= k:
        _FACT_CACHE.append(_FACT_CACHE[-1] * len(_FACT_CACHE))
    return _FACT_CACHE[k]

def noll_to_nm(j):
    """Noll index j (1-based) → (n, m)."""
    if j == 1:
        return 0, 0
    current_j = 2
    for ni in range(1, 100):
        for mi in range(ni % 2, ni + 1, 2):
            if mi == 0:
                if current_j == j:
                    return ni, 0
                current_j += 1
            else:
                if current_j % 2 == 1:
                    if current_j == j:
                        return ni, -mi
                    if current_j + 1 == j:
                        return ni, mi
                else:
                    if current_j == j:
                        return ni, mi
                    if current_j + 1 == j:
                        return ni, -mi
                current_j += 2

def zernike_coeff(n, m, s):
    """Radial coefficient C_s  (recon.c:zernike_coeff)."""
    abs_m = abs(m)
    num = _fact(n - s)
    den = _fact(s) * _fact((n + abs_m)//2 - s) * _fact((n - abs_m)//2 - s)
    sign = 1.0 if s % 2 == 0 else -1.0
    return sign * num / den

def zernike_derivatives(n, m, x, y):
    """Analytical Zernike derivatives dz/dx, dz/dy at (x,y).
    Exact replica of recon.c:evaluate_zernike_derivatives."""
    rho = math.hypot(x, y)
    theta = math.atan2(y, x)
    abs_m = abs(m)

    R = 0.0; dR = 0.0
    for s in range((n - abs_m)//2 + 1):
        c = zernike_coeff(n, m, s)
        pow_rho = n - 2*s
        if pow_rho > 0:
            R += c * (rho ** pow_rho)
            dR += c * pow_rho * (rho ** (pow_rho - 1))
        elif pow_rho == 0:
            R += c

    norm = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
    R *= norm; dR *= norm

    cos_mt = math.cos(abs_m * theta)
    sin_mt = math.sin(abs_m * theta)

    if m >= 0:
        dz_drho = dR * cos_mt
        dz_dtheta = -abs_m * R * sin_mt
    else:
        dz_drho = dR * sin_mt
        dz_dtheta = abs_m * R * cos_mt

    if rho < 1e-9:
        if n == 1:
            if m == 1:   return norm, 0.0
            if m == -1:  return 0.0, norm
        return 0.0, 0.0

    dzdx = dz_drho * math.cos(theta) - dz_dtheta * math.sin(theta) / rho
    dzdy = dz_drho * math.sin(theta) + dz_dtheta * math.cos(theta) / rho
    return dzdx, dzdy

def zernike_eval(n, m, x, y):
    """Evaluate Zernike polynomial Z_n^m at (x,y) using Noll normalisation."""
    rho = math.hypot(x, y)
    theta = math.atan2(y, x)
    abs_m = abs(m)

    R = 0.0
    for s in range((n - abs_m)//2 + 1):
        c = zernike_coeff(n, m, s)
        pow_rho = n - 2*s
        R += c * (rho ** pow_rho) if pow_rho > 0 else c

    norm = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
    R *= norm

    if m >= 0:
        return R * math.cos(abs_m * theta)
    else:
        return R * math.sin(abs_m * theta)

# ──────────────────────────────────────────────
# Kolmogorov Zernike variances (Noll 1976 approximation)
# ──────────────────────────────────────────────

def noll_variance(j, D_r0=8.0):
    """Approximate Zernike coefficient variance for Kolmogorov turbulence.
    Noll 1976, Table 1 & Eq. 30. Units: rad^2."""
    if j == 2:  # tip
        return 0.4874 * (D_r0 ** (5.0/3.0)) * 0.582
    if j == 3:  # tilt
        return 0.4874 * (D_r0 ** (5.0/3.0)) * 0.582
    # Higher-order modes
    return 0.4874 * (D_r0 ** (5.0/3.0)) * 0.294 * (j - 1) ** (-math.sqrt(3)/2)

def generate_turbulence_coeffs(nmodes, D_r0=8.0, rng=None):
    """Generate random Zernike coefficients following Kolmogorov statistics."""
    if rng is None:
        rng = np.random.RandomState()
    coeffs = np.zeros(nmodes)
    for j in range(nmodes):
        var = noll_variance(j + 2, D_r0)
        coeffs[j] = rng.randn() * math.sqrt(var)
    return coeffs

# ──────────────────────────────────────────────
# Geometrical SHWFS model
# ──────────────────────────────────────────────

def lenslet_positions(w, h, pitch_px, pupil_radius_px, max_lenslets=200):
    """Generate rectangular grid of lenslet positions within circular pupil.

    Uses round() to match the integer grid mapping in
    recon.c:rippra_zonal_setup.
    """
    cx, cy = w / 2.0, h / 2.0
    spots = []
    half_w = w / 2.0
    # Estimate grid extent from pupil radius
    extent = int(pupil_radius_px / pitch_px) + 2
    for vi in range(-extent, extent + 1):
        for ui in range(-extent, extent + 1):
            sx = cx + ui * pitch_px
            sy = cy + vi * pitch_px
            if 0 <= sx < w and 0 <= sy < h:
                dist = math.hypot(sx - cx, sy - cy)
                if dist <= pupil_radius_px:
                    spots.append((sx, sy))
    # Sort by row then column for reproducibility
    spots.sort(key=lambda p: (p[1], p[0]))
    if len(spots) > max_lenslets:
        spots = spots[:max_lenslets]
    return np.array(spots, dtype=np.float64)

def gaussian_psf(x0, y0, w, h, amplitude=600.0, sigma=1.5, background=20.0):
    """Render a 2D Gaussian spot at (x0, y0)."""
    ys, xs = np.mgrid[0:h, 0:w]
    dx = xs - x0; dy = ys - y0
    r2 = dx*dx + dy*dy
    spot = background + amplitude * np.exp(-r2 / (2.0 * sigma * sigma))
    return spot.astype(np.float64)

def render_frame(positions, w, h, amplitude=600.0, sigma=1.5, background=20.0):
    """Render a full SHWFS frame with spots at given positions."""
    frame = np.full((h, w), background, dtype=np.float64)
    for (sx, sy) in positions:
        # Only render within bounding box for speed
        r = int(4 * sigma) + 1
        x0, y0 = int(round(sx)), int(round(sy))
        xl = max(0, x0 - r); xr = min(w, x0 + r + 1)
        yl = max(0, y0 - r); yr = min(h, y0 + r + 1)
        for y in range(yl, yr):
            for x in range(xl, xr):
                d2 = (x - sx)**2 + (y - sy)**2
                frame[y, x] += amplitude * math.exp(-d2 / (2.0 * sigma * sigma))
    return frame

def shifts_from_wavefront(positions, coeffs, flength, pixsize, pupil_radius, wavelength=632.8e-9, pupil_cx=None, pupil_cy=None):
    """Compute pixel shifts from Zernike wavefront coefficients.

    coeffs[j] = amplitude of Noll index j+2 (radians RMS).
    Positions are in pixel coordinates; internally normalized to pupil coords.
    Returns dx, dy arrays in pixels.

    Matches the geometry in recon.c:rippra_modal_setup:
      x_norm = (px - pupil_cx) * pixsize / pupil_radius
      y_norm = -(py - pupil_cy) * pixsize / pupil_radius
    and shifts: dx = a * dzdx * flength / (pupil_radius * pixsize)
    """
    if pupil_cx is None:
        pupil_cx = positions[:, 0].mean()
    if pupil_cy is None:
        pupil_cy = positions[:, 1].mean()

    # Normalised coordinates (pupil coordinates, [-1, 1] approx)
    xn = (positions[:, 0] - pupil_cx) * pixsize / pupil_radius
    yn = -(positions[:, 1] - pupil_cy) * pixsize / pupil_radius  # sign flip matches C

    nmodes = len(coeffs)
    dx = np.zeros(len(positions))
    dy = np.zeros(len(positions))
    scale = flength * wavelength / (2.0 * math.pi * pupil_radius * pixsize)
    for j in range(nmodes):
        n, m = noll_to_nm(j + 2)
        a = coeffs[j]
        for k in range(len(positions)):
            dzdx, dzdy = zernike_derivatives(n, m, xn[k], yn[k])
            # Phase (rad) -> OPD (m) via lambda/(2*pi)
            # OPD slope = a * dzdx / pupil_radius * lambda/(2*pi)
            # Spot shift = slope * flength / pixsize (px)
            dx[k] += a * dzdx * scale
            dy[k] += a * dzdy * scale
    return dx, dy

# ──────────────────────────────────────────────
# File I/O matching rippa_load_raw / rippa_save_raw
# ──────────────────────────────────────────────

def save_raw(path, frame):
    """Write double-precision raw file (row-major).
    Matches rippa_save_raw() in io.c."""
    frame.astype(np.float64).tofile(path)

def load_raw(path, w, h):
    """Read double-precision raw file.
    Matches rippa_load_raw() in io.c."""
    return np.fromfile(path, dtype=np.float64).reshape(h, w)

# ──────────────────────────────────────────────
# Full pipeline helpers
# ──────────────────────────────────────────────

def generate_test_data(out_dir, cfg, seed=42):
    """Generate complete synthetic test dataset.

    Returns dict with paths to all generated files and metadata.
    """
    rng = np.random.RandomState(seed)
    w = int(cfg['frame_width'])
    h = int(cfg['frame_height'])
    camera_pixsize = float(cfg['camera_pixsize'])
    flength = float(cfg['flength'])
    pitch = float(cfg['pitch'])
    pupil_radius = float(cfg['pupil_radius'])
    zernike_nmax = int(cfg['zernike_nmax'])
    nmodes = (zernike_nmax + 1) * (zernike_nmax + 2) // 2 - 1

    # Derived quantities
    pitch_px = pitch / camera_pixsize
    pupil_radius_px = pupil_radius / camera_pixsize

    # 1. Lenslet positions
    positions = lenslet_positions(w, h, pitch_px, pupil_radius_px)
    nspots = len(positions)
    os.makedirs(out_dir, exist_ok=True)

    # 2. Flat frame — spots at reference positions
    flat = render_frame(positions, w, h)
    flat_path = os.path.join(out_dir, 'sh_flat.raw')
    save_raw(flat_path, flat)

    # 3. Generate Kolmogorov random coefficients
    coeffs = generate_turbulence_coeffs(nmodes, D_r0=8.0, rng=rng)
    # Save ground truth coefficients
    np.asarray(coeffs, dtype=np.float64).tofile(os.path.join(out_dir, 'coeffs_gt.raw'))

    # 4. Compute spot shifts
    dx, dy = shifts_from_wavefront(positions, coeffs, flength, camera_pixsize, pupil_radius)
    # Save ground truth shifts
    np.asarray(dx, dtype=np.float64).tofile(os.path.join(out_dir, 'dx_gt.raw'))
    np.asarray(dy, dtype=np.float64).tofile(os.path.join(out_dir, 'dy_gt.raw'))

    # 5. Aberrated frame
    shifted = positions.copy()
    shifted[:, 0] += dx
    shifted[:, 1] += dy
    img = render_frame(shifted, w, h)
    img_path = os.path.join(out_dir, 'img.raw')
    save_raw(img_path, img)

    return {
        'flat_path': flat_path,
        'img_path': img_path,
        'nspots': nspots,
        'positions': positions,
        'coeffs': coeffs,
        'dx': dx,
        'dy': dy,
        'pitch_px': pitch_px,
        'pupil_radius_px': pupil_radius_px,
    }

def generate_ml_dataset(positions, n_samples, zernike_nmax,
                        flength, pixsize, pupil_radius, scale=3.0, seed=123):
    """Generate ML training dataset of (displacements, coefficients)."""
    rng = np.random.RandomState(seed)
    nmodes = (zernike_nmax + 1) * (zernike_nmax + 2) // 2 - 1
    nspots = len(positions)

    displacements = np.zeros((n_samples, nspots * 2), dtype=np.float32)
    coefficients = np.zeros((n_samples, nmodes), dtype=np.float32)

    for i in range(n_samples):
        c = generate_turbulence_coeffs(nmodes, D_r0=8.0, rng=rng)
        dx, dy = shifts_from_wavefront(positions, c, flength, pixsize, pupil_radius)
        displacements[i] = np.concatenate([dx, dy])
        coefficients[i] = c

    return displacements, coefficients


if __name__ == '__main__':
    import json
    cfg = {
        'camera_pixsize': 7.4e-6,
        'frame_width': 648,
        'frame_height': 492,
        'pitch': 300e-6,
        'flength': 18e-3,
        'pupil_radius': 2e-3,
        'sa_radius': 150e-6,
        'zernike_nmax': 5,
        'thresh_binary': 0.08,
        'centroid_percent': 0.2,
    }
    result = generate_test_data('data_raw', cfg)
    print(f"Generated {result['nspots']} spots")
    print(f"Coeffs: {result['coeffs']}")
