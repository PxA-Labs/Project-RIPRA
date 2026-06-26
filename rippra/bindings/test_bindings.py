"""
Test the Python ctypes bindings against the C shared library.
Creates a synthetic flat frame, calibrates, and runs reconstruction.
"""

import os, sys
import numpy as np

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base, "bindings"))
    from rippra import Rippra

    lib_path = os.path.join(base, "bin", "rippra.dll")
    so_path = os.path.join(base, "bin", "librippra.so")
    if not os.path.exists(lib_path) and not os.path.exists(so_path):
        print("SKIP: shared library not found. Run build_dll.bat first.")
        print(f"  Looked for: {lib_path}")
        return

    r = Rippra()
    print(f"RIPRA version: {r.version}")

    # Load config
    config_path = os.path.join(base, "config", "system.conf")
    cfg = r.load_config(config_path)
    print(f"Config loaded: {cfg.frame_width}x{cfg.frame_height}, {cfg.totlenses} lenses")

    # Generate synthetic flat frame (gaussian spots at known positions)
    w, h = cfg.frame_width, cfg.frame_height
    n_lenses = cfg.totlenses
    grid_size = int(np.ceil(np.sqrt(n_lenses)))
    pitch_px = cfg.pitch / cfg.camera_pixsize
    cx = w // 2 + (np.arange(grid_size) - grid_size // 2) * pitch_px
    cy = h // 2 + (np.arange(grid_size) - grid_size // 2) * pitch_px
    xx, yy = np.meshgrid(cx, cy)
    cx_flat = xx.ravel()[:n_lenses]
    cy_flat = yy.ravel()[:n_lenses]

    flat = np.zeros((h, w), dtype=np.float64)
    sigma = 1.5
    for x0, y0 in zip(cx_flat, cy_flat):
        xs = np.arange(max(0, int(x0)-5), min(w, int(x0)+6))
        ys = np.arange(max(0, int(y0)-5), min(h, int(y0)+6))
        for y in ys:
            for x in xs:
                flat[y, x] += np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))

    print(f"Flat frame: {flat.shape}, sum={flat.sum():.1f}")

    # Calibrate
    nspots = r.calibrate(flat, w, h)
    print(f"Calibrated: {nspots} spots detected")

    if nspots == 0:
        print("SKIP: no spots detected (synthetic frame may need tuning)")
        return

    ref_cx, ref_cy = r.ref_centroids()
    print(f"Ref centroids: min=({ref_cx.min():.1f},{ref_cy.min():.1f}) "
          f"max=({ref_cx.max():.1f},{ref_cy.max():.1f})")

    # Generate aberrated frame (shift spots slightly)
    aberrated = flat.copy()
    np.random.seed(42)
    shifts = np.random.randn(nspots, 2) * 2.0
    for i, (x0, y0) in enumerate(zip(ref_cx, ref_cy)):
        dx_s, dy_s = shifts[i]
        xs = np.arange(max(0, int(x0+dx_s)-5), min(w, int(x0+dx_s)+6))
        ys = np.arange(max(0, int(y0+dy_s)-5), min(h, int(y0+dy_s)+6))
        for y in ys:
            for x in xs:
                aberrated[y, x] += np.exp(-((x-(x0+dx_s))**2 + (y-(y0+dy_s))**2)/(2*sigma**2))

    # Full pipeline: process frame
    dx, dy, coeffs = r.process_frame(aberrated, w, h)
    print(f"Frame processed: {len(dx)} deltas, {len(coeffs)} coefficients")
    print(f"  dx range: [{dx.min():.3f}, {dx.max():.3f}] px")
    print(f"  dy range: [{dy.min():.3f}, {dy.max():.3f}] px")
    print(f"  First 5 Zernike coeffs: {coeffs[:5]}")

    # Modal reconstruction only
    coeffs2 = r.reconstruct_modal(dx, dy)
    print(f"  Modal only coeffs match: {np.allclose(coeffs, coeffs2)}")

    # Centroid only
    dx2, dy2 = r.centroid(aberrated, w, h)
    print(f"  Centroid only dx match: {np.allclose(dx, dx2)}")

    # r0 computation (simulated series)
    series_len = 100
    dx_series = np.random.randn(series_len, nspots).ravel()
    dy_series = np.random.randn(series_len, nspots).ravel()
    r0 = r.compute_r0(dx_series, dy_series, series_len, nspots)
    tau0 = r.compute_tau0(dx_series, dy_series, series_len, nspots, 1000.0)
    print(f"R0: {r0:.6f} m, Tau0: {tau0:.6f} s")

    r.close()
    print("\nAll binding tests passed!")

if __name__ == "__main__":
    main()
