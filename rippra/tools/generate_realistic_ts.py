"""
Generate physically realistic SH-WFS time-series data.
- Real Zernike coefficients from measured frame as initial condition
- Kolmogorov turbulence spatial correlations (Noll covariance)
- AR(1) temporal evolution with turbulence strength from real data
- Saves per-frame centroids + Zernike coefficients + turbulence params
"""
import os, sys, argparse, math
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'ml'))
from evaluate_inference import noll_to_nm, compute_classical_zernike, load_system_config
from evaluate_inference import zernike_derivatives
from generate_dataset import get_noll_covariance_matrix

def estimate_tau0_from_deltas(dx_series, dy_series, nspots, frame_rate=1000.0):
    nf = len(dx_series) // nspots
    acf = np.zeros(nf // 2)
    for k in range(nspots):
        for sig in [dx_series, dy_series]:
            s = sig[np.arange(k, len(sig), nspots)]
            s -= s.mean()
            c = np.correlate(s, s, mode='full')[len(s)-1:] / (s.std()**2 * len(s) + 1e-12)
            acf += c[:nf//2]
    acf /= (2 * nspots)
    for t in range(1, len(acf)):
        if acf[t] <= 1.0/np.e:
            frac = (1.0/np.e - acf[t-1]) / (acf[t] - acf[t-1] + 1e-12)
            return (t - 1 + frac) / frame_rate
    return nf / frame_rate

def main():
    parser = argparse.ArgumentParser(description="Generate realistic SH-WFS time series")
    parser.add_argument("--frames", type=int, default=500, help="Number of frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="results/time_series.csv",
                        help="Output CSV path (relative to project root)")
    args = parser.parse_args()
    np.random.seed(args.seed)

    cfg_path = os.path.join(BASE, "config", "system.conf")
    cfg = load_system_config(cfg_path)
    spots_df = pd.read_csv(os.path.join(BASE, "results", "reference_centroids_c.csv"))
    dev_df = pd.read_csv(os.path.join(BASE, "results", "spot_deviations_c.csv"))
    nspots = len(spots_df)
    dx_real = dev_df['Delta_X'].values
    dy_real = dev_df['Delta_Y'].values

    print("Computing real Zernike coefficients from measured frame...")
    coeffs_real, Zprime = compute_classical_zernike(dx_real, dy_real, cfg, spots_df)
    nmodes = len(coeffs_real)
    print(f"  First 6 coeffs (rad): {np.round(coeffs_real[:6], 4)}")

    # Compute effective D/r0 from variance of real Zernike coefficients
    D = 2.0 * cfg['pupil_radius']  # aperture DIAMETER, not radius
    zernike_nmax = int(cfg['zernike_nmax'])
    max_j = (zernike_nmax + 1) * (zernike_nmax + 2) // 2
    C, modes = get_noll_covariance_matrix(max_j)

    # Compute effective (D/r0)^(5/3) by matching coefficient variance
    # Var[z_i] = C_ii * (D/r0)^(5/3). Solve for D/r0 from mean variance ratio
    var_real = np.var(coeffs_real)
    mean_C = np.mean(np.diag(C))
    if mean_C > 0:
        D_r0_eff = (var_real / mean_C) ** (3.0/10.0)  # since var = C_ii * (D/r0)^(5/3)
    else:
        D_r0_eff = 1.0
    # Apply a mild clamp to avoid extreme values but keep it physically meaningful
    D_r0_eff = max(0.1, min(D_r0_eff, 10.0))
    print(f"  Aperture D = {D*1000:.1f} mm, effective D/r0 = {D_r0_eff:.2f}")
    print(f"  Effective r0 = {D/D_r0_eff*100:.1f} cm")

    # Setup AR(1) temporal model
    frame_rate = 1000.0
    dt = 1.0 / frame_rate
    # tau0 from D/r0 with typical wind = 10 m/s
    v_wind = 10.0
    tau0_eff = 0.314 * (D / D_r0_eff) / v_wind
    tau0_eff = max(tau0_eff, 0.001)
    print(f"  tau0 = {tau0_eff*1000:.1f} ms (wind {v_wind} m/s)")
    rho = np.exp(-dt / tau0_eff)

    # Scale Noll covariance by effective (D/r0)^(5/3)
    cov = C * (D_r0_eff ** (5.0 / 3.0))
    reg_cov = cov + np.eye(nmodes) * 1e-8
    L = np.linalg.cholesky(reg_cov)

    scale_factor = cfg['flength'] / cfg['camera_pixsize']
    a_rad = coeffs_real.copy()

    print(f"Generating {args.frames} frames with AR(1) Kolmogorov evolution...")
    nf = args.frames
    zernike_ts = np.zeros((nf, nmodes))
    dx_ts = np.zeros((nf, nspots))
    dy_ts = np.zeros((nf, nspots))
    r0_ts = np.zeros(nf)
    tau0_ts = np.zeros(nf)

    for t in range(nf):
        if t > 0:
            epsilon = L.dot(np.random.randn(nmodes))
            a_rad = rho * a_rad + np.sqrt(1.0 - rho * rho) * epsilon
        zernike_ts[t] = a_rad

        a_m = a_rad * (cfg['wavelength'] / (2.0 * np.pi))
        disp = Zprime.dot(a_m) * scale_factor
        dx_ts[t] = disp[:nspots]
        dy_ts[t] = disp[nspots:]

        # Per-frame r0 estimate from displacement variance
        var_x = np.var(dx_ts[t])
        var_y = np.var(dy_ts[t])
        mean_var = (var_x + var_y) / 2.0
        d = cfg['pitch'] / cfg['pupil_radius']
        lam = cfg['wavelength']
        r0_ts[t] = (0.170 * lam**2 * d**(-1/3) / (mean_var * (cfg['camera_pixsize'])**2 + 1e-20))**(3/5)

        # tau0 from 50-frame sliding window
        if t >= 50:
            wdx = dx_ts[t-49:t+1].ravel()
            wdy = dy_ts[t-49:t+1].ravel()
            tau0_ts[t] = estimate_tau0_from_deltas(wdx, wdy, nspots, frame_rate)
        else:
            tau0_ts[t] = tau0_eff

    # CSV columns
    columns = (['frame'] + [f'z{i}' for i in range(nmodes)] +
               [f'dx{i}' for i in range(nspots)] + [f'dy{i}' for i in range(nspots)] +
               ['r0', 'tau0'])
    data = np.column_stack([np.arange(nf), zernike_ts, dx_ts, dy_ts, r0_ts, tau0_ts])
    df = pd.DataFrame(data, columns=columns)

    out_path = os.path.join(BASE, args.out)
    df.to_csv(out_path, index=False, float_format='%.8f')
    print("Saved time_series.csv")

    zcols = ['frame'] + [c for c in df.columns if c.startswith('z')]
    zp = os.path.join(BASE, "results", "zernike_time_series.csv")
    df[zcols].to_csv(zp, index=False, float_format='%.8f')
    print("Saved zernike_time_series.csv")

    print(f"  r0 range: {r0_ts.min()*100:.1f}-{r0_ts.max()*100:.1f} cm")
    print(f"  tau0 range: {tau0_ts.min()*1000:.1f}-{tau0_ts.max()*1000:.1f} ms")

if __name__ == '__main__':
    main()
