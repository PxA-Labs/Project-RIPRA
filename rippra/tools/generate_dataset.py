#!/usr/bin/env python3
"""
generate_dataset.py - Generate synthetic Kolmogorov turbulence dataset for Shack-Hartmann WFS
"""
import os
import argparse
import numpy as np
import scipy.special as special
import pandas as pd

def noll_to_nm(j):
    """Convert Noll index j (1-based) to radial order n and azimuthal frequency m"""
    if j == 1:
        return 0, 0
    current_j = 2
    for n in range(1, 100):
        for m in range(n % 2, n + 1, 2):
            if m == 0:
                if current_j == j:
                    return n, 0
                current_j += 1
            else:
                # Same parity
                if current_j % 2 == 1:
                    if current_j == j:
                        return n, -m
                    if current_j + 1 == j:
                        return n, m
                else:
                    if current_j == j:
                        return n, m
                    if current_j + 1 == j:
                        return n, -m
                current_j += 2

def noll_covariance(n1, m1, j1, n2, m2, j2):
    """Compute the Noll covariance coefficient C_ij for Kolmogorov turbulence"""
    if abs(m1) != abs(m2):
        return 0.0
    # Parity check: both must be even or both odd Noll indices for same m
    if (j1 % 2) != (j2 % 2) and abs(m1) != 0:
        return 0.0
    
    # Sign factor
    sign = (-1.0) ** ((n1 + n2 - 2 * abs(m1)) / 2)
    
    # Gamma terms
    num = (sign * np.sqrt((n1 + 1) * (n2 + 1)) * 
           special.gamma(14.0 / 3.0) * 
           special.gamma((n1 + n2 - 5.0 / 3.0) / 2.0))
    den = (special.gamma((n1 - n2 + 17.0 / 3.0) / 2.0) * 
           special.gamma((n2 - n1 + 17.0 / 3.0) / 2.0) * 
           special.gamma((n1 + n2 + 23.0 / 3.0) / 2.0))
    
    return num / den

def get_noll_covariance_matrix(max_j):
    """Compute the Noll covariance matrix for j = 2 to max_j (excluding piston)"""
    nmodes = max_j - 1
    C = np.zeros((nmodes, nmodes))
    modes = []
    for idx in range(nmodes):
        j = idx + 2
        n, m = noll_to_nm(j)
        modes.append((j, n, m))
        
    for i in range(nmodes):
        j1, n1, m1 = modes[i]
        for j_idx in range(nmodes):
            j2, n2, m2 = modes[j_idx]
            C[i, j_idx] = noll_covariance(n1, m1, j1, n2, m2, j2)
            
    return C, modes

# Zernike radial coefficient Helper
def zernike_coeff(n, m, s):
    abs_m = abs(m)
    num = special.factorial(n - s)
    den = special.factorial(s) * special.factorial((n + abs_m)//2 - s) * special.factorial((n - abs_m)//2 - s)
    sign = 1.0 if s % 2 == 0 else -1.0
    return sign * num / den

# Evaluate Zernike derivatives at normalized polar/cartesian coordinate
def zernike_derivatives(n, m, x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    abs_m = abs(m)
    
    R = 0.0
    dR = 0.0
    for s in range((n - abs_m) // 2 + 1):
        c = zernike_coeff(n, m, s)
        pow_rho = n - 2*s
        if pow_rho > 0:
            R += c * (rho ** pow_rho)
            dR += c * pow_rho * (rho ** (pow_rho - 1))
        elif pow_rho == 0:
            R += c
            
    norm_factor = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    R *= norm_factor
    dR *= norm_factor
    
    cos_mt = np.cos(abs_m * theta)
    sin_mt = np.sin(abs_m * theta)
    
    if m >= 0:
        dz_drho = dR * cos_mt
        dz_dtheta = -abs_m * R * sin_mt
    else:
        dz_drho = dR * sin_mt
        dz_dtheta = abs_m * R * cos_mt
        
    if rho < 1e-9:
        if n == 1:
            if m == 1:
                return norm_factor, 0.0
            elif m == -1:
                return 0.0, norm_factor
        return 0.0, 0.0
        
    dzdx = dz_drho * np.cos(theta) - dz_dtheta * np.sin(theta) / rho
    dzdy = dz_drho * np.sin(theta) + dz_dtheta * np.cos(theta) / rho
    return dzdx, dzdy

def load_system_config(config_path):
    cfg = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.split('#', 1)[0].strip()
                try:
                    cfg[key] = float(val) if '.' in val or 'e' in val else int(val)
                except ValueError:
                    cfg[key] = val
    return cfg

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Shack-Hartmann dataset")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--noise", type=float, default=0.1, help="Standard deviation of displacement noise in pixels")
    parser.add_argument("--out", type=str, default="data_ai/dataset.npz", help="Output filepath")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # 1. Load config and spot coordinates
    print("Loading system configuration...")
    cfg = load_system_config("config/system.conf")
    
    spots_file = "results/reference_centroids_c.csv"
    if not os.path.exists(spots_file):
        print(f"Error: {spots_file} not found. Please run test_centroid first to calibrate.")
        return
        
    spots_df = pd.read_csv(spots_file)
    nspots = len(spots_df)
    print(f"Loaded {nspots} active sub-aperture coordinates.")
    
    # 2. Setup Zernike covariance
    zernike_nmax = int(cfg["zernike_nmax"])
    max_j = (zernike_nmax + 1) * (zernike_nmax + 2) // 2
    nmodes = max_j - 1 # exclude piston
    print(f"Setting up Zernike covariance for {nmodes} modes (Noll radial order <= {zernike_nmax})...")
    
    C, modes = get_noll_covariance_matrix(max_j)
    
    # 3. Precompute Zernike Derivative Matrix (Zprime) via area integration
    print("Precomputing Zernike derivative matrix (Zprime)...")
    M = 15 # 15x15 quadrature grid
    rbar = cfg["sa_radius"] / cfg["pupil_radius"]
    kk = cfg["pupil_radius"] / (np.pi * cfg["sa_radius"]**2)
    
    Zprime = np.zeros((2 * nspots, nmodes))
    
    for k in range(nspots):
        ref_cx = spots_df.loc[k, "ref_cx"]
        ref_cy = spots_df.loc[k, "ref_cy"]
        
        # Canonical center coordinates (origin at pupil center, Y pointing up)
        x_c = (ref_cx - spots_df["ref_cx"].mean()) * cfg["camera_pixsize"] / cfg["pupil_radius"]
        y_c = -(ref_cy - spots_df["ref_cy"].mean()) * cfg["camera_pixsize"] / cfg["pupil_radius"]
        
        for m_idx, (j, n, m) in enumerate(modes):
            sum_dzdx = 0.0
            sum_dzdy = 0.0
            count_pts = 0
            
            # 2D area integration over sub-aperture disk
            for r_step in range(M):
                dy = -rbar + 2.0 * rbar * r_step / (M - 1)
                for c_step in range(M):
                    dx = -rbar + 2.0 * rbar * c_step / (M - 1)
                    if dx**2 + dy**2 <= rbar**2:
                        dzdx, dzdy = zernike_derivatives(n, m, x_c + dx, y_c + dy)
                        sum_dzdx += dzdx
                        sum_dzdy += dzdy
                        count_pts += 1
                        
            avg_dzdx = sum_dzdx / count_pts if count_pts > 0 else 0.0;
            avg_dzdy = sum_dzdy / count_pts if count_pts > 0 else 0.0;
            
            # Map to Zprime elements
            Zprime[k, m_idx] = kk * avg_dzdx
            Zprime[k + nspots, m_idx] = -kk * avg_dzdy
            
    # 4. Generate Samples using AR(1) Temporal Correlation
    print(f"Generating {args.samples} samples (arranged in temporally correlated sequences)...")
    displacements = np.zeros((args.samples, 2 * nspots))
    coefficients = np.zeros((args.samples, nmodes))
    D_r0_arr = np.zeros(args.samples)
    
    # Scale factor mapping coefficients in meters to displacements in pixels
    scale_factor = cfg["flength"] / cfg["camera_pixsize"]
    
    # Define sequence lengths (default 1000 frames per sequence representing 1 second at 1000 Hz)
    seq_len = 1000
    n_seq = args.samples // seq_len
    if n_seq == 0:
        n_seq = 1
        seq_len = args.samples
        
    print(f"  Configuration: {n_seq} sequences of length {seq_len} frames.")
    
    frame_idx = 0
    for s in range(n_seq):
        # Turbulence strength for this sequence
        D_r0 = np.random.uniform(1.0, 10.0)
        # Coherence time for this sequence: tau0 in range 2 ms to 10 ms
        tau0 = np.random.uniform(0.002, 0.010)
        fs = 1000.0 # 1000 Hz frame rate (1 ms interval)
        
        # AR(1) coefficient: rho = exp(-dt / tau0)
        rho = np.exp(-1.0 / (fs * tau0))
        
        # Scale covariance by (D/r0)^(5/3)
        cov = C * (D_r0 ** (5.0 / 3.0))
        reg_cov = cov + np.eye(nmodes) * 1e-8
        
        # Draw starting coefficients for the sequence
        a_rad = np.random.multivariate_normal(np.zeros(nmodes), reg_cov)
        
        for t in range(seq_len):
            if t > 0:
                # AR(1) step: update Zernike coefficients with temporal correlation
                epsilon = np.random.multivariate_normal(np.zeros(nmodes), reg_cov)
                a_rad = rho * a_rad + np.sqrt(1.0 - rho * rho) * epsilon
                
            # Convert to meters for physical slope mapping: a_m = a_rad * (lambda / 2pi)
            a_m = a_rad * (cfg["wavelength"] / (2.0 * np.pi))
            
            # Calculate clean displacements in pixels
            clean_disp = Zprime.dot(a_m) * scale_factor
            
            # Add noise
            noise = np.random.normal(0.0, args.noise, 2 * nspots)
            noisy_disp = clean_disp + noise
            
            displacements[frame_idx] = noisy_disp
            coefficients[frame_idx] = a_rad
            D_r0_arr[frame_idx] = D_r0
            
            frame_idx += 1
            if frame_idx >= args.samples:
                break
                
        print(f"  Sequence {s + 1:02d}/{n_seq:02d} complete (D/r0={D_r0:.2f}, tau0={tau0*1000.0:.2f} ms).")
        
    # 5. Save dataset
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    np.savez(args.out, 
             displacements=displacements, 
             coefficients=coefficients,
             D_r0=D_r0_arr)
             
    print(f"Successfully saved dataset to {args.out}!")
    print(f"  Inputs shape:  {displacements.shape}")
    print(f"  Targets shape: {coefficients.shape}")

if __name__ == "__main__":
    import sys
    if not (len(sys.argv) > 0 and ('kernel' in sys.argv[0] or '-f' in sys.argv)):
        main()
