# ml/evaluate_inference.py - Compare classical reconstruction vs. ML models
import os
import sys
import numpy as np
import pandas as pd
import torch
import math
from functools import lru_cache

# Add the current directory to sys.path to allow imports
sys.path.append(os.path.dirname(__file__))

from models import WavefrontMLP, WavefrontCNN
from train import WavefrontDataset

@lru_cache(maxsize=None)
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

@lru_cache(maxsize=None)
def zernike_coef(n, m, s):
    abs_m = abs(m)
    num = math.factorial(n - s)
    den = math.factorial(s) * math.factorial((n + abs_m)//2 - s) * math.factorial((n - abs_m)//2 - s)
    sign = 1.0 if s % 2 == 0 else -1.0
    return sign * num / den

def zernike_derivatives(n, m, x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    abs_m = abs(m)
    
    R = 0.0; dR = 0.0
    for s in range((n - abs_m) // 2 + 1):
        c = zernike_coef(n, m, s)
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

def compute_classical_zernike(dx, dy, cfg, spots_df):
    """Compute classical least-squares modal reconstruction coefficients (in radians)"""
    nspots = len(spots_df)
    zernike_nmax = int(cfg["zernike_nmax"])
    max_j = (zernike_nmax + 1) * (zernike_nmax + 2) // 2
    nmodes = max_j - 1
    
    # Initialize Zprime matrix
    Zprime = np.zeros((2 * nspots, nmodes))
    modes = []
    for idx in range(nmodes):
        j = idx + 2
        n, m = noll_to_nm(j)
        modes.append((j, n, m))
        
    M = 15 # 15x15 quadrature grid
    rbar = cfg["sa_radius"] / cfg["pupil_radius"]
    kk = cfg["pupil_radius"] / (np.pi * cfg["sa_radius"]**2)
    
    mean_cx = spots_df["ref_cx"].mean()
    mean_cy = spots_df["ref_cy"].mean()
    
    for k in range(nspots):
        ref_cx = spots_df.loc[k, "ref_cx"]
        ref_cy = spots_df.loc[k, "ref_cy"]
        
        # Center coordinates
        x_c = (ref_cx - mean_cx) * cfg["camera_pixsize"] / cfg["pupil_radius"]
        y_c = -(ref_cy - mean_cy) * cfg["camera_pixsize"] / cfg["pupil_radius"]
        
        for m_idx, (j, n, m) in enumerate(modes):
            sum_dzdx = 0.0
            sum_dzdy = 0.0
            count_pts = 0
            
            for r_step in range(M):
                dy_grid = -rbar + 2.0 * rbar * r_step / (M - 1)
                for c_step in range(M):
                    dx_grid = -rbar + 2.0 * rbar * c_step / (M - 1)
                    if dx_grid**2 + dy_grid**2 <= rbar**2:
                        dzdx_val, dzdy_val = zernike_derivatives(n, m, x_c + dx_grid, y_c + dy_grid)
                        sum_dzdx += dzdx_val
                        sum_dzdy += dzdy_val
                        count_pts += 1
                        
            avg_dzdx = sum_dzdx / count_pts if count_pts > 0 else 0.0
            avg_dzdy = sum_dzdy / count_pts if count_pts > 0 else 0.0
            
            Zprime[k, m_idx] = kk * avg_dzdx
            Zprime[k + nspots, m_idx] = -kk * avg_dzdy
            
    # Solve pseudo-inverse
    Zprime_pinv = np.linalg.pinv(Zprime)
    
    # Prepare displacements vector in meters
    s = np.zeros(2 * nspots)
    for i in range(nspots):
        s[i] = dx[i] * cfg["camera_pixsize"]
        s[i + nspots] = dy[i] * cfg["camera_pixsize"]
        
    a_m = Zprime_pinv.dot(s)
    
    # Convert to radians: a = (1/f) * a_m * (2*pi / lambda)
    coeffs_rad = (1.0 / cfg["flength"]) * a_m * (2.0 * np.pi / cfg["wavelength"])
    return coeffs_rad, Zprime

def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation script running on device: {device}")
    
    # Paths setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "system.conf")
    spots_csv = os.path.join(base_dir, "results", "reference_centroids_c.csv")
    deviations_csv = os.path.join(base_dir, "results", "spot_deviations_c.csv")
    dataset_path = os.path.join(base_dir, "data_ai", "dataset.npz")
    
    # Load system config
    cfg = load_system_config(config_path)
    spots_df = pd.read_csv(spots_csv)
    nspots = len(spots_df)
    zernike_nmax = int(cfg["zernike_nmax"])
    max_j = (zernike_nmax + 1) * (zernike_nmax + 2) // 2
    nmodes = max_j - 1
    
    print(f"System loaded: {nspots} sub-apertures, predicting {nmodes} Zernike modes.")
    
    # 1. Look for trained checkpoints (Kaggle or Local fallback)
    checkpoint_dir_kaggle = os.path.join(base_dir, "ml_checkpoints", "kaggle")
    checkpoint_dir_local = os.path.join(base_dir, "ml_checkpoints", "local")
    
    mlp_path = os.path.join(checkpoint_dir_kaggle, "best_mlp.pt")
    if not os.path.exists(mlp_path):
        mlp_path = os.path.join(checkpoint_dir_local, "best_mlp.pt")
        
    cnn_path = os.path.join(checkpoint_dir_kaggle, "best_cnn.pt")
    if not os.path.exists(cnn_path):
        cnn_path = os.path.join(checkpoint_dir_local, "best_cnn.pt")
        
    print(f"MLP weights: {mlp_path}")
    print(f"CNN weights: {cnn_path}")
    
    models_loaded = True
    try:
        # Load MLP
        mlp_model = WavefrontMLP(input_dim=nspots * 2, output_dim=nmodes).to(device)
        mlp_ckpt = torch.load(mlp_path, map_location=device)
        mlp_model.load_state_dict(mlp_ckpt['model_state_dict'])
        mlp_model.eval()
        
        # Load CNN
        cnn_model = WavefrontCNN(output_dim=nmodes).to(device)
        cnn_ckpt = torch.load(cnn_path, map_location=device)
        cnn_model.load_state_dict(cnn_ckpt['model_state_dict'])
        cnn_model.eval()
        print("Successfully loaded both trained models!")
    except Exception as e:
        print(f"Warning: Could not load trained models: {e}")
        models_loaded = False
        
    # 2. Evaluate overall test performance if dataset.npz is available
    if models_loaded and os.path.exists(dataset_path):
        print(f"\nEvaluating performance on local dataset {dataset_path}...")
        dataset = WavefrontDataset(dataset_path, spots_csv, model_type='mlp')
        # Splitting manually to get the test loader
        train_len = int(0.8 * len(dataset))
        val_len = int(0.1 * len(dataset))
        test_len = len(dataset) - train_len - val_len
        _, _, test_set = torch.utils.data.random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )
        test_loader_mlp = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
        
        # Eval MLP
        mlp_loss = 0.0
        criterion = torch.nn.MSELoss()
        with torch.no_grad():
            for inputs, targets in test_loader_mlp:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = mlp_model(inputs)
                mlp_loss += criterion(outputs, targets).item() * inputs.size(0)
        mlp_test_mse = mlp_loss / len(test_set)
        
        # Eval CNN
        dataset_cnn = WavefrontDataset(dataset_path, spots_csv, model_type='cnn')
        _, _, test_set_cnn = torch.utils.data.random_split(
            dataset_cnn, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )
        test_loader_cnn = torch.utils.data.DataLoader(test_set_cnn, batch_size=64, shuffle=False)
        
        cnn_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader_cnn:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = cnn_model(inputs)
                cnn_loss += criterion(outputs, targets).item() * inputs.size(0)
        cnn_test_mse = cnn_loss / len(test_set)
        
        print(f"  MLP Test MSE: {mlp_test_mse:.6f}")
        print(f"  CNN Test MSE: {cnn_test_mse:.6f}")
    elif models_loaded:
        print("\nNote: data_ai/dataset.npz not found; skipping test set evaluation.")
        
    # 3. Load actual laboratory frame aberrated displacements
    if os.path.exists(deviations_csv):
        print(f"\nProcessing real laboratory aberrated frame displacements ({deviations_csv})...")
        dev_df = pd.read_csv(deviations_csv)
        dx = dev_df["Delta_X"].values
        dy = dev_df["Delta_Y"].values
        
        # Classical Modal Solver (Python check)
        a_classical, _ = compute_classical_zernike(dx, dy, cfg, spots_df)
        
        # ML Inference
        if models_loaded:
            # MLP Inference
            disp_vector = np.concatenate([dx, dy])
            disp_tensor = torch.tensor(disp_vector, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_mlp = mlp_model(disp_tensor).cpu().squeeze().numpy()
                
            # CNN Inference
            # Recreate spatial grid matching train.py
            # Pitch estimate
            mean_cx = spots_df["ref_cx"].mean()
            mean_cy = spots_df["ref_cy"].mean()
            dists = []
            for i in range(len(spots_df)):
                dx_pts = spots_df["ref_cx"].values - spots_df["ref_cx"].values[i]
                dy_pts = spots_df["ref_cy"].values - spots_df["ref_cy"].values[i]
                d = np.hypot(dx_pts, dy_pts)
                d = d[d > 1e-3]
                if len(d) > 0:
                    dists.append(d.min())
            pitch_px = np.mean(dists) if len(dists) > 0 else 40.1
            
            u_coords = np.round((spots_df["ref_cx"].values - mean_cx) / pitch_px).astype(int)
            v_coords = np.round((spots_df["ref_cy"].values - mean_cy) / pitch_px).astype(int)
            
            u_min, u_max = u_coords.min(), u_coords.max()
            v_min, v_max = v_coords.min(), v_coords.max()
            grid_w = int(u_max - u_min + 1)
            grid_h = int(v_max - v_min + 1)
            u_offset = int(-u_min)
            v_offset = int(-v_min)
            
            grid = torch.zeros((1, 2, grid_h, grid_w), dtype=torch.float32)
            for k in range(nspots):
                row = v_coords[k] + v_offset
                col = u_coords[k] + u_offset
                grid[0, 0, row, col] = dx[k]
                grid[0, 1, row, col] = dy[k]
                
            with torch.no_grad():
                pred_cnn = cnn_model(grid.to(device)).cpu().squeeze().numpy()
        
        # Display side-by-side comparison table
        print("\n" + "="*80)
        print("                     WAVEFRONT RECONSTRUCTION COMPARISONS")
        print("="*80)
        if models_loaded:
            print(f" Noll ID | Radial (n,m) | Classical (rad) | MLP Pred (rad)  | CNN Pred (rad)")
            print(f" --------|--------------|-----------------|-----------------|----------------")
            for idx in range(nmodes):
                j = idx + 2
                n, m = noll_to_nm(j)
                print(f"   {j:5d} |    ({n:2d},{m:2d})   |   {a_classical[idx]:+11.6f}   |   {pred_mlp[idx]:+11.6f}   |   {pred_cnn[idx]:+11.6f}")
        else:
            print(f" Noll ID | Radial (n,m) | Classical (rad)")
            print(f" --------|--------------|-----------------")
            for idx in range(nmodes):
                j = idx + 2
                n, m = noll_to_nm(j)
                print(f"   {j:5d} |    ({n:2d},{m:2d})   |   {a_classical[idx]:+11.6f}")
        print("="*80)
        
        if models_loaded:
            # Calculate correlation metrics
            corr_mlp = np.corrcoef(a_classical, pred_mlp)[0, 1]
            corr_cnn = np.corrcoef(a_classical, pred_cnn)[0, 1]
            print(f"Pearson Correlation (Classical vs MLP): {corr_mlp:.4f}")
            print(f"Pearson Correlation (Classical vs CNN): {corr_cnn:.4f}")
            
    else:
        print(f"Error: {deviations_csv} not found. Please calibrate first using C pipeline.")

if __name__ == "__main__":
    main()
