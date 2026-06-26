# ml/noise_robustness.py - Phase 8.2: Noise & Robustness Testing
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from models import WavefrontMLP, WavefrontCNN
from evaluate_inference import compute_classical_zernike, load_system_config, noll_to_nm

def add_photon_noise(disp, alpha=1.0):
    """Simulate photon shot noise (Poisson). alpha scales photon count."""
    intensity = np.abs(disp) * alpha
    noisy = np.random.poisson(np.maximum(intensity, 0)) / max(alpha, 1e-9)
    return np.where(disp >= 0, noisy, -noisy)

def add_gaussian_noise(disp, sigma_px=0.1):
    """Add Gaussian readout noise."""
    return disp + np.random.randn(*disp.shape) * sigma_px

def occlude_spots(disp, spots_df, frac=0.1):
    """Randomly zero out a fraction of spot measurements."""
    nspots = len(spots_df)
    n_occ = max(1, int(nspots * frac))
    occ_idx = np.random.choice(nspots, n_occ, replace=False)
    disp_copy = disp.copy()
    disp_copy[occ_idx] = 0.0
    disp_copy[nspots + occ_idx] = 0.0
    return disp_copy

def compute_metrics(preds, targets):
    if preds.shape != targets.shape:
        min_n = min(preds.shape[0], targets.shape[0])
        preds, targets = preds[:min_n], targets[:min_n]
    rmse = np.sqrt(np.mean((preds - targets)**2))
    corr_vals = []
    for j in range(preds.shape[1]):
        if np.std(preds[:, j]) > 1e-9 and np.std(targets[:, j]) > 1e-9:
            corr_vals.append(np.corrcoef(preds[:, j], targets[:, j])[0, 1])
    corr = np.mean(corr_vals) if corr_vals else 0
    return rmse, corr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_test', type=int, default=200, help='Number of test frames')
    parser.add_argument('--visualize', action='store_true', help='Save robustness plots')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Noise & Robustness Testing on device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "system.conf")
    spots_csv = os.path.join(base_dir, "results", "reference_centroids_c.csv")
    dataset_path = os.path.join(base_dir, "data_ai", "dataset.npz")

    cfg = load_system_config(config_path)
    spots_df = pd.read_csv(spots_csv)
    nspots = len(spots_df)
    zernike_nmax = int(cfg["zernike_nmax"])
    max_j = (zernike_nmax + 1) * (zernike_nmax + 2) // 2
    nmodes = max_j - 1

    # Load dataset and get test split
    data = np.load(dataset_path)
    displacements = data['displacements']
    coefficients = data['coefficients']
    n_total = len(displacements)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n_total)
    train_len = int(0.8 * n_total)
    val_len = int(0.1 * n_total)
    test_idx = idx[train_len + val_len:]
    test_disp_clean = displacements[test_idx][:args.n_test]
    test_coeff = coefficients[test_idx][:args.n_test]

    # Load ML models
    checkpoint_dir = os.path.join(base_dir, "ml_checkpoints", "kaggle")
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = os.path.join(base_dir, "ml_checkpoints", "local")
    mlp_model = WavefrontMLP(input_dim=nspots * 2, output_dim=nmodes).to(device)
    mlp_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_mlp.pt"), map_location=device)['model_state_dict'])
    mlp_model.eval()
    cnn_model = WavefrontCNN(output_dim=nmodes).to(device)
    cnn_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_cnn.pt"), map_location=device)['model_state_dict'])
    cnn_model.eval()
    print("Models loaded.")

    # Prepare CNN grid
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
    u_offset, v_offset = int(-u_min), int(-v_min)

    # Precompute Zprime_pinv for Modal
    dummy_dx = test_disp_clean[0, :nspots]
    dummy_dy = test_disp_clean[0, nspots:]
    _, Zprime = compute_classical_zernike(dummy_dx, dummy_dy, cfg, spots_df)
    Zprime_pinv = np.linalg.pinv(Zprime)
    pixsize = cfg["camera_pixsize"]
    flength = cfg["flength"]
    wavelength = cfg["wavelength"]
    conv = (2.0 * np.pi / wavelength) / flength

    def run_inference(disp_batch, method='modal'):
        """Run inference on a batch of displacements."""
        preds = []
        for i in range(len(disp_batch)):
            dx = disp_batch[i, :nspots]
            dy = disp_batch[i, nspots:]
            if method == 'modal':
                s = np.zeros(2 * nspots)
                for k in range(nspots):
                    s[k] = dx[k] * pixsize
                    s[k + nspots] = dy[k] * pixsize
                a_cls = Zprime_pinv.dot(s) * conv
                preds.append(a_cls)
            elif method == 'mlp':
                dv = np.concatenate([dx, dy])
                inp = torch.tensor(dv, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = mlp_model(inp).cpu().numpy().flatten()
                preds.append(out)
            elif method == 'cnn':
                grid = np.zeros((1, 2, grid_h, grid_w), dtype=np.float32)
                for k in range(nspots):
                    row = v_coords[k] + v_offset
                    col = u_coords[k] + u_offset
                    grid[0, 0, row, col] = dx[k]
                    grid[0, 1, row, col] = dy[k]
                inp = torch.tensor(grid).to(device)
                with torch.no_grad():
                    out = cnn_model(inp).cpu().numpy().flatten()
                preds.append(out)
        return np.array(preds)

    # Test 1: Gaussian readout noise
    print("\n[Test 1] Gaussian Readout Noise Sweep...")
    sigma_levels = np.logspace(-2, 0.5, 12)
    results_gauss = {m: {'rmse': [], 'corr': []} for m in ['modal', 'mlp', 'cnn']}
    for sigma in sigma_levels:
        disp_noisy = np.array([add_gaussian_noise(d, sigma) for d in test_disp_clean])
        for method in ['modal', 'mlp', 'cnn']:
            preds = run_inference(disp_noisy, method)
            rmse, corr = compute_metrics(preds, test_coeff[:, :nmodes])
            results_gauss[method]['rmse'].append(rmse)
            results_gauss[method]['corr'].append(corr)
        if sigma >= 0.01:
            print(f"  sigma={sigma:.4f}: modal rmse={results_gauss['modal']['rmse'][-1]:.5f}, mlp={results_gauss['mlp']['rmse'][-1]:.5f}, cnn={results_gauss['cnn']['rmse'][-1]:.5f}")

    # Test 2: Photon shot noise
    print("\n[Test 2] Photon Shot Noise Sweep...")
    gamma_levels = np.logspace(-2, 1.5, 12)
    results_photon = {m: {'rmse': [], 'corr': []} for m in ['modal', 'mlp', 'cnn']}
    for gamma in gamma_levels:
        disp_noisy = np.array([add_photon_noise(d, gamma) for d in test_disp_clean])
        for method in ['modal', 'mlp', 'cnn']:
            preds = run_inference(disp_noisy, method)
            rmse, corr = compute_metrics(preds, test_coeff[:, :nmodes])
            results_photon[method]['rmse'].append(rmse)
            results_photon[method]['corr'].append(corr)
        if gamma >= 0.01:
            print(f"  gamma={gamma:.4f}: modal rmse={results_photon['modal']['rmse'][-1]:.5f}, mlp={results_photon['mlp']['rmse'][-1]:.5f}, cnn={results_photon['cnn']['rmse'][-1]:.5f}")

    # Test 3: Spot occlusion
    print("\n[Test 3] Spot Occlusion Sweep...")
    occ_levels = np.linspace(0, 0.5, 11)
    results_occ = {m: {'rmse': [], 'corr': []} for m in ['modal', 'mlp', 'cnn']}
    for occ in occ_levels:
        rng_occ = np.random.RandomState(0)
        disp_occ = np.array([occlude_spots(d.copy(), spots_df, occ) for d in test_disp_clean])
        for method in ['modal', 'mlp', 'cnn']:
            preds = run_inference(disp_occ, method)
            rmse, corr = compute_metrics(preds, test_coeff[:, :nmodes])
            results_occ[method]['rmse'].append(rmse)
            results_occ[method]['corr'].append(corr)
        print(f"  occlusion={occ:.2f}: modal rmse={results_occ['modal']['rmse'][-1]:.5f}, mlp={results_occ['mlp']['rmse'][-1]:.5f}, cnn={results_occ['cnn']['rmse'][-1]:.5f}")

    # Save results as separate CSVs
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    gauss_df = pd.DataFrame({'sigma': sigma_levels, 'modal_rmse': results_gauss['modal']['rmse'],
                             'mlp_rmse': results_gauss['mlp']['rmse'], 'cnn_rmse': results_gauss['cnn']['rmse']})
    gauss_df.to_csv(os.path.join(base_dir, "results", "noise_gaussian.csv"), index=False)
    photon_df = pd.DataFrame({'gamma': gamma_levels, 'modal_rmse': results_photon['modal']['rmse'],
                              'mlp_rmse': results_photon['mlp']['rmse'], 'cnn_rmse': results_photon['cnn']['rmse']})
    photon_df.to_csv(os.path.join(base_dir, "results", "noise_photon.csv"), index=False)
    occ_df = pd.DataFrame({'occ_frac': occ_levels, 'modal_rmse': results_occ['modal']['rmse'],
                           'mlp_rmse': results_occ['mlp']['rmse'], 'cnn_rmse': results_occ['cnn']['rmse']})
    occ_df.to_csv(os.path.join(base_dir, "results", "noise_occlusion.csv"), index=False)
    print("\nResults saved to results/noise_gaussian.csv, noise_photon.csv, noise_occlusion.csv")

    # Visualization
    if args.visualize:
        os.makedirs(os.path.join(base_dir, "visualizations"), exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.patch.set_facecolor('#1a1a2e')
        for ax in axes:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        ax = axes[0]
        ax.plot(sigma_levels, results_gauss['modal']['rmse'], 'o-', color='#ffd93d', label='Modal', linewidth=2)
        ax.plot(sigma_levels, results_gauss['mlp']['rmse'], 's-', color='#6bcbff', label='MLP', linewidth=2)
        ax.plot(sigma_levels, results_gauss['cnn']['rmse'], '^-', color='#51cf66', label='CNN', linewidth=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Gaussian Noise σ (px)')
        ax.set_ylabel('RMSE (rad)')
        ax.set_title('Gaussian Readout Noise')
        ax.legend()
        ax.grid(True, alpha=0.2)

        ax = axes[1]
        ax.plot(gamma_levels, results_photon['modal']['rmse'], 'o-', color='#ffd93d', label='Modal', linewidth=2)
        ax.plot(gamma_levels, results_photon['mlp']['rmse'], 's-', color='#6bcbff', label='MLP', linewidth=2)
        ax.plot(gamma_levels, results_photon['cnn']['rmse'], '^-', color='#51cf66', label='CNN', linewidth=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Photon Count Scale γ')
        ax.set_ylabel('RMSE (rad)')
        ax.set_title('Photon Shot Noise')
        ax.legend()
        ax.grid(True, alpha=0.2)

        ax = axes[2]
        ax.plot(occ_levels, results_occ['modal']['rmse'], 'o-', color='#ffd93d', label='Modal', linewidth=2)
        ax.plot(occ_levels, results_occ['mlp']['rmse'], 's-', color='#6bcbff', label='MLP', linewidth=2)
        ax.plot(occ_levels, results_occ['cnn']['rmse'], '^-', color='#51cf66', label='CNN', linewidth=2)
        ax.set_xlabel('Spot Occlusion Fraction')
        ax.set_ylabel('RMSE (rad)')
        ax.set_title('Spot Occlusion Robustness')
        ax.legend()
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "visualizations", "noise_robustness.png"), dpi=150, facecolor='#1a1a2e')
        plt.close()
        print("Saved: visualizations/noise_robustness.png")

    print("\nPhase 8.2 Noise & Robustness Testing complete.\n")

if __name__ == "__main__":
    main()
