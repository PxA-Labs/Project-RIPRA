# ml/baseline_comparison.py - Phase 8.1: Baseline Comparison across all 4 methods
import os, sys, argparse, time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from models import WavefrontMLP, WavefrontCNN
from train import WavefrontDataset
from evaluate_inference import compute_classical_zernike, load_system_config, noll_to_nm

def compute_zonal_wavefront(dx, dy, spots_df, cfg):
    """Zonal reconstruction: stitch local slopes into phase via least-squares."""
    nspots = len(spots_df)
    pixsize = cfg["camera_pixsize"]
    flength = cfg["flength"]
    wavelength = cfg["wavelength"]
    pupil_r = cfg["pupil_radius"]
    mean_cx = spots_df["ref_cx"].mean()
    mean_cy = spots_df["ref_cy"].mean()
    A = np.zeros((2 * nspots, nspots))
    for i in range(nspots):
        A[i, i] = 1.0
        A[i + nspots, i] = 1.0
    s = np.zeros(2 * nspots)
    for i in range(nspots):
        s[i] = dx[i] * pixsize
        s[i + nspots] = dy[i] * pixsize
    phase = np.linalg.lstsq(A, s, rcond=None)[0]
    conv = (2.0 * np.pi / wavelength) / flength
    return phase * conv

def compute_strehl(coeffs_rad):
    """Estimate Strehl ratio from Marechal approximation."""
    var_wf = np.var(coeffs_rad)
    return np.exp(-var_wf)

def phase_rms(coeffs_rad):
    return np.sqrt(np.mean(coeffs_rad**2))

def phase_pv(coeffs_rad):
    return coeffs_rad.max() - coeffs_rad.min()

def evaluate(model, device, loader, model_type='mlp', nspots=None):
    preds, targets_list = [], []
    model.eval()
    with torch.no_grad():
        for inputs, tgt in loader:
            inputs, tgt = inputs.to(device), tgt.to(device)
            if model_type == 'cnn':
                pass
            outputs = model(inputs)
            preds.append(outputs.cpu().numpy())
            targets_list.append(tgt.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    return preds, targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true', help='Save comparison plots')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Baseline Comparison on device: {device}")
    print("=" * 80)

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
    print(f"System: {nspots} sub-apertures, {nmodes} Zernike modes")

    # Load dataset
    print("\nLoading dataset...")
    data = np.load(dataset_path)
    displacements = data['displacements']
    coefficients = data['coefficients']
    D_r0 = data['D_r0']

    n_total = len(displacements)
    train_len = int(0.8 * n_total)
    val_len = int(0.1 * n_total)
    test_len = n_total - train_len - val_len

    rng = np.random.RandomState(42)
    idx = rng.permutation(n_total)
    test_idx = idx[train_len + val_len:]
    test_disp = displacements[test_idx]
    test_coeff = coefficients[test_idx]
    print(f"Test set size: {len(test_idx)} frames")

    # Load ML models
    checkpoint_dir = os.path.join(base_dir, "ml_checkpoints", "kaggle")
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = os.path.join(base_dir, "ml_checkpoints", "local")

    mlp_path = os.path.join(checkpoint_dir, "best_mlp.pt")
    cnn_path = os.path.join(checkpoint_dir, "best_cnn.pt")

    mlp_model = WavefrontMLP(input_dim=nspots * 2, output_dim=nmodes).to(device)
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=device)['model_state_dict'])
    mlp_model.eval()

    cnn_model = WavefrontCNN(output_dim=nmodes).to(device)
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device)['model_state_dict'])
    cnn_model.eval()
    print("ML models loaded successfully")

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

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON ON TEST SET")
    print("=" * 80)

    methods = {}

    # 1. Zonal reconstruction (first 500 frames, too slow for all)
    print("\n[1/4] Zonal reconstruction on 100 test frames...")
    zonal_preds = []
    zonal_times = []
    for i in range(min(100, len(test_disp))):
        dx = test_disp[i, :nspots]
        dy = test_disp[i, nspots:]
        t0 = time.perf_counter()
        z_phase = compute_zonal_wavefront(dx, dy, spots_df, cfg)
        zonal_times.append(time.perf_counter() - t0)
        zonal_preds.append(z_phase)
    zonal_preds = np.array(zonal_preds)
    zonal_coeff_test = test_coeff[:len(zonal_preds), :nmodes]
    methods['Zonal'] = {
        'preds': zonal_preds,
        'targets': zonal_coeff_test,
        'time': np.mean(zonal_times),
        'color': '#ff6b6b'
    }
    print(f"  Zonal done: {len(zonal_preds)} frames, {np.mean(zonal_times)*1000:.2f} ms/frame")

    # Precompute Zprime pseudo-inverse for Modal (doesn't depend on displacements)
    print("\n[2/4] Classical Modal reconstruction (precomputing Z' matrix)...")
    dummy_dx = test_disp[0, :nspots]
    dummy_dy = test_disp[0, nspots:]
    _, Zprime = compute_classical_zernike(dummy_dx, dummy_dy, cfg, spots_df)
    Zprime_pinv = np.linalg.pinv(Zprime)

    pixsize = cfg["camera_pixsize"]
    flength = cfg["flength"]
    wavelength = cfg["wavelength"]
    conv = (2.0 * np.pi / wavelength) / flength

    modal_preds = []
    modal_times = []
    n_modal = min(300, len(test_disp))
    for i in range(n_modal):
        dx = test_disp[i, :nspots]
        dy = test_disp[i, nspots:]
        s = np.zeros(2 * nspots)
        for k in range(nspots):
            s[k] = dx[k] * pixsize
            s[k + nspots] = dy[k] * pixsize
        t0 = time.perf_counter()
        a_m = Zprime_pinv.dot(s)
        a_cls = a_m * conv
        modal_times.append(time.perf_counter() - t0)
        modal_preds.append(a_cls)
    modal_preds = np.array(modal_preds)
    n_modal_actual = len(modal_preds)
    methods['Modal'] = {
        'preds': modal_preds,
        'targets': test_coeff[:n_modal, :nmodes],
        'time': np.mean(modal_times),
        'color': '#ffd93d'
    }
    print(f"  Modal done: {n_modal_actual} frames, {np.mean(modal_times)*1000:.2f} ms/frame")

    # 3. MLP
    print("\n[3/4] MLP inference...")
    mlp_preds = []
    mlp_times = []
    n_mlp = min(300, len(test_disp))
    with torch.no_grad():
        for i in range(n_mlp):
            dv = np.concatenate([test_disp[i, :nspots], test_disp[i, nspots:]])
            inp = torch.tensor(dv, dtype=torch.float32).unsqueeze(0).to(device)
            t0 = time.perf_counter()
            out = mlp_model(inp).cpu().numpy().flatten()
            mlp_times.append(time.perf_counter() - t0)
            mlp_preds.append(out)
    mlp_preds = np.array(mlp_preds)
    methods['MLP'] = {
        'preds': mlp_preds,
        'targets': test_coeff[:n_mlp, :nmodes],
        'time': np.mean(mlp_times),
        'color': '#6bcbff'
    }
    print(f"  MLP done: {len(mlp_preds)} frames, {np.mean(mlp_times)*1000:.2f} ms/frame")

    # 4. CNN
    print("\n[4/4] CNN inference...")
    cnn_preds = []
    cnn_times = []
    n_cnn = min(300, len(test_disp))
    with torch.no_grad():
        for i in range(n_cnn):
            dx = test_disp[i, :nspots]
            dy = test_disp[i, nspots:]
            grid = np.zeros((1, 2, grid_h, grid_w), dtype=np.float32)
            for k in range(nspots):
                row = v_coords[k] + v_offset
                col = u_coords[k] + u_offset
                grid[0, 0, row, col] = dx[k]
                grid[0, 1, row, col] = dy[k]
            inp = torch.tensor(grid).to(device)
            t0 = time.perf_counter()
            out = cnn_model(inp).cpu().numpy().flatten()
            cnn_times.append(time.perf_counter() - t0)
            cnn_preds.append(out)
    cnn_preds = np.array(cnn_preds)
    methods['CNN'] = {
        'preds': cnn_preds,
        'targets': test_coeff[:n_cnn, :nmodes],
        'time': np.mean(cnn_times),
        'color': '#51cf66'
    }
    print(f"  CNN done: {len(cnn_preds)} frames, {np.mean(cnn_times)*1000:.2f} ms/frame")

    # Compute metrics per method (skip Zonal - different output space)
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    header = f"{'Method':<12} {'RMSE':<12} {'Pearson r':<12} {'Strehl':<12} {'PTV':<14} {'Latency':<12}"
    print(header)
    print("-" * 72)

    metrics_rec = {}
    skip_methods = {'Zonal'}
    for name, m in methods.items():
        if name in skip_methods:
            continue
        p, t = m['preds'], m['targets']
        if p.shape != t.shape:
            min_n = min(p.shape[0], t.shape[0])
            p, t = p[:min_n], t[:min_n]
        rmse = np.sqrt(np.mean((p - t)**2))
        corr_vals = [np.corrcoef(p[:, j], t[:, j])[0, 1] if np.std(p[:, j]) > 1e-9 and np.std(t[:, j]) > 1e-9 else 0 for j in range(p.shape[1])]
        corr = np.mean(corr_vals)
        strehl = np.mean([compute_strehl(p[j]) for j in range(p.shape[0])])
        ptv_vals = [phase_pv(p[j]) - phase_pv(t[j]) for j in range(p.shape[0])]
        ptv = np.mean(np.abs(ptv_vals))
        lat = m['time'] * 1000
        print(f"{name:<12} {rmse:<12.6f} {corr:<12.4f} {strehl:<12.4f} {ptv:<14.6f} {lat:<12.3f}")
        metrics_rec[name] = {'rmse': rmse, 'corr': corr, 'strehl': strehl, 'ptv': ptv, 'latency_ms': lat}

    # Save metrics to CSV
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    metrics_df = pd.DataFrame(metrics_rec).T
    metrics_df.to_csv(os.path.join(base_dir, "results", "baseline_metrics.csv"))
    print(f"\nMetrics saved to results/baseline_metrics.csv")

    # Visualization
    if args.visualize:
        os.makedirs(os.path.join(base_dir, "visualizations"), exist_ok=True)
        print("\nGenerating visualizations...")

        # 1. RMSE bar chart
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.patch.set_facecolor('#1a1a2e')
        for ax in axes.flat:
            ax.set_facecolor('#16213e')

        names = [n for n in methods if n not in skip_methods]
        rmse_vals = [metrics_rec[n]['rmse'] for n in names]
        corr_vals = [metrics_rec[n]['corr'] for n in names]
        strehl_vals = [metrics_rec[n]['strehl'] for n in names]
        ptv_vals = [metrics_rec[n]['ptv'] for n in names]
        lat_vals = [metrics_rec[n]['latency_ms'] for n in names]
        colors = [methods[n]['color'] for n in names]

        ax = axes[0, 0]
        bars = ax.bar(names, rmse_vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title('RMSE (rad)', color='white')
        ax.set_ylabel('RMSE', color='white')
        for b, v in zip(bars, rmse_vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{v:.5f}', ha='center', va='bottom', color='white', fontsize=9)
        ax.tick_params(colors='white')

        ax = axes[0, 1]
        bars = ax.bar(names, corr_vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title('Mean Pearson Correlation', color='white')
        ax.set_ylabel('r', color='white')
        for b, v in zip(bars, corr_vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{v:.4f}', ha='center', va='bottom', color='white', fontsize=9)
        ax.tick_params(colors='white')

        ax = axes[0, 2]
        bars = ax.bar(names, strehl_vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title('Mean Strehl Ratio', color='white')
        ax.set_ylabel('Strehl', color='white')
        for b, v in zip(bars, strehl_vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{v:.4f}', ha='center', va='bottom', color='white', fontsize=9)
        axes[0, 2].set_ylim(0, 1)
        ax.tick_params(colors='white')

        ax = axes[1, 0]
        bars = ax.bar(names, ptv_vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title('PTV Error (rad)', color='white')
        ax.set_ylabel('|Delta PTV|', color='white')
        for b, v in zip(bars, ptv_vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{v:.5f}', ha='center', va='bottom', color='white', fontsize=9)
        ax.tick_params(colors='white')

        ax = axes[1, 1]
        bars = ax.bar(names, lat_vals, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_title('Per-Frame Latency (ms)', color='white')
        ax.set_ylabel('ms', color='white')
        for b, v in zip(bars, lat_vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{v:.3f}', ha='center', va='bottom', color='white', fontsize=9)
        ax.tick_params(colors='white')

        # 2. Modal vs MLP vs CNN scatter (1st 100 frames, Zernike #2)
        ax = axes[1, 2]
        m_p = methods['Modal']['preds'][:100, 0]
        m_t = methods['Modal']['targets'][:100, 0]
        mlp_p = methods['MLP']['preds'][:100, 0]
        mlp_t = methods['MLP']['targets'][:100, 0]
        cnn_p = methods['CNN']['preds'][:100, 0]
        cnn_t = methods['CNN']['targets'][:100, 0]
        ax.scatter(m_t, m_p, alpha=0.5, label='Modal', color=methods['Modal']['color'], s=10)
        ax.scatter(mlp_t, mlp_p, alpha=0.5, label='MLP', color=methods['MLP']['color'], s=10)
        ax.scatter(cnn_t, cnn_p, alpha=0.5, label='CNN', color=methods['CNN']['color'], s=10)
        lims = [min(m_t.min(), mlp_t.min(), cnn_t.min()), max(m_t.max(), mlp_t.max(), cnn_t.max())]
        ax.plot(lims, lims, '--', color='gray', alpha=0.5)
        ax.set_title('True vs Predicted (Zernike #2)', color='white')
        ax.set_xlabel('True (rad)', color='white')
        ax.set_ylabel('Predicted (rad)', color='white')
        ax.legend()
        ax.tick_params(colors='white')

        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "visualizations", "baseline_comparison.png"), dpi=150, facecolor='#1a1a2e')
        plt.close()
        print("Saved: visualizations/baseline_comparison.png")

    print("\nPhase 8.1 Baseline Comparison complete.\n")

if __name__ == "__main__":
    main()
