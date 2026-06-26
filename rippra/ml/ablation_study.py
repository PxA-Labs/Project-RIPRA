# ml/ablation_study.py - Phase 8.3: Ablation Study
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from models import WavefrontMLP, WavefrontCNN
from sequence_models import WavefrontLSTM, TurbulenceClassifierLSTM
from evaluate_inference import load_system_config
from train import WavefrontDataset

def compute_rmse(preds, targets):
    return np.sqrt(np.mean((preds - targets)**2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_test', type=int, default=500, help='Test frames')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ablation Study on device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "system.conf")
    spots_csv = os.path.join(base_dir, "results", "reference_centroids_c.csv")
    dataset_path = os.path.join(base_dir, "data_ai", "dataset.npz")
    checkpoint_dir = os.path.join(base_dir, "ml_checkpoints", "kaggle")
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = os.path.join(base_dir, "ml_checkpoints", "local")

    cfg = load_system_config(config_path)
    spots_df = pd.read_csv(spots_csv)
    nspots = len(spots_df)
    zernike_nmax = int(cfg["zernike_nmax"])
    max_j = (zernike_nmax + 1) * (zernike_nmax + 2) // 2
    nmodes = max_j - 1

    data = np.load(dataset_path)
    displacements = data['displacements']
    coefficients = data['coefficients']
    n_total = len(displacements)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n_total)
    test_idx = idx[int(0.9 * n_total):]
    test_disp = displacements[test_idx][:args.n_test]
    test_coeff = coefficients[test_idx][:args.n_test]
    print(f"Using {len(test_disp)} test frames, {nmodes} output modes.")

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
    print(f"CNN grid: {grid_h}x{grid_w}")

    results = {}

    # A. MLP Ablation: hidden layer width (single hidden layer)
    print("\n[A] MLP Hidden Width Ablation")
    mlp_widths = [64, 128, 256, 512, 1024]
    for w in mlp_widths:
        model = WavefrontMLP(input_dim=nspots*2, hidden_dims=[w], output_dim=nmodes).to(device)
        preds = []
        times = []
        with torch.no_grad():
            for i in range(len(test_disp)):
                dv = np.concatenate([test_disp[i, :nspots], test_disp[i, nspots:]])
                inp = torch.tensor(dv, dtype=torch.float32).unsqueeze(0).to(device)
                t0 = time.perf_counter()
                out = model(inp).cpu().numpy().flatten()
                times.append(time.perf_counter() - t0)
                preds.append(out)
        preds = np.array(preds)
        rmse = compute_rmse(preds, test_coeff[:, :nmodes])
        lat = np.mean(times) * 1000
        params = sum(p.numel() for p in model.parameters())
        results[f'MLP_w={w}'] = {'rmse': rmse, 'latency_ms': lat, 'params': params, 'group': 'MLP'}
        print(f"  width={w:5d} | rmse={rmse:.6f} | latency={lat:.4f}ms | params={params:,}")

    # B. MLP Ablation: depth (layers)
    print("\n[B] MLP Depth Ablation")
    mlp_depths = [1, 2, 3, 4, 6]
    for d in mlp_depths:
        layers = []
        in_dim = nspots * 2
        for _ in range(d):
            layers.append(nn.Linear(in_dim, 256))
            layers.append(nn.ReLU())
            in_dim = 256
        layers.append(nn.Linear(256, nmodes))
        model = nn.Sequential(*layers).to(device)
        preds = []
        times = []
        with torch.no_grad():
            for i in range(len(test_disp)):
                dv = np.concatenate([test_disp[i, :nspots], test_disp[i, nspots:]])
                inp = torch.tensor(dv, dtype=torch.float32).unsqueeze(0).to(device)
                t0 = time.perf_counter()
                out = model(inp).cpu().numpy().flatten()
                times.append(time.perf_counter() - t0)
                preds.append(out)
        preds = np.array(preds)
        rmse = compute_rmse(preds, test_coeff[:, :nmodes])
        lat = np.mean(times) * 1000
        params = sum(p.numel() for p in model.parameters())
        results[f'MLP_depth={d}'] = {'rmse': rmse, 'latency_ms': lat, 'params': params, 'group': 'MLP'}
        print(f"  depth={d:5d} | rmse={rmse:.6f} | latency={lat:.4f}ms | params={params:,}")

    # C. CNN Ablation: grid resolution (simulated via stride)
    print("\n[C] CNN Grid Resolution Ablation")
    cnn_model = WavefrontCNN(output_dim=nmodes).to(device)
    ckpt = torch.load(os.path.join(checkpoint_dir, "best_cnn.pt"), map_location=device)
    cnn_model.load_state_dict(ckpt['model_state_dict'])
    cnn_model.eval()

    # Test original CNN
    preds = []
    times = []
    with torch.no_grad():
        for i in range(len(test_disp)):
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
            times.append(time.perf_counter() - t0)
            preds.append(out)
    preds = np.array(preds)
    rmse = compute_rmse(preds, test_coeff[:, :nmodes])
    lat = np.mean(times) * 1000
    params = sum(p.numel() for p in cnn_model.parameters())
    results['CNN_default'] = {'rmse': rmse, 'latency_ms': lat, 'params': params, 'group': 'CNN'}
    print(f"  default | rmse={rmse:.6f} | latency={lat:.4f}ms | params={params:,}")

    # D. LSTM Lookback Window Ablation
    print("\n[D] LSTM Lookback Window Ablation")
    lookbacks = [1, 3, 5, 10, 20]
    seq_len_total = 1000
    n_sequences = n_total // seq_len_total
    seq_coeff = coefficients[:n_sequences * seq_len_total].reshape(n_sequences, seq_len_total, -1)
    seq_disp = displacements[:n_sequences * seq_len_total].reshape(n_sequences, seq_len_total, -1)

    for lb in lookbacks:
        lstm_model = WavefrontLSTM(input_dim=nmodes, hidden_dim=128, output_dim=nmodes, num_layers=2).to(device)
        lstm_path = os.path.join(checkpoint_dir, "best_sequence_predict.pt")
        if os.path.exists(lstm_path):
            lstm_model.load_state_dict(torch.load(lstm_path, map_location=device)['model_state_dict'])
        lstm_model.eval()
        preds = []
        times = []
        with torch.no_grad():
            for seq_idx in range(min(5, n_sequences)):
                cseq = seq_coeff[seq_idx]
                for t in range(lb, len(cseq) - 1):
                    inp = torch.tensor(cseq[t-lb:t], dtype=torch.float32).unsqueeze(0).to(device)
                    t0 = time.perf_counter()
                    out = lstm_model(inp).cpu().numpy().flatten()
                    times.append(time.perf_counter() - t0)
                    preds.append(out)
        preds = np.array(preds)
        target = seq_coeff[:5, lb:-1, :nmodes].reshape(-1, nmodes)[:len(preds)]
        if len(preds) > 0 and preds.shape == target.shape:
            rmse = compute_rmse(preds, target)
            lat = np.mean(times) * 1000
            params = sum(p.numel() for p in lstm_model.parameters())
            results[f'LSTM_lb={lb}'] = {'rmse': rmse, 'latency_ms': lat, 'params': params, 'group': 'LSTM'}
            print(f"  lookback={lb:2d} | rmse={rmse:.6f} | latency={lat:.4f}ms | params={params:,}")

    # Save results
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(base_dir, "results", "ablation_results.csv"))
    print("\nResults saved to results/ablation_results.csv")

    # Visualization
    if args.visualize:
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join(base_dir, "visualizations"), exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a2e')
        for ax in axes.flat:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        # MLP width vs RMSE & latency
        ax = axes[0, 0]
        mlp_w_data = {k: v for k, v in results.items() if k.startswith('MLP_w')}
        widths = [int(k.split('=')[1]) for k in mlp_w_data]
        rmses = [v['rmse'] for v in mlp_w_data.values()]
        lats = [v['latency_ms'] for v in mlp_w_data.values()]
        ax2 = ax.twinx()
        ax.plot(widths, rmses, 'o-', color='#ff6b6b', linewidth=2, label='RMSE')
        ax2.plot(widths, lats, 's-', color='#6bcbff', linewidth=2, label='Latency')
        ax.set_xlabel('MLP Hidden Width')
        ax.set_ylabel('RMSE (rad)', color='#ff6b6b')
        ax2.set_ylabel('Latency (ms)', color='#6bcbff')
        ax.set_title('MLP Width vs Accuracy & Speed')
        ax.tick_params(axis='y', colors='#ff6b6b')
        ax2.tick_params(axis='y', colors='#6bcbff')
        ax.set_xscale('log', base=2)

        # MLP depth vs RMSE & latency
        ax = axes[0, 1]
        mlp_d_data = {k: v for k, v in results.items() if k.startswith('MLP_depth')}
        depths = [int(k.split('=')[1]) for k in mlp_d_data]
        rmses = [v['rmse'] for v in mlp_d_data.values()]
        lats = [v['latency_ms'] for v in mlp_d_data.values()]
        ax2 = ax.twinx()
        ax.plot(depths, rmses, 'o-', color='#ff6b6b', linewidth=2, label='RMSE')
        ax2.plot(depths, lats, 's-', color='#6bcbff', linewidth=2, label='Latency')
        ax.set_xlabel('MLP Depth (layers)')
        ax.set_ylabel('RMSE (rad)', color='#ff6b6b')
        ax2.set_ylabel('Latency (ms)', color='#6bcbff')
        ax.set_title('MLP Depth vs Accuracy & Speed')
        ax.tick_params(axis='y', colors='#ff6b6b')
        ax2.tick_params(axis='y', colors='#6bcbff')

        # LSTM lookback vs RMSE & latency
        ax = axes[1, 0]
        lstm_data = {k: v for k, v in results.items() if k.startswith('LSTM')}
        lbs = [int(k.split('=')[1]) for k in lstm_data]
        lbs_sorted, rmses_sorted, lats_sorted = zip(*sorted(zip(lbs, [v['rmse'] for v in lstm_data.values()], [v['latency_ms'] for v in lstm_data.values()])))
        ax2 = ax.twinx()
        ax.plot(lbs_sorted, rmses_sorted, 'o-', color='#ffd93d', linewidth=2, label='RMSE')
        ax2.plot(lbs_sorted, lats_sorted, 's-', color='#6bcbff', linewidth=2, label='Latency')
        ax.set_xlabel('LSTM Lookback Window')
        ax.set_ylabel('RMSE (rad)', color='#ffd93d')
        ax2.set_ylabel('Latency (ms)', color='#6bcbff')
        ax.set_title('LSTM Lookback vs Accuracy & Speed')
        ax.tick_params(axis='y', colors='#ffd93d')
        ax2.tick_params(axis='y', colors='#6bcbff')

        # Parameter count comparison
        ax = axes[1, 1]
        groups = {'MLP': {}, 'CNN': {}, 'LSTM': {}}
        for k, v in results.items():
            g = v['group']
            groups[g][k] = v
        all_labels, all_params, all_rmses = [], [], []
        all_colors = []
        color_map = {'MLP': '#ff6b6b', 'CNN': '#51cf66', 'LSTM': '#ffd93d'}
        for g in ['MLP', 'CNN', 'LSTM']:
            for k, v in sorted(groups[g].items()):
                all_labels.append(k)
                all_params.append(v['params'])
                all_rmses.append(v['rmse'])
                all_colors.append(color_map[g])
        scatter = ax.scatter(all_params, all_rmses, c=all_colors, s=80, edgecolors='white', linewidth=0.5)
        for i, lbl in enumerate(all_labels):
            ax.annotate(lbl, (all_params[i], all_rmses[i]), fontsize=7, color='white', ha='left', va='bottom', rotation=45)
        ax.set_xscale('log')
        ax.set_xlabel('Parameters')
        ax.set_ylabel('RMSE (rad)')
        ax.set_title('Parameters vs Accuracy')

        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "visualizations", "ablation_study.png"), dpi=150, facecolor='#1a1a2e')
        plt.close()
        print("Saved: visualizations/ablation_study.png")

    print("\nPhase 8.3 Ablation Study complete.\n")

if __name__ == "__main__":
    main()
