# ml/performance_profile.py - Phase 8.4: Performance Benchmarking (latency, memory, jitter)
import os, sys, time, argparse, gc
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from models import WavefrontMLP, WavefrontCNN
from evaluate_inference import compute_classical_zernike, load_system_config

def estimate_memory(obj):
    """Rough estimate of object memory size in MB."""
    if isinstance(obj, torch.nn.Module):
        param_bytes = sum(p.numel() * p.element_size() for p in obj.parameters())
        buf_bytes = sum(b.numel() * b.element_size() for b in obj.buffers())
        return (param_bytes + buf_bytes) / (1024 * 1024)
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_warmup', type=int, default=50, help='Warmup iterations')
    parser.add_argument('--n_bench', type=int, default=1000, help='Benchmark iterations')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Performance Profiling on device: {device}")
    print("=" * 80)

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
    test_disp = displacements[test_idx]
    test_coeff = coefficients[test_idx]

    # CNN grid
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

    # Load models
    mlp_model = WavefrontMLP(input_dim=nspots*2, output_dim=nmodes).to(device)
    mlp_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_mlp.pt"), map_location=device)['model_state_dict'])
    mlp_model.eval()

    cnn_model = WavefrontCNN(output_dim=nmodes).to(device)
    cnn_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_cnn.pt"), map_location=device)['model_state_dict'])
    cnn_model.eval()

    # Cache model memory & precompute Zprime_pinv
    mlp_mem = estimate_memory(mlp_model)
    cnn_mem = estimate_memory(cnn_model)
    dummy_dx, dummy_dy = test_disp[0, :nspots], test_disp[0, nspots:]
    _, Zprime = compute_classical_zernike(dummy_dx, dummy_dy, cfg, spots_df)
    Zprime_pinv = np.linalg.pinv(Zprime)
    pixsize = cfg["camera_pixsize"]
    flength = cfg["flength"]
    wavelength = cfg["wavelength"]
    conv = (2.0 * np.pi / wavelength) / flength
    Zprime_mem = Zprime.nbytes / (1024*1024)
    pm = Zprime_pinv
    pm_mem = pm.nbytes / (1024*1024)
    modal_mem = Zprime_mem + pm_mem

    print("\n--- Memory Footprint ---")
    print(f"  Modal reconstruction matrices: {modal_mem:.2f} MB (Z' {Zprime_mem:.2f} MB + pinv {pm_mem:.2f} MB)")
    print(f"  MLP model parameters:         {mlp_mem:.2f} MB")
    print(f"  CNN model parameters:         {cnn_mem:.2f} MB")

    # Benchmark each method
    n_warmup = args.n_warmup
    n_bench = args.n_bench
    results = {}

    def modal_fn(i):
        dx, dy = test_disp[i, :nspots], test_disp[i, nspots:]
        s = np.zeros(2 * nspots)
        for k in range(nspots):
            s[k] = dx[k] * pixsize
            s[k + nspots] = dy[k] * pixsize
        return Zprime_pinv.dot(s) * conv

    def mlp_fn(i):
        dv = np.concatenate([test_disp[i, :nspots], test_disp[i, nspots:]]).astype(np.float32)
        inp = torch.tensor(dv, dtype=torch.float32).unsqueeze(0).to(device)
        return mlp_model(inp)

    for method_name, method_fn, n_avail in [
        ('Modal Classical', modal_fn, len(test_disp)),
        ('MLP Inference', mlp_fn, len(test_disp)),
    ]:
        print(f"\nBenchmarking {method_name}...")
        # Warmup
        for i in range(min(n_warmup, n_avail)):
            _ = method_fn(i)
        if device == 'cuda':
            torch.cuda.synchronize()
        # Timed run
        times = []
        n_iter = min(n_bench, n_avail)
        for i in range(n_iter):
            t0 = time.perf_counter()
            _ = method_fn(i)
            times.append(time.perf_counter() - t0)
        times_ms = np.array(times) * 1000
        results[method_name] = {
            'mean_ms': np.mean(times_ms),
            'std_ms': np.std(times_ms),
            'min_ms': np.min(times_ms),
            'max_ms': np.max(times_ms),
            'p50_ms': np.median(times_ms),
            'p95_ms': np.percentile(times_ms, 95),
            'p99_ms': np.percentile(times_ms, 99),
            'jitter_ms': np.std(times_ms),
        }
        print(f"  {n_iter} iterations")
        print(f"  Mean: {results[method_name]['mean_ms']:.4f} ms")
        print(f"  Std:  {results[method_name]['std_ms']:.4f} ms")
        print(f"  Min:  {results[method_name]['min_ms']:.4f} ms")
        print(f"  Max:  {results[method_name]['max_ms']:.4f} ms")
        print(f"  P50:  {results[method_name]['p50_ms']:.4f} ms")
        print(f"  P95:  {results[method_name]['p95_ms']:.4f} ms")
        print(f"  P99:  {results[method_name]['p99_ms']:.4f} ms")

    # CNN benchmark (separate due to grid construction)
    print(f"\nBenchmarking CNN Inference...")
    cnn_times = []
    n_iter = min(n_bench, len(test_disp))
    for i in range(min(n_warmup, len(test_disp))):
        dx, dy = test_disp[i, :nspots], test_disp[i, nspots:]
        grid = np.zeros((1, 2, grid_h, grid_w), dtype=np.float32)
        for k in range(nspots):
            row = v_coords[k] + v_offset
            col = u_coords[k] + u_offset
            grid[0, 0, row, col] = dx[k]
            grid[0, 1, row, col] = dy[k]
        _ = cnn_model(torch.tensor(grid).to(device))
    if device == 'cuda':
        torch.cuda.synchronize()
    for i in range(n_iter):
        dx, dy = test_disp[i, :nspots], test_disp[i, nspots:]
        grid = np.zeros((1, 2, grid_h, grid_w), dtype=np.float32)
        for k in range(nspots):
            row = v_coords[k] + v_offset
            col = u_coords[k] + u_offset
            grid[0, 0, row, col] = dx[k]
            grid[0, 1, row, col] = dy[k]
        t0 = time.perf_counter()
        _ = cnn_model(torch.tensor(grid).to(device))
        cnn_times.append(time.perf_counter() - t0)
    times_ms = np.array(cnn_times) * 1000
    results['CNN Inference'] = {
        'mean_ms': np.mean(times_ms),
        'std_ms': np.std(times_ms),
        'min_ms': np.min(times_ms),
        'max_ms': np.max(times_ms),
        'p50_ms': np.median(times_ms),
        'p95_ms': np.percentile(times_ms, 95),
        'p99_ms': np.percentile(times_ms, 99),
        'jitter_ms': np.std(times_ms),
    }
    print(f"  {n_iter} iterations")
    print(f"  Mean: {results['CNN Inference']['mean_ms']:.4f} ms")
    print(f"  Std:  {results['CNN Inference']['std_ms']:.4f} ms")
    print(f"  Min:  {results['CNN Inference']['min_ms']:.4f} ms")
    print(f"  Max:  {results['CNN Inference']['max_ms']:.4f} ms")
    print(f"  P50:  {results['CNN Inference']['p50_ms']:.4f} ms")
    print(f"  P95:  {results['CNN Inference']['p95_ms']:.4f} ms")
    print(f"  P99:  {results['CNN Inference']['p99_ms']:.4f} ms")

    # Save results
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    perf_df = pd.DataFrame(results).T
    perf_df.index.name = 'method'
    perf_df.to_csv(os.path.join(base_dir, "results", "performance_profile.csv"))

    # Save memory footprint
    mem_df = pd.DataFrame({
        'method': ['Modal', 'MLP', 'CNN'],
        'memory_mb': [modal_mem, mlp_mem, cnn_mem]
    })
    mem_df.set_index('method').to_csv(os.path.join(base_dir, "results", "memory_footprint.csv"))
    print(f"\nResults saved to results/performance_profile.csv, results/memory_footprint.csv")

    # Visualization
    if args.visualize:
        os.makedirs(os.path.join(base_dir, "visualizations"), exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a2e')
        for ax in axes.flat:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        # Latency bar chart with error bars
        ax = axes[0, 0]
        names = list(results.keys())
        means = [results[n]['mean_ms'] for n in names]
        stds = [results[n]['std_ms'] for n in names]
        colors = ['#ffd93d', '#6bcbff', '#51cf66']
        bars = ax.bar(names, means, yerr=stds, color=colors, edgecolor='white', linewidth=0.5, capsize=5)
        ax.set_ylabel('Mean Latency (ms)')
        ax.set_title('Per-Frame Latency (mean ± std)')
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{v:.3f}ms', ha='center', va='bottom', color='white', fontsize=9)

        # Jitter (std) bar chart
        ax = axes[0, 1]
        jitters = [results[n]['jitter_ms'] for n in names]
        bars = ax.bar(names, jitters, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_ylabel('Jitter σ (ms)')
        ax.set_title('Latency Jitter (Standard Deviation)')
        for b, v in zip(bars, jitters):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{v:.4f}ms', ha='center', va='bottom', color='white', fontsize=9)

        # Memory footprint
        ax = axes[1, 0]
        mem_names = ['Modal (Z\'+pinv)', 'MLP', 'CNN']
        mem_values = [modal_mem, mlp_mem, cnn_mem]
        bars = ax.bar(mem_names, mem_values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Model Memory Footprint')
        for b, v in zip(bars, mem_values):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{v:.3f} MB', ha='center', va='bottom', color='white', fontsize=9)

        # P50/P95/P99 comparison
        ax = axes[1, 1]
        x = np.arange(len(names))
        w = 0.25
        for i, (pct, lbl, clr) in enumerate([('p50_ms', 'P50', '#6bcbff'), ('p95_ms', 'P95', '#ffd93d'), ('p99_ms', 'P99', '#ff6b6b')]):
            vals = [results[n][pct] for n in names]
            ax.bar(x + i*w - w, vals, w, label=lbl, color=clr, edgecolor='white', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Percentiles')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "visualizations", "performance_profile.png"), dpi=150, facecolor='#1a1a2e')
        plt.close()
        print("Saved: visualizations/performance_profile.png")

    print("\nPhase 8.4 Performance Benchmarking complete.\n")

if __name__ == "__main__":
    main()
