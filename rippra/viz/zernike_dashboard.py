# viz/zernike_dashboard.py - Zernike Coefficient Dashboard (Checkpoint 7.2)
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))
from evaluate_inference import load_system_config, compute_classical_zernike, noll_to_nm

class ZernikeDashboard:
    def __init__(self, results_dir=None, config_path=None):
        self.base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = results_dir or os.path.join(self.base, 'results')
        self.config_path = config_path or os.path.join(self.base, 'config', 'system.conf')
        self.out_dir = os.path.join(self.base, '..', 'visualizations')
        os.makedirs(self.out_dir, exist_ok=True)

        self.cfg = load_system_config(self.config_path)
        self.df_ref = pd.read_csv(os.path.join(self.results_dir, 'reference_centroids_c.csv'))
        self.df_dev = pd.read_csv(os.path.join(self.results_dir, 'spot_deviations_c.csv'))

    def compute_coeffs(self):
        coeffs, _ = compute_classical_zernike(
            self.df_dev['Delta_X'].values, self.df_dev['Delta_Y'].values,
            self.cfg, self.df_ref
        )
        return coeffs

    def plot_zernike_bar_chart(self, save=True):
        """7.2a: Modal weight distribution bar chart"""
        coeffs = self.compute_coeffs()
        zernike_nmax = int(self.cfg['zernike_nmax'])
        max_j = (zernike_nmax + 1) * (zernike_nmax + 2) // 2
        nmodes = max_j - 1

        labels = []
        mode_names = []
        for idx in range(nmodes):
            j = idx + 2
            n, m = noll_to_nm(j)
            names = {2: 'Tip', 3: 'Tilt', 4: 'Defocus', 5: 'Astig(3)', 6: 'Astig(4)',
                     7: 'Coma(5)', 8: 'Coma(6)', 9: 'Trefoil(7)', 10: 'Trefoil(8)'}
            name = names.get(j, f'Z{j}')
            labels.append(f'Z{j}\n{name}')
            mode_names.append(name)

        colors = ['#00ff88' if c >= 0 else '#ff4444' for c in coeffs]

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        bars = ax.bar(range(nmodes), coeffs, color=colors, alpha=0.85,
                       edgecolor='white', linewidth=0.5)
        ax.axhline(y=0, color='white', linewidth=0.8)
        ax.set_xticks(range(nmodes))
        ax.set_xticklabels(labels, color='white', fontsize=8, rotation=45, ha='right')
        ax.set_ylabel('Coefficient (radians)', color='white', fontsize=11)
        ax.set_title('Zernike Modal Weight Distribution', color='white',
                     fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        ax.grid(axis='y', alpha=0.2, color='gray')

        for bar, val in zip(bars, coeffs):
            y_pos = val + 0.02 * (1 if val >= 0 else -1)
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                    color='white', fontsize=7)

        if save:
            path = os.path.join(self.out_dir, 'zernike_bar_chart.png')
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Saved: {path}")
        return fig

    def plot_zernike_time_series(self, save=True):
        """7.2b: Real time-series tracking of low-order Zernike modes"""
        ts_path = os.path.join(self.results_dir, 'zernike_time_series.csv')
        if not os.path.exists(ts_path):
            print(f"  WARNING: {ts_path} not found. Generate with tools/generate_realistic_ts.py")
            return self._plot_synthetic_time_series(save)

        df = pd.read_csv(ts_path)
        n_frames = len(df)
        nmodes = len([c for c in df.columns if c.startswith('z')])

        # First 5 non-piston modes: z0=Tip(Z2), z1=Tilt(Z3), z2=Defocus(Z4), z3=Astig(Z5), z4=Astig(Z6)
        n_low = min(5, nmodes)
        labels = ['Tip (Z2)', 'Tilt (Z3)', 'Defocus (Z4)', 'Astig (Z5)', 'Astig (Z6)']
        colors = ['#00ff88', '#ff4444', '#4488ff', '#ffaa00', '#cc44ff']

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        frames = np.arange(n_frames)
        for m in range(n_low):
            ax.plot(frames, df[f'z{m}'].values, color=colors[m], label=labels[m],
                    linewidth=1.5, alpha=0.85)

        ax.set_xlabel('Frame', color='white', fontsize=11)
        ax.set_ylabel('Coefficient (radians)', color='white', fontsize=11)
        ax.set_title('Low-Order Zernike Modes Over Time', color='white',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.15, color='gray')

        if save:
            path = os.path.join(self.out_dir, 'zernike_time_series.png')
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Saved: {path}")
        return fig

    def _plot_synthetic_time_series(self, save=True):
        """Fallback synthetic time series if real data unavailable"""
        np.random.seed(42)
        n_frames = 500
        n_low = 5
        drift = np.zeros((n_frames, n_low))
        for m in range(n_low):
            x = np.cumsum(np.random.randn(n_frames) * 0.05)
            drift[:, m] = 0.1 * np.sin(np.linspace(0, 4*np.pi, n_frames) * (m+1)) + 0.02 * x

        labels = ['Tip (Z2)', 'Tilt (Z3)', 'Defocus (Z4)', 'Astig (Z5)', 'Astig (Z6)']
        colors = ['#00ff88', '#ff4444', '#4488ff', '#ffaa00', '#cc44ff']

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        time_ms = np.arange(n_frames)
        for m in range(n_low):
            ax.plot(time_ms, drift[:, m], color=colors[m], label=labels[m],
                    linewidth=1.5, alpha=0.85)

        ax.set_xlabel('Frame', color='white', fontsize=11)
        ax.set_ylabel('Coefficient (radians)', color='white', fontsize=11)
        ax.set_title('Low-Order Zernike Modes Over Time (SYNTHETIC)', color='white',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.15, color='gray')

        if save:
            path = os.path.join(self.out_dir, 'zernike_time_series.png')
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Saved: {path} (SYNTHETIC)")
        return fig

    def render_all(self):
        print("Rendering Zernike dashboard...")
        self.plot_zernike_bar_chart()
        self.plot_zernike_time_series()
        print("Done.")
