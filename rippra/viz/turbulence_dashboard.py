# viz/turbulence_dashboard.py - Turbulence Analytics (Checkpoint 7.3)
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))
from evaluate_inference import load_system_config, compute_classical_zernike

class TurbulenceDashboard:
    def __init__(self, results_dir=None, config_path=None):
        self.base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = results_dir or os.path.join(self.base, 'results')
        self.config_path = config_path or os.path.join(self.base, 'config', 'system.conf')
        self.out_dir = os.path.join(self.base, '..', 'visualizations')
        os.makedirs(self.out_dir, exist_ok=True)

        self.cfg = load_system_config(self.config_path)
        self.df_ref = pd.read_csv(os.path.join(self.results_dir, 'reference_centroids_c.csv'))
        self.df_dev = pd.read_csv(os.path.join(self.results_dir, 'spot_deviations_c.csv'))
        self.nspots = len(self.df_ref)

    def estimate_r0(self, dx, dy):
        """Estimate Fried parameter from slope variance"""
        p = self.cfg['camera_pixsize']
        f = self.cfg['flength']
        d = self.cfg['pitch']
        lam = self.cfg['wavelength']

        var_x = np.var(dx, ddof=1)
        var_y = np.var(dy, ddof=1)
        mean_var = 0.5 * (var_x * (p/f)**2 + var_y * (p/f)**2)

        if mean_var < 1e-15:
            return 0.0
        return (0.170 * lam**2 * d**(-1/3) / mean_var) ** (3/5)

    def plot_turbulence_telemetry(self, save=True):
        """7.3a: Turbulence parameters summary panel (from real time-series data)"""
        ts_path = os.path.join(self.results_dir, 'time_series.csv')
        if os.path.exists(ts_path):
            ts = pd.read_csv(ts_path)
            r0 = ts['r0'].mean()
            tau0 = ts['tau0'].mean()
        else:
            dx = self.df_dev['Delta_X'].values
            dy = self.df_dev['Delta_Y'].values
            r0 = self.estimate_r0(dx, dy)
            tau0 = 0.0035
            print("  WARNING: time_series.csv not found, using single-frame estimate")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('#1a1a2e')

        for ax, title, val, unit, color in [
            (ax1, 'Fried Parameter (r0)', r0*100, 'cm', '#00ff88'),
            (ax2, 'Coherence Time (tau0)', tau0*1000, 'ms', '#4488ff'),
        ]:
            ax.set_facecolor('#0d0d1a')
            ax.text(0.5, 0.6, f'{val:.2f}', ha='center', va='center',
                    fontsize=64, fontweight='bold', color=color, transform=ax.transAxes)
            ax.text(0.5, 0.25, unit, ha='center', va='center',
                    fontsize=20, color='gray', transform=ax.transAxes)
            ax.text(0.5, 0.88, title, ha='center', va='center',
                    fontsize=14, color='white', fontweight='bold', transform=ax.transAxes)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis('off')
            circle = plt.Circle((0.5, 0.55), 0.35, fill=False,
                                color=color, linewidth=2, linestyle='--', alpha=0.3)
            ax.add_patch(circle)

        if save:
            path = os.path.join(self.out_dir, 'turbulence_telemetry.png')
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Saved: {path}")
        return fig

    def plot_turbulence_regime(self, save=True):
        """7.3b: Turbulence regime classification (from real time-series data)"""
        ts_path = os.path.join(self.results_dir, 'time_series.csv')
        if os.path.exists(ts_path):
            ts = pd.read_csv(ts_path)
            n_frames = len(ts)
            D = 2.0 * self.cfg['pupil_radius']
            d_r0 = D / ts['r0'].values
        else:
            n_frames = 500
            np.random.seed(42)
            d_r0 = 1.0 + 9.0 * (0.5 + 0.5 * np.sin(np.linspace(0, 3*np.pi, n_frames)))
            d_r0 += np.random.randn(n_frames) * 0.3
            print("  WARNING: time_series.csv not found, using synthetic data")

        thresholds = [3.0, 6.0]
        labels = ['Strong', 'Moderate', 'Weak']
        colors_map = ['#ff4444', '#ffaa00', '#00ff88']

        regimes = np.zeros(n_frames, dtype=int)
        for i, t in enumerate(thresholds):
            regimes[d_r0 > t] = i + 1

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        fig.patch.set_facecolor('#1a1a2e')

        for ax in [ax1, ax2]:
            ax.set_facecolor('#0d0d1a')

        ax1.plot(d_r0, color='#4488ff', linewidth=1.5, alpha=0.85)
        for th in thresholds:
            ax1.axhline(y=th, color='gray', linestyle='--', alpha=0.5)
        ax1.fill_between(range(n_frames), thresholds[0], thresholds[1],
                          color='#ffaa00', alpha=0.1)
        ax1.fill_between(range(n_frames), thresholds[1], d_r0.max()+1,
                          color='#00ff88', alpha=0.1)
        ax1.fill_between(range(n_frames), 0, thresholds[0],
                          color='#ff4444', alpha=0.1)
        ax1.text(n_frames-10, thresholds[0]+0.3, 'Moderate', color='#ffaa00',
                 ha='right', fontsize=10, fontweight='bold')
        ax1.text(n_frames-10, thresholds[1]+0.3, 'Weak', color='#00ff88',
                 ha='right', fontsize=10, fontweight='bold')
        ax1.text(n_frames-10, thresholds[0]-0.8, 'Strong', color='#ff4444',
                 ha='right', fontsize=10, fontweight='bold')
        ax1.set_ylabel('D/r0', color='white', fontsize=11)
        ax1.set_title('Turbulence Strength Over Time', color='white',
                      fontsize=14, fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.15, color='gray')

        regime_colors = [colors_map[r] for r in regimes]
        ax2.scatter(range(n_frames), regimes, c=regime_colors, s=5, alpha=0.7)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(labels, color='white', fontsize=10)
        ax2.set_xlabel('Frame', color='white', fontsize=11)
        ax2.set_ylabel('Regime', color='white', fontsize=11)
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.15, color='gray')

        if save:
            path = os.path.join(self.out_dir, 'turbulence_regime.png')
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Saved: {path}")
        return fig

    def render_all(self):
        print("Rendering turbulence dashboard...")
        self.plot_turbulence_telemetry()
        self.plot_turbulence_regime()
        print("Done.")
