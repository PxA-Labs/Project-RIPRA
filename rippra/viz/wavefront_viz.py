# viz/wavefront_viz.py - Wavefront Visualization (Checkpoint 7.1)
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))
from evaluate_inference import load_system_config, compute_classical_zernike, noll_to_nm

class WavefrontVisualizer:
    def __init__(self, results_dir=None, config_path=None):
        self.base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = results_dir or os.path.join(self.base, 'results')
        self.config_path = config_path or os.path.join(self.base, 'config', 'system.conf')
        self.raw_dir = os.path.join(self.base, 'data_raw')
        self.out_dir = os.path.join(self.base, '..', 'visualizations')
        os.makedirs(self.out_dir, exist_ok=True)

        # Load config and data
        self.cfg = load_system_config(self.config_path)
        self.df_ref = pd.read_csv(os.path.join(self.results_dir, 'reference_centroids_c.csv'))
        self.df_dev = pd.read_csv(os.path.join(self.results_dir, 'spot_deviations_c.csv'))
        self.nspots = len(self.df_ref)

    def plot_spot_centroid_offsets(self, save=True):
        """7.1c: Spot centroid offsets overlay on frame"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#1a1a2e')

        ref_cx, ref_cy = self.df_ref['ref_cx'], self.df_ref['ref_cy']
        dx, dy = self.df_dev['Delta_X'], self.df_dev['Delta_Y']
        cur_cx, cur_cy = ref_cx + dx, ref_cy + dy

        ax.scatter(ref_cx, ref_cy, c='#00ff88', s=40, marker='o', alpha=0.8,
                   label='Reference centroids', edgecolors='white', linewidth=0.5)
        ax.scatter(cur_cx, cur_cy, c='#ff4444', s=40, marker='x', alpha=0.8,
                   label='Aberrated centroids', linewidth=2)

        for i in range(len(ref_cx)):
            ax.arrow(ref_cx[i], ref_cy[i],
                     dx[i], dy[i],
                     head_width=1.5, head_length=1.5,
                     fc='#ffaa00', ec='#ffaa00', alpha=0.5, width=0.2)

        ax.set_title('SH-WFS Spot Centroid Offsets', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)', color='white', fontsize=11)
        ax.set_ylabel('Y (pixels)', color='white', fontsize=11)
        ax.legend(fontsize=10, loc='upper right')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15, color='white')

        if save:
            path = os.path.join(self.out_dir, 'spot_centroid_offsets.png')
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Saved: {path}")
        return fig

    def plot_zonal_wavefront_2d(self, save=True, n_grid=200):
        """7.1a: 2D zonal wavefront phase map via interpolation"""
        zernike_nmax = int(self.cfg['zernike_nmax'])
        max_j = (zernike_nmax + 1) * (zernike_nmax + 2) // 2
        nmodes = max_j - 1

        coeffs, _ = compute_classical_zernike(
            self.df_dev['Delta_X'].values,
            self.df_dev['Delta_Y'].values,
            self.cfg, self.df_ref
        )

        pupil_r = self.cfg['pupil_radius']
        x = np.linspace(-pupil_r, pupil_r, n_grid)
        y = np.linspace(-pupil_r, pupil_r, n_grid)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        rho = np.sqrt(X**2 + Y**2) / pupil_r
        mask = rho <= 1.0

        for idx in range(nmodes):
            j = idx + 2
            n, m = noll_to_nm(j)
            abs_m = abs(m)
            theta = np.arctan2(Y, X)
            R = np.zeros_like(X)
            for s in range((n - abs_m) // 2 + 1):
                c = ((-1)**s * math.factorial(n - s) /
                     (math.factorial(s) * math.factorial((n + abs_m)//2 - s) * math.factorial((n - abs_m)//2 - s)))
                R += c * (rho ** (n - 2*s))
            Z_ij = R * (np.cos(abs_m * theta) if m >= 0 else np.sin(abs_m * theta))
            norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
            Z += coeffs[idx] * norm * Z_ij

        Z[~mask] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(10, 9), subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('#1a1a2e')

        r_plot = np.linspace(0, 1, n_grid)
        theta_plot = np.linspace(0, 2*np.pi, n_grid)
        R_grid, T_grid = np.meshgrid(r_plot, theta_plot)

        from scipy.interpolate import griddata
        points = np.column_stack([np.sqrt(X**2+Y**2)[mask] / pupil_r, np.arctan2(Y, X)[mask]])
        Z_interp = griddata(points, Z[mask], (R_grid, T_grid), method='cubic')

        vmax = max(abs(np.nanmin(Z)), abs(np.nanmax(Z)))
        c = ax.pcolormesh(T_grid, R_grid, Z_interp, cmap='RdBu_r',
                          shading='auto', vmin=-vmax, vmax=vmax)
        ax.set_title('Reconstructed Wavefront Phase Map', color='white', fontsize=14,
                     fontweight='bold', pad=20)
        ax.grid(True, alpha=0.2, color='gray')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        cbar = plt.colorbar(c, ax=ax, pad=0.08, shrink=0.8)
        cbar.set_label('Phase (radians)', color='white')
        for t in cbar.ax.yaxis.get_ticklabels():
            t.set_color('white')

        ax.figure.axes[1].tick_params(colors='white')

        if save:
            path = os.path.join(self.out_dir, 'wavefront_phase_2d.png')
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Saved: {path}")
        return fig

    def plot_wavefront_3d(self, save=True, n_grid=100):
        """7.1b: 3D wavefront surface"""
        zernike_nmax = int(self.cfg['zernike_nmax'])
        max_j = (zernike_nmax + 1) * (zernike_nmax + 2) // 2
        nmodes = max_j - 1

        coeffs, _ = compute_classical_zernike(
            self.df_dev['Delta_X'].values,
            self.df_dev['Delta_Y'].values,
            self.cfg, self.df_ref
        )

        pupil_r = self.cfg['pupil_radius']
        x = np.linspace(-pupil_r, pupil_r, n_grid)
        y = np.linspace(-pupil_r, pupil_r, n_grid)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        rho = np.sqrt(X**2 + Y**2) / pupil_r
        mask = rho <= 1.0

        for idx in range(nmodes):
            j = idx + 2
            n, m = noll_to_nm(j)
            abs_m = abs(m)
            theta = np.arctan2(Y, X)
            R = np.zeros_like(X)
            for s in range((n - abs_m) // 2 + 1):
                c = ((-1)**s * math.factorial(n - s) /
                     (math.factorial(s) * math.factorial((n + abs_m)//2 - s) * math.factorial((n - abs_m)//2 - s)))
                R += c * (rho ** (n - 2*s))
            Z_ij = R * (np.cos(abs_m * theta) if m >= 0 else np.sin(abs_m * theta))
            norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
            Z += coeffs[idx] * norm * Z_ij
        Z[~mask] = np.nan

        fig = plt.figure(figsize=(14, 8))
        fig.patch.set_facecolor('#1a1a2e')
        ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')

        X_masked = np.ma.masked_where(~mask, X)
        Y_masked = np.ma.masked_where(~mask, Y)
        Z_masked = np.ma.masked_where(~mask, Z)

        surf = ax.plot_surface(X_masked * 1e6, Y_masked * 1e6, Z_masked,
                               cmap='RdBu_r', linewidth=0, antialiased=True,
                               alpha=0.95, vmin=-np.nanmax(abs(Z)), vmax=np.nanmax(abs(Z)))

        ax.set_xlabel('X (μm)', color='white', fontsize=11)
        ax.set_ylabel('Y (μm)', color='white', fontsize=11)
        ax.set_zlabel('Phase (rad)', color='white', fontsize=11)
        ax.set_title('3D Reconstructed Wavefront', color='white', fontsize=14, fontweight='bold')
        ax.tick_params(colors='white')
        ax.xaxis.pane.set_facecolor('#0d0d1a')
        ax.yaxis.pane.set_facecolor('#0d0d1a')
        ax.zaxis.pane.set_facecolor('#0d0d1a')
        ax.grid(True, alpha=0.15, color='gray')

        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Phase (rad)', color='white')
        for t in cbar.ax.yaxis.get_ticklabels():
            t.set_color('white')

        if save:
            path = os.path.join(self.out_dir, 'wavefront_3d.png')
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Saved: {path}")
        return fig

    def render_all(self):
        print("Rendering wavefront visualizations...")
        self.plot_spot_centroid_offsets()
        self.plot_zonal_wavefront_2d()
        self.plot_wavefront_3d()
        print("Done.")
