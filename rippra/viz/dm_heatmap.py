# viz/dm_heatmap.py — DM actuator stroke heatmap visualization
#
# Generates a heatmap of DM actuator commands over the actuator grid.
# Requires pipeline results (reference centroids + spot deviations).
#
# Usage: python dm_heatmap.py
# Output: visualizations/dm_actuator_heatmap.png

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))
from evaluate_inference import load_system_config, compute_classical_zernike, noll_to_nm

class DMHeatmap:
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
        self.pupil_r = float(self.cfg.get('pupil_radius', 2e-3))
        self.pitch = float(self.cfg.get('pitch', 300e-6))

    def _compute_dm_commands(self):
        """Compute DM commands from Zernike coefficients via influence matrix.
        Returns (commands, node_u, node_v)."""
        coeffs, meta = compute_classical_zernike(
            self.df_dev['Delta_X'].values, self.df_dev['Delta_Y'].values,
            self.cfg, self.df_ref
        )

        nmodes = len(coeffs)
        pupil_cx = self.df_ref['ref_cx'].mean()
        pupil_cy = self.df_ref['ref_cy'].mean()
        pitch_px = float(self.cfg.get('pitch', 300e-6)) / float(self.cfg.get('camera_pixsize', 7.4e-6))

        # Build actuator grid as unique corners of sub-aperture grid
        u = np.round((self.df_ref['ref_cx'].values - pupil_cx) / pitch_px).astype(int)
        v = np.round((self.df_ref['ref_cy'].values - pupil_cy) / pitch_px).astype(int)

        corners_u = np.concatenate([u, u + 1, u, u + 1])
        corners_v = np.concatenate([v, v, v + 1, v + 1])
        nodes = np.unique(np.column_stack([corners_u, corners_v]), axis=0)
        node_u, node_v = nodes[:, 0], nodes[:, 1]
        nnodes = len(node_u)

        # Evaluate Zernike surface at each node (normalized pupil coords)
        r_norm = np.sqrt(node_u**2 + node_v**2) / (np.sqrt(self.nspots) / 2)
        theta = np.arctan2(node_v, node_u)

        phase = np.zeros(nnodes)
        for j, c in enumerate(coeffs):
            n, m = noll_to_nm(j + 1)
            # Zernike radial polynomial
            R = np.zeros(nnodes)
            for s in range((n - abs(m)) // 2 + 1):
                num = (-1)**s * np.math.factorial(n - s)
                den = (np.math.factorial(s) *
                       np.math.factorial((n + abs(m)) // 2 - s) *
                       np.math.factorial((n - abs(m)) // 2 - s))
                R += num / den * r_norm**(n - 2 * s)

            N = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
            R *= N

            inside = r_norm <= 1.0
            phase[inside] += c * R[inside] * (
                np.cos(abs(m) * theta[inside]) if m >= 0
                else np.sin(abs(m) * theta[inside])
            )

        # DM commands = conjugate phase (with coupling)
        coupling = float(self.cfg.get('coupling', 0.15))
        commands = -phase.copy()
        for i in range(nnodes):
            for j in range(nnodes):
                if i == j:
                    continue
                du = abs(int(node_u[i]) - int(node_u[j]))
                dv = abs(int(node_v[i]) - int(node_v[j]))
                if (du == 1 and dv == 0) or (du == 0 and dv == 1):
                    commands[i] -= coupling * phase[j]
                elif du == 1 and dv == 1:
                    commands[i] -= coupling * coupling * phase[j]

        return commands, node_u, node_v, phase

    def plot_dm_heatmap(self, save=True):
        """Generate DM actuator stroke heatmap."""
        commands, node_u, node_v, phase = self._compute_dm_commands()

        # Map to a regular grid for imshow
        u_min, u_max = node_u.min(), node_u.max()
        v_min, v_max = node_v.min(), node_v.max()
        nu = u_max - u_min + 1
        nv = v_max - v_min + 1
        grid = np.full((nv, nu), np.nan)
        for u, v, cmd in zip(node_u, node_v, commands):
            grid[int(v_max - v), int(u - u_min)] = cmd

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('#1a1a2e')

        for ax, title, data, cmap, label in [
            (ax1, 'DM Actuator Commands', grid, 'RdBu_r', 'Stroke (rad)'),
            (ax2, 'Zonal Phase Map', None, 'viridis', 'Phase (rad)')
        ]:
            ax.set_facecolor('#1a1a2e')
            ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Actuator X (grid units)', color='white')
            ax.set_ylabel('Actuator Y (grid units)', color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('#555555')

            if data is not None:
                im = ax.imshow(data, cmap=cmap, origin='upper',
                               interpolation='nearest', aspect='equal')
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label(label, color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

                # Pupil circle overlay
                cx = (u_max - u_min) / 2
                cy = (v_max - v_min) / 2
                radius = cx * 0.95
                pupil = Circle((cx, cy), radius, fill=False,
                               edgecolor='#e94560', linewidth=2, linestyle='--')
                ax.add_patch(pupil)
            else:
                # Phase scatter plot
                r_norm = np.sqrt(node_u**2 + node_v**2) / (np.sqrt(self.nspots) / 2)
                sc = ax.scatter(node_u, node_v, c=phase, cmap='viridis',
                                s=80, edgecolors='white', linewidth=0.3,
                                vmin=-np.max(np.abs(phase)), vmax=np.max(np.abs(phase)))
                cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
                cbar.set_label('Phase (rad)', color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

                pupil = Circle((0, 0), np.sqrt(self.nspots) / 2 * 0.95,
                               fill=False, edgecolor='#e94560', linewidth=2, linestyle='--')
                ax.add_patch(pupil)
                ax.set_xlim(node_u.min() - 1, node_u.max() + 1)
                ax.set_ylim(node_v.min() - 1, node_v.max() + 1)
                ax.set_aspect('equal')

        plt.tight_layout(pad=3)

        if save:
            path = os.path.join(self.out_dir, 'dm_actuator_heatmap.png')
            fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
            print(f"Saved: {path}")

        plt.close(fig)


if __name__ == '__main__':
    dm = DMHeatmap()
    dm.plot_dm_heatmap()
