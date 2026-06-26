"""Generate animated matplotlib 3D wavefront GIF from real time-series data"""
import os, sys, math, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'ml'))
from evaluate_inference import noll_to_nm

OUT = os.path.join(BASE, '..', 'visualizations')
os.makedirs(OUT, exist_ok=True)

# Load time-series data
ts_path = os.path.join(BASE, 'results', 'zernike_time_series.csv')
if not os.path.exists(ts_path):
    print(f"ERROR: {ts_path} not found. Run tools/generate_realistic_ts.py first.")
    sys.exit(1)

ts_df = pd.read_csv(ts_path)
nmodes = len([c for c in ts_df.columns if c.startswith('z')])
coeffs_all = ts_df[[f'z{i}' for i in range(nmodes)]].values
n_frames_total = len(ts_df)
n_frames = min(60, n_frames_total)

# Config (hard-code pupil radius for rendering)
pupil_r = 2e-3
n_grid = 100
x = np.linspace(-pupil_r, pupil_r, n_grid)
y = np.linspace(-pupil_r, pupil_r, n_grid)
X, Y = np.meshgrid(x, y)
rho = np.sqrt(X**2 + Y**2) / pupil_r
mask = rho <= 1.0
theta = np.arctan2(Y, X)

# Precompute Zernike basis shapes
basis = np.zeros((nmodes, n_grid, n_grid))
for idx in range(nmodes):
    j = idx + 2
    n, m = noll_to_nm(j)
    abs_m = abs(m)
    R = np.zeros_like(X)
    for s in range((n - abs_m) // 2 + 1):
        c = ((-1)**s * math.factorial(n - s) /
             (math.factorial(s) * math.factorial((n + abs_m)//2 - s) * math.factorial((n - abs_m)//2 - s)))
        R += c * (rho ** (n - 2*s))
    Z_ij = R * (np.cos(abs_m * theta) if m >= 0 else np.sin(abs_m * theta))
    norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    basis[idx] = norm * Z_ij

# Compute global vmax from all frames
vmax_global = 0
for f in range(n_frames):
    coeffs = coeffs_all[f * n_frames_total // n_frames]
    Z = np.zeros_like(X)
    for idx in range(nmodes):
        Z += coeffs[idx] * basis[idx]
    Z[~mask] = np.nan
    v = np.nanmax(abs(Z))
    if v > vmax_global:
        vmax_global = v

print(f"Generating {n_frames} frames from real time-series ({n_frames_total} available)...")
frames = []

# Select evenly spaced frames
frame_indices = np.linspace(0, n_frames_total - 1, n_frames, dtype=int)

for fi, fi_idx in enumerate(frame_indices):
    coeffs = coeffs_all[fi_idx]
    Z = np.zeros_like(X)
    for idx in range(nmodes):
        Z += coeffs[idx] * basis[idx]
    Z[~mask] = np.nan

    fig = plt.figure(figsize=(14, 8), dpi=100)
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')

    X_m = np.ma.masked_where(~mask, X)
    Y_m = np.ma.masked_where(~mask, Y)
    Z_m = np.ma.masked_where(~mask, Z)

    surf = ax.plot_surface(X_m * 1e6, Y_m * 1e6, Z_m,
                           cmap='RdBu_r', linewidth=0, antialiased=True,
                           alpha=0.95, vmin=-vmax_global, vmax=vmax_global)

    ax.set_xlim(-pupil_r * 1e6, pupil_r * 1e6)
    ax.set_ylim(-pupil_r * 1e6, pupil_r * 1e6)
    ax.set_zlim(-vmax_global, vmax_global)

    ax.set_xlabel('X (um)', color='white', fontsize=11)
    ax.set_ylabel('Y (um)', color='white', fontsize=11)
    ax.set_zlabel('Phase (rad)', color='white', fontsize=11)
    ax.set_title('3D Reconstructed Wavefront', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.xaxis.pane.set_facecolor('#0d0d1a')
    ax.yaxis.pane.set_facecolor('#0d0d1a')
    ax.zaxis.pane.set_facecolor('#0d0d1a')
    ax.grid(True, alpha=0.15, color='gray')

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Phase (rad)', color='white')
    for t2 in cbar.ax.yaxis.get_ticklabels():
        t2.set_color('white')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    frames.append(buf.read())

# Create GIF
from PIL import Image
images = []
for frame_data in frames:
    img = Image.open(io.BytesIO(frame_data))
    images.append(img.convert('P', palette=Image.Palette.ADAPTIVE, colors=256))

gif_path = os.path.join(OUT, 'wavefront_3d_anim.gif')
images[0].save(gif_path, save_all=True, append_images=images[1:],
               duration=100, loop=0, optimize=True)
print(f"Saved GIF: {gif_path} ({os.path.getsize(gif_path)/1024:.0f} KB)")
