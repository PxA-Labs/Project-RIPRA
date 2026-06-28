# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RIPRA Synthetic SHWFS Data Generator
# 
# Generates physically accurate synthetic Shack-Hartmann wavefront sensor
# data for the RIPRA AO pipeline. All parameters exactly match the C code's
# expectations (recon.c, centroid.c, io.c).
# 
# **Outputs:**
# - `data_raw/sh_flat.raw` – reference flat frame (double, row-major)
# - `data_raw/img.raw` – aberrated frame with known Zernike coefficients
# - `ml_checkpoints/local/best_mlp.pt` – trained MLP model
# - `ml_checkpoints/local/best_cnn.pt` – trained CNN model
# - `data_ai/dataset.npz` – ML training dataset
# 
# Run this once on Kaggle, download the outputs, and commit to the repo.

# %% [markdown]
# ## 1. Setup — Install dependencies & configure paths

# %%
import os, sys, math, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Install additional deps if on Kaggle
try:
    import torch
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'torch'])
    import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Determine working directory
if os.path.exists('/kaggle'):
    BASE = '/kaggle/working'
else:
    BASE = os.path.abspath(os.path.join(os.getcwd(), '..'))

os.chdir(BASE)
print(f"Working directory: {BASE}")
os.makedirs('data_raw', exist_ok=True)
os.makedirs('ml_checkpoints/local', exist_ok=True)
os.makedirs('data_ai', exist_ok=True)

# %% [markdown]
# ## 2. System Configuration (matches config/system.conf)

# %%
CFG = {
    'camera_pixsize': 7.4e-6,   # m
    'frame_width': 648,           # px
    'frame_height': 492,          # px
    'totlenses': 140,             # allow 137 detected spots
    'flength': 18e-3,            # m
    'pitch': 300e-6,             # m
    'sa_radius': 150e-6,        # m
    'pupil_radius': 2e-3,       # m
    'wavelength': 632.8e-9,     # m
    'thresh_binary': 0.08,
    'centroid_percent': 0.2,
    'coarse_grid_radius': 12,
    'zernike_nmax': 5,
    'dm_nact_x': 12,
    'dm_nact_y': 12,
    'coupling': 0.15,
}
W, H = CFG['frame_width'], CFG['frame_height']
PIXSIZE = CFG['camera_pixsize']
FLENGTH = CFG['flength']
PITCH = CFG['pitch']
PUPIL_R = CFG['pupil_radius']
SA_R = CFG['sa_radius']
WL = CFG['wavelength']
NMAX = CFG['zernike_nmax']
NMODES = (NMAX + 1) * (NMAX + 2) // 2 - 1  # 20 modes

PITCH_PX = PITCH / PIXSIZE            # ~40.5 px
PUPIL_R_PX = PUPIL_R / PIXSIZE        # ~270 px

print(f"Pitch: {PITCH_PX:.2f} px")
print(f"Pupil radius: {PUPIL_R_PX:.1f} px")
print(f"Zernike modes: {NMODES}")

# %% [markdown]
# ## 3. Zernike Polynomials (bit-exact with recon.c)
# 
# These implementations exactly match the C code in `recon.c`:
# - `noll_to_nm()` – Noll index j → (n, m)
# - `zernike_coeff()` – radial coefficient C_s
# - `zernike_derivatives()` – analytical dz/dx, dz/dy
# - `zernike_eval()` – evaluate Z_n^m(x, y)

# %%
import math

_FACT_CACHE = [1.0]
def _fact(k):
    while len(_FACT_CACHE) <= k:
        _FACT_CACHE.append(_FACT_CACHE[-1] * len(_FACT_CACHE))
    return _FACT_CACHE[k]

def noll_to_nm(j):
    if j == 1:
        return 0, 0
    current_j = 2
    for ni in range(1, 100):
        for mi in range(ni % 2, ni + 1, 2):
            if mi == 0:
                if current_j == j: return ni, 0
                current_j += 1
            else:
                if current_j % 2 == 1:
                    if current_j == j: return ni, -mi
                    if current_j + 1 == j: return ni, mi
                else:
                    if current_j == j: return ni, mi
                    if current_j + 1 == j: return ni, -mi
                current_j += 2

def zernike_coeff(n, m, s):
    abs_m = abs(m)
    num = _fact(n - s)
    den = _fact(s) * _fact((n + abs_m)//2 - s) * _fact((n - abs_m)//2 - s)
    return (1.0 if s % 2 == 0 else -1.0) * num / den

def zernike_derivatives(n, m, x, y):
    rho = math.hypot(x, y); theta = math.atan2(y, x)
    abs_m = abs(m)
    R = 0.0; dR = 0.0
    for s in range((n - abs_m)//2 + 1):
        c = zernike_coeff(n, m, s)
        p = n - 2*s
        if p > 0:
            R += c * (rho ** p)
            dR += c * p * (rho ** (p - 1))
        elif p == 0:
            R += c
    norm = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
    R *= norm; dR *= norm
    c = math.cos(abs_m * theta); s = math.sin(abs_m * theta)
    if m >= 0:
        dz_drho = dR * c; dz_dtheta = -abs_m * R * s
    else:
        dz_drho = dR * s; dz_dtheta = abs_m * R * c
    if rho < 1e-9:
        if n == 1:
            if m == 1: return norm, 0.0
            if m == -1: return 0.0, norm
        return 0.0, 0.0
    dzdx = dz_drho * math.cos(theta) - dz_dtheta * math.sin(theta) / rho
    dzdy = dz_drho * math.sin(theta) + dz_dtheta * math.cos(theta) / rho
    return dzdx, dzdy

def zernike_eval(n, m, x, y):
    rho = math.hypot(x, y); theta = math.atan2(y, x)
    abs_m = abs(m)
    R = 0.0
    for s in range((n - abs_m)//2 + 1):
        c = zernike_coeff(n, m, s)
        p = n - 2*s
        R += c * (rho ** p) if p > 0 else c
    norm = math.sqrt(n + 1) if m == 0 else math.sqrt(2 * (n + 1))
    R *= norm
    if m >= 0: return R * math.cos(abs_m * theta)
    else: return R * math.sin(abs_m * theta)

# %% [markdown]
# ## 4. SHWFS Geometry — Lenslet Positions
# 
# Generates a rectangular grid of lenslet positions within the circular pupil,
# matching the Fried geometry used in `recon.c:rippra_zonal_setup`.

# %%
def lenslet_positions():
    cx, cy = W / 2.0, H / 2.0
    spots = []
    extent = int(PUPIL_R_PX / PITCH_PX) + 2
    for vi in range(-extent, extent + 1):
        for ui in range(-extent, extent + 1):
            sx = cx + ui * PITCH_PX
            sy = cy + vi * PITCH_PX
            if 0 <= sx < W and 0 <= sy < H:
                if math.hypot(sx - cx, sy - cy) <= PUPIL_R_PX:
                    spots.append((sx, sy))
    spots.sort(key=lambda p: (p[1], p[0]))

    # Also compute normalized pupil coordinates (matching recon.c modal setup)
    xn = np.array([(s[0] - cx) * PIXSIZE / PUPIL_R for s in spots])
    yn = np.array([-(s[1] - cy) * PIXSIZE / PUPIL_R for s in spots])

    return np.array(spots), xn, yn

positions, xn, yn = lenslet_positions()
NSPOTS = len(positions)
print(f"Detected {NSPOTS} lenslets within pupil")
print(f"Expected: ~127 (config: {CFG['totlenses']})")

# %% [markdown]
# ## 5. PSF Model — Gaussian Spot Rendering

# %%
def render_frame(positions, amplitude=600.0, sigma=1.5, background=20.0):
    """Render SHWFS frame with Gaussian PSF spots."""
    frame = np.full((H, W), background, dtype=np.float64)
    r = int(4 * sigma) + 1
    for sx, sy in positions:
        x0, y0 = int(round(sx)), int(round(sy))
        xl = max(0, x0 - r); xr = min(W, x0 + r + 1)
        yl = max(0, y0 - r); yr = min(H, y0 + r + 1)
        for y in range(yl, yr):
            dy2 = (y - sy) ** 2
            for x in range(xl, xr):
                d2 = (x - sx) ** 2 + dy2
                frame[y, x] += amplitude * math.exp(-d2 / (2.0 * sigma * sigma))
    return frame

def shifts_from_wavefront(coeffs):
    """Compute spot shifts from Zernike coefficients (radians)."""
    dx = np.zeros(NSPOTS); dy = np.zeros(NSPOTS)
    scale = FLENGTH * WL / (2.0 * math.pi * PUPIL_R * PIXSIZE)
    for j in range(NMODES):
        n, m = noll_to_nm(j + 2)
        a = coeffs[j]
        for k in range(NSPOTS):
            dzdx, dzdy = zernike_derivatives(n, m, xn[k], yn[k])
            dx[k] += a * dzdx * scale
            dy[k] += a * dzdy * scale
    return dx, dy

def save_raw(path, frame):
    frame.astype(np.float64).tofile(path)

# %% [markdown]
# ## 6. Generate Flat Frame (Reference)

# %%
print("Generating flat frame...")
flat = render_frame(positions)
flat_path = 'data_raw/sh_flat.raw'
save_raw(flat_path, flat)
flat_size = os.path.getsize(flat_path)
print(f"  Saved: {flat_path} ({flat_size//1024} KB)")

# Check detection readiness
thresh = flat.min() + 0.08 * (flat.max() - flat.min())
fg_px = (flat >= thresh).sum()
min_needed = NSPOTS * 8
print(f"  Binary threshold: {thresh:.1f}")
print(f"  Foreground pixels: {fg_px} (need >= {min_needed})")
assert fg_px >= min_needed, "Not enough foreground pixels for spot detection!"

# %% [markdown]
# ## 7. Generate Aberrated Frame with Known Wavefront

# %%
rng = np.random.RandomState(42)
# Kolmogorov-like random coefficients with power-law decay
coeffs_truth = rng.randn(NMODES)
for j in range(NMODES):
    coeffs_truth[j] *= 3.0 * (j + 2) ** (-0.7)  # power-law decay
# Normalize to desired RMS
rms_target = 2.0  # rad RMS
coeffs_truth = coeffs_truth / coeffs_truth.std() * rms_target

dx, dy = shifts_from_wavefront(coeffs_truth)
shifted = positions.copy()
shifted[:, 0] += dx; shifted[:, 1] += dy

print("Generating aberrated frame...")
img = render_frame(shifted)
img_path = 'data_raw/img.raw'
save_raw(img_path, img)
img_size = os.path.getsize(img_path)
print(f"  Saved: {img_path} ({img_size//1024} KB)")

print(f"\n  Truth coefficients (first 5): {coeffs_truth[:5].round(4)}")
print(f"  Spot shifts: |dx|max={abs(dx).max():.3f} px, |dy|max={abs(dy).max():.3f} px")

# %% [markdown]
# ## 8. Visualisation

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, frame, title in zip(axes, [flat, img], ['Flat Frame', 'Aberrated Frame']):
    vmin, vmax = 0, min(flat.max(), 255)
    im = ax.imshow(frame, cmap='hot', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14, color='white')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    # Overlay lenslet positions
    ax.scatter(positions[:, 0], positions[:, 1], c='cyan', s=5, alpha=0.5, label='Ref')
    if title == 'Aberrated Frame':
        ax.scatter(shifted[:, 0], shifted[:, 1], c='lime', s=5, alpha=0.5, label='Shifted')
    ax.legend(fontsize=8)

fig.patch.set_facecolor('#1a1a2e')
fig.suptitle('Synthetic SHWFS Frames', fontsize=16, color='white', y=1.02)
plt.tight_layout()
plt.savefig('shwfs_frames.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("Saved: shwfs_frames.png")

# %% [markdown]
# ## 9. Generate ML Training Dataset

# %%
def noll_variance(j, D_r0=8.0):
    if j <= 3: return 0.4874 * (D_r0 ** (5/3)) * 0.582
    return 0.4874 * (D_r0 ** (5/3)) * 0.294 * (j - 1) ** (-math.sqrt(3)/2)

def generate_dataset(n_samples, seed=123):
    """Generate (displacements, coefficients) pairs."""
    rng = np.random.RandomState(seed)
    disp = np.zeros((n_samples, NSPOTS * 2), dtype=np.float32)
    coef = np.zeros((n_samples, NMODES), dtype=np.float32)
    for i in range(n_samples):
        c = np.array([rng.randn() * math.sqrt(noll_variance(j + 2)) for j in range(NMODES)])
        dx_i, dy_i = shifts_from_wavefront(c)
        disp[i] = np.concatenate([dx_i, dy_i])
        coef[i] = c
    return disp, coef

print("Generating ML training dataset...")
N_TRAIN = 50000
N_VAL = 10000
N_TEST = 5000

X_train, Y_train = generate_dataset(N_TRAIN, seed=42)
X_val, Y_val = generate_dataset(N_VAL, seed=99)
X_test, Y_test = generate_dataset(N_TEST, seed=123)

np.savez('data_ai/dataset.npz',
         displacements=np.concatenate([X_train, X_val, X_test]),
         coefficients=np.concatenate([Y_train, Y_val, Y_test]))

print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")
print(f"  Total: {os.path.getsize('data_ai/dataset.npz')//1024**2:.1f} MB")

# %% [markdown]
# ## 10. Define & Train ML Models (MLP + CNN)

# %%
class WavefrontMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    def forward(self, x):
        return self.net(x)

class WavefrontCNN(nn.Module):
    def __init__(self, output_dim, grid_h=13, grid_w=17):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    def forward(self, x):
        return self.cnn(x)

# Build spatial grid for CNN
def build_spatial_grid(disp_batch):
    B = disp_batch.shape[0]
    xn_vals = (positions[:, 0] - W/2) / PITCH_PX
    yn_vals = (positions[:, 1] - H/2) / PITCH_PX
    u = np.round(xn_vals).astype(int)
    v = np.round(yn_vals).astype(int)
    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()
    grid_w_cnn = int(u_max - u_min + 1)
    grid_h_cnn = int(v_max - v_min + 1)
    grid = np.zeros((B, 2, grid_h_cnn, grid_w_cnn), dtype=np.float32)
    for k in range(NSPOTS):
        row = int(v[k] - v_min)
        col = int(u[k] - u_min)
        grid[:, 0, row, col] = disp_batch[:, k]
        grid[:, 1, row, col] = disp_batch[:, NSPOTS + k]
    return torch.from_numpy(grid), grid_h_cnn, grid_w_cnn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# MLP
print("\n--- Training MLP ---")
mlp = WavefrontMLP(input_dim=NSPOTS*2, output_dim=NMODES).to(device)
opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)), batch_size=128)

best_val = float('inf')
for epoch in range(15):
    mlp.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt_mlp.zero_grad()
        loss_fn(mlp(xb), yb).backward()
        opt_mlp.step()
    mlp.eval()
    vl = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            vl += loss_fn(mlp(xb.to(device)), yb.to(device)).item() * len(xb)
    vl /= len(val_loader.dataset)
    print(f"  Epoch {epoch+1:2d}: val_loss = {vl:.6f}")
    if vl < best_val:
        best_val = vl
        torch.save({'model_state_dict': mlp.state_dict()}, 'ml_checkpoints/local/best_mlp.pt')
        print(f"    -> saved (best)")

# CNN
print("\n--- Training CNN ---")
grid, gh, gw = build_spatial_grid(X_train)
grid_val, _, _ = build_spatial_grid(X_val)
cnn = WavefrontCNN(output_dim=NMODES, grid_h=gh, grid_w=gw).to(device)
opt_cnn = torch.optim.AdamW(cnn.parameters(), lr=1e-3, weight_decay=1e-4)

cnn_train = TensorDataset(grid, torch.from_numpy(Y_train))
cnn_val = TensorDataset(grid_val, torch.from_numpy(Y_val))
cnn_loader = DataLoader(cnn_train, batch_size=64, shuffle=True)
cnn_val_loader = DataLoader(cnn_val, batch_size=64)

best_val = float('inf')
for epoch in range(15):
    cnn.train()
    for xb, yb in cnn_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt_cnn.zero_grad()
        loss_fn(cnn(xb), yb).backward()
        opt_cnn.step()
    cnn.eval()
    vl = 0
    with torch.no_grad():
        for xb, yb in cnn_val_loader:
            vl += loss_fn(cnn(xb.to(device)), yb.to(device)).item() * len(xb)
    vl /= len(cnn_val_loader.dataset)
    print(f"  Epoch {epoch+1:2d}: val_loss = {vl:.6f}")
    if vl < best_val:
        best_val = vl
        torch.save({'model_state_dict': cnn.state_dict()}, 'ml_checkpoints/local/best_cnn.pt')
        print(f"    -> saved (best)")

# %% [markdown]
# ## 11. Test Set Evaluation

# %%
print("=== Test Set Evaluation ===")
mlp.eval(); cnn.eval()
X_test_t = torch.from_numpy(X_test).to(device)
Y_test_t = torch.from_numpy(Y_test).to(device)

with torch.no_grad():
    pred_mlp = mlp(X_test_t)
    mlp_mse = loss_fn(pred_mlp, Y_test_t).item()
    pred_cnn = cnn(build_spatial_grid(X_test)[0].to(device))
    cnn_mse = loss_fn(pred_cnn, Y_test_t).item()

print(f"  MLP Test MSE: {mlp_mse:.6f}")
print(f"  CNN Test MSE: {cnn_mse:.6f}")

# Per-mode correlation
corrs = []
for j in range(NMODES):
    c = np.corrcoef(Y_test[:, j], pred_mlp.cpu().numpy()[:, j])[0, 1]
    corrs.append(c)
print(f"  MLP mean correlation: {np.mean(corrs):.4f}")

# %% [markdown]
# ## 12. Export Config for C Pipeline

# %%
# Write an updated config with totlenses matching our lenslet count
with open('config/system.conf', 'w') as f:
    f.write(f"""# RIPPA system configuration (auto-generated for synthetic data)
camera_pixsize = {CFG['camera_pixsize']}
frame_width    = {CFG['frame_width']}
frame_height   = {CFG['frame_height']}
totlenses      = {NSPOTS + 5}
flength        = {CFG['flength']}
pitch          = {CFG['pitch']}
sa_radius      = {CFG['sa_radius']}
pupil_radius   = {CFG['pupil_radius']}
wavelength     = {CFG['wavelength']}
thresh_binary  = {CFG['thresh_binary']}
centroid_percent = {CFG['centroid_percent']}
coarse_grid_radius = {CFG['coarse_grid_radius']}
zernike_nmax   = {CFG['zernike_nmax']}
dm_nact_x      = {CFG['dm_nact_x']}
dm_nact_y      = {CFG['dm_nact_y']}
coupling       = {CFG['coupling']}
""")

# %% [markdown]
# ## 13. Summary of Generated Artifacts

# %%
print("=" * 60)
print("  GENERATED ARTIFACTS")
print("=" * 60)
artifacts = [
    ('data_raw/sh_flat.raw', 'Reference flat frame (double, row-major)'),
    ('data_raw/img.raw', 'Aberrated frame with known wavefront'),
    ('data_ai/dataset.npz', 'ML training dataset (70k samples)'),
    ('ml_checkpoints/local/best_mlp.pt', 'Trained MLP model'),
    ('ml_checkpoints/local/best_cnn.pt', 'Trained CNN model'),
    ('config/system.conf', 'Updated config (totlenses matched to lenslet count)'),
]
for path, desc in artifacts:
    sz = os.path.getsize(path) if os.path.exists(path) else 0
    print(f"  {path:40s} {sz//1024:>6d} KB  {desc}")

print(f"\n  Lenslets: {NSPOTS} ({'OK' if NSPOTS <= CFG['totlenses'] else 'exceeds config!'})")
print(f"  Config totlenses: {CFG['totlenses']}")
print(f"  Pitch: {PITCH_PX:.1f} px")
print(f"  MLP Test MSE: {mlp_mse:.6f}")
print(f"  CNN Test MSE: {cnn_mse:.6f}")
print("=" * 60)
print("Download these files and commit to the repo.")
print("Then run `ci.yml` to verify the full C pipeline passes.")
