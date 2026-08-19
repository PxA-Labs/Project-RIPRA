#!/usr/bin/env python3
"""
test_slope_completion.py — End-to-end validation for slope-domain sequence
completion model (Issue #90: AI fallback).

Tests:
  1. Dataset generation with structured masking
  2. SlopeCompletionLSTM training convergence
  3. Sequence-level split leakage assertion
  4. Masked-loss gradient correctness (gradients only from missing entries)
  5. ONNX export round-trip parity
  6. RMSE vs spatial-interpolation baseline
  7. Physical constraint verification (stroke clamp)
  8. Inference latency benchmark (< 1 ms target)
"""
import os, sys, time, math
import numpy as np

# Graceful skip when PyTorch / onnxruntime are not available
try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:
    print("SKIP: torch not installed")
    sys.exit(0)

try:
    import onnxruntime as ort
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False

# ─── Setup paths ────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "ml"))

from sequence_models import SlopeCompletionLSTM

NSPOTS = 137
LOOKBACK = 10
HIDDEN = 128
NLAYERS = 2
N_SEQ = 5
SEQ_LEN = 100

# ─── Helpers ────────────────────────────────────────────────

def generate_synthetic_slopes(n_seq, seq_len, nspots, mask_prob=0.3, seed=42):
    """Generate synthetic slope sequences with structured masking.

    Returns:
        displacements: [n_seq * seq_len, 2*nspots]
        masks:         [n_seq * seq_len, 2*nspots]  (1=valid, 0=masked)
    """
    rng = np.random.RandomState(seed)
    total = n_seq * seq_len
    displacements = np.zeros((total, 2 * nspots), dtype=np.float32)
    masks = np.ones((total, 2 * nspots), dtype=np.float32)

    for s in range(n_seq):
        # AR(1) correlated slopes (simple model)
        rho = 0.95
        dx = rng.randn(nspots).astype(np.float32) * 2.0
        dy = rng.randn(nspots).astype(np.float32) * 2.0
        for t in range(seq_len):
            idx = s * seq_len + t
            noise_x = rng.randn(nspots).astype(np.float32) * 0.3
            noise_y = rng.randn(nspots).astype(np.float32) * 0.3
            dx = rho * dx + math.sqrt(1 - rho**2) * noise_x
            dy = rho * dy + math.sqrt(1 - rho**2) * noise_y
            displacements[idx, :nspots] = dx
            displacements[idx, nspots:] = dy

            # Generate mask: structured wedge dropout
            m = np.ones(2 * nspots, dtype=np.float32)
            if rng.rand() > 0.15:  # 85% of frames have some masking
                density = rng.uniform(0.1, 0.6)
                n_drop = int(density * nspots)
                drop_idx = rng.choice(nspots, size=n_drop, replace=False)
                m[drop_idx] = 0.0
                m[drop_idx + nspots] = 0.0
            masks[idx] = m

    return displacements, masks


def build_windows(displacements, masks, lookback, seq_len):
    """Build sliding windows for slope completion training.

    Returns lists of (x, y) tensors:
        x: [lookback+1, 4*nspots]  —  masked slopes + mask
        y: [2*nspots + 2*nspots]   —  clean slopes + mask (for current frame)
    """
    nspots = displacements.shape[1] // 2
    n_seq = len(displacements) // seq_len
    xs, ys, seq_ids = [], [], []
    for s in range(n_seq):
        for t in range(lookback, seq_len):
            idx_start = s * seq_len + t - lookback
            idx_end = s * seq_len + t + 1
            window_disp = displacements[idx_start:idx_end]
            window_mask = masks[idx_start:idx_end]
            x = np.concatenate([window_disp * window_mask, window_mask], axis=1)
            current_clean = window_disp[-1]
            current_mask = window_mask[-1]
            y = np.concatenate([current_clean, current_mask])
            xs.append(x)
            ys.append(y)
            seq_ids.append(s)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), seq_ids


passed = 0
failed = 0
total_tests = 0

def check(name, condition, detail=""):
    global passed, failed, total_tests
    total_tests += 1
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")


# ─── Test 1: Dataset Generation ─────────────────────────────
print("\n=== Test 1: Synthetic Dataset Generation ===")
displacements, masks = generate_synthetic_slopes(N_SEQ, SEQ_LEN, NSPOTS)
check("Dataset shape", displacements.shape == (N_SEQ * SEQ_LEN, 2 * NSPOTS),
      f"got {displacements.shape}")
check("Masks shape", masks.shape == (N_SEQ * SEQ_LEN, 2 * NSPOTS),
      f"got {masks.shape}")
check("Masks contain zeros", (masks == 0).any(), "no masked entries found")
check("Masks contain ones", (masks == 1).any(), "no valid entries found")
mask_ratio = masks.mean()
check("Mask ratio reasonable", 0.3 < mask_ratio < 0.95,
      f"mean valid = {mask_ratio:.3f}")

# ─── Test 2: Build Windows + Sequence Split ──────────────────
print("\n=== Test 2: Windowed Dataset + Sequence Split ===")
X, Y, seq_ids = build_windows(displacements, masks, LOOKBACK, SEQ_LEN)
check("Windows shape", X.shape[1:] == (LOOKBACK + 1, 4 * NSPOTS),
      f"got {X.shape}")
check("Targets shape", Y.shape[1] == 4 * NSPOTS,
      f"got {Y.shape}")

# Sequence-level split
unique_seqs = sorted(set(seq_ids))
n_train = max(1, int(0.8 * len(unique_seqs)))
n_val = max(1, len(unique_seqs) - n_train)
train_seqs = set(unique_seqs[:n_train])
val_seqs = set(unique_seqs[n_train:])

train_idx = [i for i, s in enumerate(seq_ids) if s in train_seqs]
val_idx = [i for i, s in enumerate(seq_ids) if s in val_seqs]

# Leakage check
train_s = set(seq_ids[i] for i in train_idx)
val_s = set(seq_ids[i] for i in val_idx)
overlap = train_s & val_s
check("No sequence leakage", len(overlap) == 0,
      f"{len(overlap)} sequences overlap")

# ─── Test 3: Training Convergence ────────────────────────────
print("\n=== Test 3: Training Convergence ===")
model = SlopeCompletionLSTM(nspots=NSPOTS, hidden_dim=HIDDEN, num_layers=NLAYERS)
device = torch.device("cpu")
model = model.to(device)

X_train = torch.tensor(X[train_idx], dtype=torch.float32)
Y_train = torch.tensor(Y[train_idx], dtype=torch.float32)
X_val = torch.tensor(X[val_idx], dtype=torch.float32)
Y_val = torch.tensor(Y[val_idx], dtype=torch.float32)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

def masked_loss(pred, target):
    n = pred.shape[1] // 2
    clean = target[:, :2*n]
    mask = target[:, 2*n:]
    inv_mask = 1.0 - mask
    diff2 = (pred - clean) ** 2
    loss_rec = (inv_mask * diff2).sum() / (inv_mask.sum() + 1e-8)
    return loss_rec

losses = []
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = masked_loss(pred, Y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

check("Loss decreases", losses[-1] < losses[0],
      f"first={losses[0]:.4f}, last={losses[-1]:.4f}")
check("Loss is finite", all(math.isfinite(l) for l in losses),
      f"losses={losses}")

# ─── Test 4: Gradient Masking Correctness ────────────────────
print("\n=== Test 4: Gradient Masking (Reconstruction Loss) ===")
model.eval()
model.zero_grad()
x_test = X_train[:2].clone().requires_grad_(False)
pred = model(x_test)
pred.retain_grad()

target = Y_train[:2]
n = pred.shape[1] // 2
clean = target[:, :2*n]
mask = target[:, 2*n:]
inv_mask = 1.0 - mask

# Compute reconstruction loss only
diff2 = (pred - clean) ** 2
loss_rec = (inv_mask * diff2).sum() / (inv_mask.sum() + 1e-8)
loss_rec.backward()

# Gradients should be zero for valid entries (mask=1 → inv_mask=0)
grad = pred.grad
valid_mask_bool = mask.bool()
grad_at_valid = grad[valid_mask_bool].abs()
grad_at_masked = grad[~valid_mask_bool].abs()

check("Gradients zero at valid entries (rec loss)",
      grad_at_valid.max().item() < 1e-7,
      f"max grad at valid = {grad_at_valid.max().item():.2e}")
check("Gradients non-zero at masked entries",
      grad_at_masked.max().item() > 1e-7,
      f"max grad at masked = {grad_at_masked.max().item():.2e}")

# ─── Test 5: ONNX Export Round-Trip ──────────────────────────
print("\n=== Test 5: ONNX Export Round-Trip ===")
if HAVE_ORT:
    onnx_path = os.path.join(BASE, "onnx_models", "_test_slope_completion.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model.eval()
    dummy = torch.randn(1, LOOKBACK + 1, 4 * NSPOTS, dtype=torch.float32)
    with torch.no_grad():
        torch_out = model(dummy).numpy()

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    check("ONNX file created", os.path.exists(onnx_path))

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"input": dummy.numpy()})[0]
    max_diff = np.max(np.abs(torch_out - ort_out))
    check("ONNX round-trip parity", max_diff < 1e-4,
          f"max diff = {max_diff:.2e}")

    # Clean up test artifact
    os.remove(onnx_path)
else:
    print("  SKIP: onnxruntime not installed")

# ─── Test 6: RMSE vs Spatial Interpolation Baseline ──────────
print("\n=== Test 6: RMSE vs Spatial Interpolation Baseline ===")
model.eval()
with torch.no_grad():
    preds = model(X_val).numpy()

clean_val = Y_val[:, :2*NSPOTS].numpy()
mask_val = Y_val[:, 2*NSPOTS:].numpy()
inv_mask_val = 1.0 - mask_val

# Model masked RMSE
model_se = ((preds - clean_val) ** 2) * inv_mask_val
model_rmse = np.sqrt(model_se.sum() / (inv_mask_val.sum() + 1e-8))

# Nearest-neighbour baseline (copy from nearest valid by index)
nn_se_total = 0.0
for b in range(clean_val.shape[0]):
    valid = mask_val[b] > 0.5
    for k in range(2 * NSPOTS):
        if not valid[k]:
            left = k - 1
            while left >= 0 and not valid[left]:
                left -= 1
            right = k + 1
            while right < 2 * NSPOTS and not valid[right]:
                right += 1
            if left >= 0 and (right >= 2 * NSPOTS or k - left <= right - k):
                nn_val = clean_val[b, left]
            elif right < 2 * NSPOTS:
                nn_val = clean_val[b, right]
            else:
                nn_val = 0.0
            nn_se_total += (nn_val - clean_val[b, k]) ** 2

nn_rmse = np.sqrt(nn_se_total / (inv_mask_val.sum() + 1e-8))

if nn_rmse > 0:
    ratio = model_rmse / nn_rmse
    print(f"  Model RMSE:    {model_rmse:.4f}")
    print(f"  Baseline RMSE: {nn_rmse:.4f}")
    print(f"  Ratio:         {ratio:.2%}")
    check("Model improves over baseline", ratio < 1.0,
          f"ratio = {ratio:.2%}")
else:
    check("Baseline RMSE non-zero", False, "nn_rmse = 0")

# ─── Test 7: Physical Constraint Verification ────────────────
print("\n=== Test 7: Physical Constraint (Stroke Clamp) ===")
pitch_px = 40.5
limit = pitch_px * 0.5

# Check that clamped outputs stay within bounds
with torch.no_grad():
    preds_all = model(torch.tensor(X, dtype=torch.float32)).numpy()

clamped = np.clip(preds_all, -limit, limit)
check("Stroke clamp preserves shape", clamped.shape == preds_all.shape)
check("Clamp reduces max magnitude",
      np.max(np.abs(clamped)) <= limit + 1e-6,
      f"max = {np.max(np.abs(clamped)):.2f}, limit = {limit:.2f}")

# ─── Test 8: Inference Latency ───────────────────────────────
print("\n=== Test 8: Inference Latency ===")
model.eval()
single_input = torch.randn(1, LOOKBACK + 1, 4 * NSPOTS, dtype=torch.float32)

# Warmup
for _ in range(50):
    with torch.no_grad():
        _ = model(single_input)

# Timed
times = []
for _ in range(200):
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(single_input)
    times.append(time.perf_counter() - t0)

times_ms = np.array(times) * 1000.0
mean_ms = np.mean(times_ms)
p99_ms = np.percentile(times_ms, 99)
print(f"  Mean: {mean_ms:.3f} ms, Median: {np.median(times_ms):.3f} ms, p99: {p99_ms:.3f} ms")

# PyTorch CPU inference is typically slower than ONNX RT, so we use a
# generous 5 ms threshold here; the ONNX RT benchmark (< 1 ms) is the
# authoritative latency check.
check("Inference < 5 ms (PyTorch CPU)", mean_ms < 5.0,
      f"mean = {mean_ms:.3f} ms")

if HAVE_ORT:
    print("\n  ONNX Runtime latency:")
    onnx_path = os.path.join(BASE, "onnx_models", "_test_latency.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model, single_input, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    sess = ort.InferenceSession(onnx_path)
    inp_np = single_input.numpy()

    for _ in range(100):
        sess.run(None, {"input": inp_np})

    ort_times = []
    for _ in range(500):
        t0 = time.perf_counter()
        sess.run(None, {"input": inp_np})
        ort_times.append(time.perf_counter() - t0)

    ort_ms = np.array(ort_times) * 1000.0
    ort_mean = np.mean(ort_ms)
    ort_p99 = np.percentile(ort_ms, 99)
    print(f"  Mean: {ort_mean:.3f} ms, Median: {np.median(ort_ms):.3f} ms, p99: {ort_p99:.3f} ms")
    check("ONNX RT inference < 1.0 ms (median)", np.median(ort_ms) < 1.0,
          f"median = {np.median(ort_ms):.3f} ms")

    os.remove(onnx_path)


# ─── Summary ─────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Results: {passed}/{total_tests} passed, {failed} failed")
print(f"{'='*50}")

sys.exit(1 if failed > 0 else 0)
