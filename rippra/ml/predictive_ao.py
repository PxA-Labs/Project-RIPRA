"""
predictive_ao.py — Phase 11.2: Predictive AO with LSTM feedforward training pipeline.
Shows the end-to-end training + evaluation of a predictive AO LSTM.
"""
import os, sys, math, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend([os.path.join(BASE, p) for p in ('ml','tools','bindings')])
from generate_dataset import get_noll_covariance_matrix

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from evaluate_inference import load_system_config
    HAVE_TORCH = True
except:
    HAVE_TORCH = False
    def load_system_config(path):
        cfg = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.split('#', 1)[0].strip()
                    try: cfg[k] = float(v) if '.' in v or 'e' in v else int(v)
                    except: cfg[k] = v
        return cfg

OUT = os.path.join(BASE, '..', 'visualizations')
os.makedirs(OUT, exist_ok=True)

cfg = load_system_config(os.path.join(BASE, "config", "system.conf"))
nmodes, lookback = 20, 10

def gen_wf_batch(n_seq=50, seq_len=200, dr0=3.0, tau0=0.005, seed=None):
    if seed is not None:
        np.random.seed(seed)
    C, _ = get_noll_covariance_matrix((int(cfg['zernike_nmax'])+1)*(int(cfg['zernike_nmax'])+2)//2)
    L = np.linalg.cholesky(C * dr0**(5/3) + np.eye(nmodes)*1e-8)
    rho = math.exp(-1e-3/tau0)
    data = np.zeros((n_seq, seq_len, nmodes), dtype=np.float32)
    for s in range(n_seq):
        a = L @ np.random.randn(nmodes)
        for t in range(seq_len):
            if t: a = rho*a + math.sqrt(1-rho**2)*L@np.random.randn(nmodes)
            data[s, t] = a
    return data

def make_sequences(data, step=1):
    """Convert raw sequences to (input_history, target_future) pairs"""
    n_seq, seq_len, _ = data.shape
    X, Y = [], []
    for s in range(n_seq):
        for t in range(lookback, seq_len - step):
            X.append(data[s, t-lookback:t])
            Y.append(data[s, t + step])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

class SmallLSTM(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=20, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def main():
    print("=== Predictive AO LSTM Training ===")
    if not HAVE_TORCH:
        print("  PyTorch not available — skipping training")
        return

    # Generate training/validation data
    print("  Generating training sequences...")
    train_data = gen_wf_batch(100, 200, dr0=3.0, tau0=0.005, seed=42)
    val_data = gen_wf_batch(20, 200, dr0=3.0, tau0=0.005, seed=99)
    X_tr, Y_tr = make_sequences(train_data, step=1)
    X_val, Y_val = make_sequences(val_data, step=1)
    print(f"  Train: {X_tr.shape}, Val: {X_val.shape}")

    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallLSTM(input_dim=nmodes, hidden_dim=64, output_dim=nmodes, num_layers=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(Y_tr)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(Y_val)), batch_size=128)

    history = {'train_loss': [], 'val_loss': []}
    for epoch in range(20):
        model.train()
        tl = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
        tl /= len(train_loader.dataset)

        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += loss_fn(model(xb), yb).item() * len(xb)
        vl /= len(val_loader.dataset)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        print(f"  Epoch {epoch+1:2d}: train={tl:.6f} val={vl:.6f}")

    # Evaluate: persistence vs LSTM prediction error
    model.eval()
    X_val_np, Y_val_np = X_val, Y_val
    with torch.no_grad():
        preds = model(torch.tensor(X_val_np).to(device)).cpu().numpy()

    pers_err = np.mean((Y_val_np - X_val_np[:, -1, :])**2)
    lstm_err = np.mean((Y_val_np - preds)**2)
    ratio = pers_err / lstm_err if lstm_err > 0 else float('inf')
    print(f"\n  Persistence MSE: {pers_err:.6f}")
    print(f"  LSTM MSE:        {lstm_err:.6f}")
    print(f"  Improvement:     {ratio:.2f}x")

    # Open-loop CL simulation with the trained model
    print("\n  Closed-loop evaluation on held-out sequence...")
    wf_test = gen_wf_batch(1, 500, dr0=3.0, tau0=0.005, seed=123)[0]

    def run_cl(latency, wf, model=None, gain=0.5):
        n = len(wf)
        rms = np.zeros(n)
        dm = np.zeros(nmodes)
        q = np.zeros((latency, nmodes))
        h_wf = np.zeros((lookback, nmodes))  # history of true wf (extracted)
        for t in range(n):
            cur = wf[t] + dm
            rms[t] = np.sqrt(np.mean(cur**2))
            # Extract true wf from residual (known dm)
            wf_true = cur - dm
            if t > 0:
                h_wf[:-1] = h_wf[1:]; h_wf[-1] = wf_true
            if model and t >= lookback and latency > 0:
                with torch.no_grad():
                    inp = torch.tensor(h_wf[np.newaxis,:,:], dtype=torch.float32, device=device)
                    wf_next = model(inp).cpu().numpy()[0]
                # Predict residual at t+1: wf_next + dm (dm stays same until correction)
                delta = -gain * (wf_next + dm)
            else:
                delta = -gain * cur
            if latency == 0:
                dm += delta
            else:
                old = q[0].copy()
                q[:-1] = q[1:]; q[-1] = delta
                dm += old
        return rms

    for lat, g in [(0, 0.5), (1, 0.5)]:
        s = run_cl(lat, wf_test, model=None, gain=g)
        p = run_cl(lat, wf_test, model=model, gain=g)
        impr = 100 * (np.mean(s) - np.mean(p)) / np.mean(s)
        print(f"  Latency={lat}, gain={g}: CL={np.mean(s):.4f}, LSTM={np.mean(p):.4f} ({impr:+.1f}%)")

    # Plot training loss + CL comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    for row in axes:
        for ax in row:
            ax.set_facecolor('#0d0d1a')
            ax.tick_params(colors='white')
            ax.title.set_color('white')

    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], color='#ff8844', lw=1.5, label='Train')
    ax.plot(history['val_loss'], color='#44bbff', lw=1.5, label='Val')
    ax.set_title('LSTM Training Loss', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.1, color='gray')

    # Persistence vs LSTM comparison bar
    ax = axes[0, 1]
    ax.bar(['Persistence', f'LSTM ({ratio:.1f}x better)'], [pers_err, lstm_err],
           color=['#888', '#4488ff'], alpha=0.85, width=0.5)
    ax.set_title('Prediction MSE', fontsize=12)
    ax.grid(axis='y', alpha=0.1, color='gray')

    # CL latency=0
    ax = axes[1, 0]
    s0 = run_cl(0, wf_test, model=None, gain=0.5)
    p0 = run_cl(0, wf_test, model=model, gain=0.5)
    ax.plot(s0, color='#00ff88', lw=1.0, alpha=0.7, label='Standard CL')
    ax.plot(p0, color='#4488ff', lw=1.0, alpha=0.7, label='LSTM CL')
    ax.set_title('No Latency', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.1, color='gray')

    # CL latency=1
    ax = axes[1, 1]
    s1 = run_cl(1, wf_test, model=None, gain=0.5)
    p1 = run_cl(1, wf_test, model=model, gain=0.5)
    ax.plot(s1, color='#00ff88', lw=1.0, alpha=0.7, label='Standard CL')
    ax.plot(p1, color='#4488ff', lw=1.0, alpha=0.7, label='LSTM CL')
    ax.set_title('Latency = 1 frame', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.1, color='gray')

    fig.suptitle('Predictive AO: LSTM Training and Closed-Loop Performance', color='white', fontsize=14, y=1.01)
    path = os.path.join(OUT, 'predictive_ao.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Saved: {path}")

if __name__ == '__main__':
    main()
