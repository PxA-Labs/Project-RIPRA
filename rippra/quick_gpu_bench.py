import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml'))
from models import WavefrontMLP, WavefrontCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

nspots, nmodes = 127, 20
batch_size = 64
n_warmup = 50
n_iter = 500

# Load models
mlp = WavefrontMLP(input_dim=nspots*2, output_dim=nmodes)
cnn = WavefrontCNN(output_dim=nmodes)

base = os.path.dirname(os.path.abspath(__file__))
ckpt_dir = os.path.join(base, "ml_checkpoints", "kaggle")
if not os.path.exists(ckpt_dir):
    ckpt_dir = os.path.join(base, "ml_checkpoints", "local")
mlp.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_mlp.pt"), map_location='cpu')['model_state_dict'])
cnn.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_cnn.pt"), map_location='cpu')['model_state_dict'])

mlp = mlp.to(device)
cnn = cnn.to(device)
mlp.eval()
cnn.eval()

# Create dummy inputs
dummy_mlp = torch.randn(batch_size, nspots*2).to(device)
dummy_cnn = torch.randn(batch_size, 2, 11, 13).to(device)

def bench(dev, label):
    mlp_dev = mlp.to(dev); cnn_dev = cnn.to(dev)
    inp_mlp = dummy_mlp.to(dev); inp_cnn = dummy_cnn.to(dev)
    for _ in range(n_warmup):
        mlp_dev(inp_mlp); cnn_dev(inp_cnn)
    if dev.type == 'cuda': torch.cuda.synchronize()
    results = {}
    for name, model, inp in [("MLP", mlp_dev, inp_mlp), ("CNN", cnn_dev, inp_cnn)]:
        if dev.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(inp)
        if dev.type == 'cuda': torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat = (t1 - t0) / n_iter * 1000
        fps = batch_size / (lat / 1000)
        results[name] = (lat, fps)
        print(f"  {label:6s} {name:10s}: {lat:.4f} ms/batch | {fps:.0f} fps")
    return results

cpu_res = bench(torch.device('cpu'), "CPU")
gpu_res = bench(torch.device('cuda'), "GPU")

print(f"\nSpeedup:")
for name in ["MLP", "CNN"]:
    ratio = cpu_res[name][0] / gpu_res[name][0]
    print(f"  {name:10s}: {ratio:.1f}x faster on GPU")
