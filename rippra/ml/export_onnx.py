# ml/export_onnx.py - Phase 9.1: Export trained PyTorch models to ONNX format
import os, sys, argparse
import numpy as np
import torch
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from models import WavefrontMLP, WavefrontCNN
from evaluate_inference import load_system_config

def export_model(model, dummy_input, save_path, model_name, device='cpu'):
    model.to(device)
    model.eval()
    fname = os.path.basename(save_path)
    print(f"Exporting {model_name} -> {fname}...")
    print(f"  Input shape: {list(dummy_input.shape)}, dtype: {dummy_input.dtype}")
    with torch.no_grad():
        output = model(dummy_input.to(device))
    print(f"  Output shape: {list(output.shape)}")
    torch.onnx.export(
        model,
        dummy_input.to(device),
        save_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=17,
    )
    size_kb = os.path.getsize(save_path) / 1024
    print(f"  Saved: {fname} ({size_kb:.1f} KB)")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default=None, help='Output directory for ONNX files')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ONNX Export Tool on device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "system.conf")
    spots_csv = os.path.join(base_dir, "results", "reference_centroids_c.csv")
    checkpoint_dir = os.path.join(base_dir, "ml_checkpoints", "kaggle")
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = os.path.join(base_dir, "ml_checkpoints", "local")

    if args.output_dir is None:
        output_dir = os.path.join(base_dir, "onnx_models")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg = load_system_config(config_path)
    spots_df = pd.read_csv(spots_csv)
    nspots = len(spots_df)
    zernike_nmax = int(cfg["zernike_nmax"])
    nmodes = (zernike_nmax + 1) * (zernike_nmax + 2) // 2 - 1

    # 1. Export MLP
    mlp_model = WavefrontMLP(input_dim=nspots*2, output_dim=nmodes)
    mlp_ckpt = torch.load(os.path.join(checkpoint_dir, "best_mlp.pt"), map_location='cpu')
    mlp_model.load_state_dict(mlp_ckpt['model_state_dict'])
    mlp_dummy = torch.randn(1, nspots * 2, dtype=torch.float32)
    export_model(mlp_model, mlp_dummy, os.path.join(output_dir, "wavefront_mlp.onnx"), "WavefrontMLP")

    # 2. Export CNN (use forward_export for ONNX compatibility)
    cnn_model = WavefrontCNN(output_dim=nmodes)
    cnn_ckpt = torch.load(os.path.join(checkpoint_dir, "best_cnn.pt"), map_location='cpu')
    cnn_model.load_state_dict(cnn_ckpt['model_state_dict'])
    # Monkey-patch forward for ONNX export
    cnn_model.forward = cnn_model.forward_export

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
    cnn_dummy = torch.randn(1, 2, grid_h, grid_w, dtype=torch.float32)
    export_model(cnn_model, cnn_dummy, os.path.join(output_dir, "wavefront_cnn.onnx"), "WavefrontCNN")

    # 3. Export LSTM sequence predictor
    from sequence_models import WavefrontLSTM
    lstm_model = WavefrontLSTM(input_dim=nmodes, hidden_dim=128, output_dim=nmodes, num_layers=2)
    lstm_path = os.path.join(checkpoint_dir, "best_sequence_predict.pt")
    if os.path.exists(lstm_path):
        lstm_ckpt = torch.load(lstm_path, map_location='cpu')
        lstm_model.load_state_dict(lstm_ckpt['model_state_dict'])
        lstm_dummy = torch.randn(1, 10, nmodes, dtype=torch.float32)  # lookback=10
        export_model(lstm_model, lstm_dummy, os.path.join(output_dir, "wavefront_lstm.onnx"), "WavefrontLSTM")

    print(f"\nAll ONNX models saved to: {output_dir}/")
    print("Files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.onnx'):
            print(f"  {f} ({os.path.getsize(os.path.join(output_dir, f))/1024:.1f} KB)")

if __name__ == "__main__":
    main()
