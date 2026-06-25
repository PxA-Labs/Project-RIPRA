# ml/train.py - Train PyTorch models for Shack-Hartmann wavefront reconstruction
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from models import WavefrontMLP, WavefrontCNN

class WavefrontDataset(Dataset):
    """
    Dataset loader mapping 1D displacements to Zernike coefficients.
    If mode is 'cnn', handles spatial mapping of the 127 spots to a 2D grid.
    """
    def __init__(self, dataset_path, spots_csv_path, model_type='mlp'):
        self.model_type = model_type
        
        # Load npz dataset
        data = np.load(dataset_path)
        self.displacements = torch.tensor(data['displacements'], dtype=torch.float32)
        self.coefficients = torch.tensor(data['coefficients'], dtype=torch.float32)
        self.n_samples = len(self.displacements)
        self.nspots = self.displacements.shape[1] // 2
        
        # Load spot coordinates for spatial CNN mapping
        spots_df = pd.read_csv(spots_csv_path)
        
        # Estimate pitch and center
        mean_cx = spots_df["ref_cx"].mean()
        mean_cy = spots_df["ref_cy"].mean()
        
        # Approximate pitch in pixels (~40.1 px)
        # Find minimum distance to neighbor for pitch
        dists = []
        for i in range(len(spots_df)):
            dx = spots_df["ref_cx"].values - spots_df["ref_cx"].values[i]
            dy = spots_df["ref_cy"].values - spots_df["ref_cy"].values[i]
            d = np.hypot(dx, dy)
            d = d[d > 1e-3]
            if len(d) > 0:
                dists.append(d.min())
        pitch_px = np.mean(dists) if len(dists) > 0 else 40.1
        
        # Map spots to integer grid indices (u, v)
        u_coords = np.round((spots_df["ref_cx"].values - mean_cx) / pitch_px).astype(int)
        v_coords = np.round((spots_df["ref_cy"].values - mean_cy) / pitch_px).astype(int)
        
        u_min, u_max = u_coords.min(), u_coords.max()
        v_min, v_max = v_coords.min(), v_coords.max()
        
        self.grid_w = int(u_max - u_min + 1)
        self.grid_h = int(v_max - v_min + 1)
        self.u_offset = int(-u_min)
        self.v_offset = int(-v_min)
        
        self.spot_grid_coords = list(zip(u_coords, v_coords))
        
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        disp = self.displacements[idx]
        coeff = self.coefficients[idx]
        
        if self.model_type == 'mlp':
            return disp, coeff
        else: # cnn
            # Arrange 1D displacements into 2-channel 2D spatial grid (dx, dy)
            grid = torch.zeros((2, self.grid_h, self.grid_w), dtype=torch.float32)
            for k in range(self.nspots):
                u, v = self.spot_grid_coords[k]
                row = v + self.v_offset
                col = u + self.u_offset
                grid[0, row, col] = disp[k]
                grid[1, row, col] = disp[k + self.nspots]
            return grid, coeff

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def main():
    parser = argparse.ArgumentParser(description="Train Shack-Hartmann neural network reconstructor")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"], help="Model type to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dataset", type=str, default="data_ai/dataset.npz", help="Dataset path")
    parser.add_argument("--out_dir", type=str, default="ml_checkpoints", help="Output checkpoints directory")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    # 1. Load dataset
    spots_csv = "results/reference_centroids_c.csv"
    if not os.path.exists(spots_csv):
        print(f"Error: {spots_csv} not found. Please calibrate first using C pipeline.")
        return
        
    print(f"Loading dataset from {args.dataset}...")
    full_dataset = WavefrontDataset(args.dataset, spots_csv, model_type=args.model)
    
    # Train / Val / Test split (80% / 10% / 10%)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    train_set, val_set, test_set = random_split(
        full_dataset, [train_len, val_len, test_len], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    print(f"Dataset split: Train={train_len}, Val={val_len}, Test={test_len}")
    
    # 2. Setup model
    if args.model == "mlp":
        model = WavefrontMLP(input_dim=full_dataset.nspots * 2, output_dim=full_dataset.coefficients.shape[1])
    else: # cnn
        model = WavefrontCNN(output_dim=full_dataset.coefficients.shape[1])
        
    model = model.to(device)
    print(model)
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Training Loop
    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")
    
    print("\nStarting Training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.out_dir, f"best_{args.model}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            
    # 4. Final Evaluation
    checkpoint_path = os.path.join(args.out_dir, f"best_{args.model}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test MSE Loss on Best Checkpoint: {test_loss:.6f}")
    
    # Compare with C Reconstructor on 5 test samples
    model.eval()
    print("\nVisual spot check: True vs Predicted Zernike coeffs (first 5 modes of sample 1):")
    with torch.no_grad():
        inputs, targets = next(iter(test_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        preds = model(inputs)
        for i in range(5):
            print(f"  Sample {i+1}: True={[float(x) for x in targets[i][:5]]} | Pred={[float(x) for x in preds[i][:5]]}")

if __name__ == "__main__":
    main()
