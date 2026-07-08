# ml/train_sequence.py - Sequence training pipeline for Phase 5
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sequence_models import WavefrontLSTM, TurbulenceClassifierLSTM, TurbulenceParameterEstimator

def check_split_leakage(dataset, train_indices, val_indices, test_indices):
    """
    Assert that no sequence ID appears in more than one split.
    Must be called after SHSequenceDataset is constructed.
    """
    def seq_ids(indices):
        return set(dataset.sequence_ids[i] for i in indices)
    train_s = seq_ids(train_indices)
    val_s = seq_ids(val_indices)
    test_s = seq_ids(test_indices)
    tv = train_s & val_s
    tt = train_s & test_s
    vt = val_s & test_s
    if tv or tt or vt:
        raise AssertionError(
            f"Temporal leakage detected: train↔val {len(tv)}, "
            f"train↔test {len(tt)}, val↔test {len(vt)} sequences overlap."
        )

class SHSequenceDataset(Dataset):
    """
    Sequence dataset loader that slices sequence frames into sliding windows
    without crossing sequence boundaries (each sequence is 1000 frames).
    
    Splits should be performed at the *sequence* level (see split_by_sequence())
    to avoid temporal leakage from adjacent overlapping windows.
    """
    def __init__(self, dataset_path, lookback=10, step=1, task='predict'):
        self.lookback = lookback
        self.step = step
        self.task = task
        
        # Load dataset
        data = np.load(dataset_path)
        displacements = data['displacements']
        coefficients = data['coefficients']
        D_r0 = data['D_r0']
        
        n_frames = len(displacements)
        self.seq_len = 1000
        n_sequences = n_frames // self.seq_len
        
        self.samples = []
        self.sequence_ids = []
        for s in range(n_sequences):
            seq_start = s * self.seq_len
            
            # Slice sequence data
            seq_disp = displacements[seq_start : seq_start + self.seq_len]
            seq_coeff = coefficients[seq_start : seq_start + self.seq_len]
            seq_dr0 = D_r0[seq_start : seq_start + self.seq_len]
            
            # Determine classification class based on average D_r0 of sequence
            avg_dr0 = float(np.mean(seq_dr0))
            if avg_dr0 < 3.0:
                class_idx = 0  # Weak
            elif avg_dr0 < 7.0:
                class_idx = 1  # Moderate
            else:
                class_idx = 2  # Strong
                
            # Sliding windows within sequence
            for t in range(self.lookback, self.seq_len - self.step + 1):
                # Input history: L frames
                hist_coeff = seq_coeff[t - self.lookback : t]
                hist_disp = seq_disp[t - self.lookback : t]
                
                if self.task == 'predict':
                    # Target: coefficients at t + step - 1
                    target = seq_coeff[t + self.step - 1]
                    self.samples.append((hist_coeff, target))
                elif self.task == 'classify':
                    # Target: class index based on sequence turbulence strength
                    self.samples.append((hist_coeff, class_idx))
                elif self.task == 'parameter':
                    # Target: average D_r0 of sequence
                    self.samples.append((hist_disp, avg_dr0))
                self.sequence_ids.append(s)
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.task == 'classify':
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
        else:
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    @staticmethod
    def split_by_sequence(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
        """
        Split dataset by contiguous blocks of sequences (not individual samples).
        This avoids temporal leakage from adjacent overlapping windows.
        
        Returns (train_set, val_set, test_set) as torch Subset instances.
        """
        n_seqs = max(dataset.sequence_ids) + 1 if dataset.sequence_ids else 0
        seq_indices = list(range(n_seqs))
        rng = np.random.RandomState(seed)
        rng.shuffle(seq_indices)
        
        n_train = int(train_ratio * n_seqs)
        n_val = int(val_ratio * n_seqs)
        
        train_seqs = set(seq_indices[:n_train])
        val_seqs = set(seq_indices[n_train:n_train + n_val])
        test_seqs = set(seq_indices[n_train + n_val:])
        
        train_idx = [i for i, sid in enumerate(dataset.sequence_ids) if sid in train_seqs]
        val_idx   = [i for i, sid in enumerate(dataset.sequence_ids) if sid in val_seqs]
        test_idx  = [i for i, sid in enumerate(dataset.sequence_ids) if sid in test_seqs]
        
        check_split_leakage(dataset, train_idx, val_idx, test_idx)
        
        return (Subset(dataset, train_idx),
                Subset(dataset, val_idx),
                Subset(dataset, test_idx))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Squeeze output if doing 1D parameter estimation regression
        if outputs.dim() > 1 and targets.dim() == 1:
            outputs = outputs.squeeze(1)
            
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if outputs.dim() > 1 and targets.dim() == 1:
                outputs = outputs.squeeze(1)
                
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def run_persistence_baseline(loader, step):
    """
    Evaluate the persistence baseline MSE on the test loader.
    Persistence predicts a_future = a_current (i.e. the last frame in the history).
    """
    total_loss = 0.0
    count = 0
    for inputs, targets in loader:
        # inputs shape: [batch, L, 20], targets shape: [batch, 20]
        # Current coefficient is inputs[:, -1, :] (the last time step in input history)
        current = inputs[:, -1, :]
        loss = np.mean((targets.numpy() - current.numpy()) ** 2)
        total_loss += loss * inputs.size(0)
        count += inputs.size(0)
    return total_loss / count


def main():
    parser = argparse.ArgumentParser(description="Train Phase 5 sequential models for Project RIPRA")
    parser.add_argument("--task", type=str, default="predict", choices=["predict", "classify", "parameter"], help="Sequence task to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lookback", type=int, default=10, help="History window length in frames")
    parser.add_argument("--step", type=int, default=1, help="Future prediction steps ahead (for predict task)")
    parser.add_argument("--dataset", type=str, default="data_ai/dataset.npz", help="Dataset path")
    parser.add_argument("--out_dir", type=str, default="ml_checkpoints/kaggle", help="Output checkpoints folder")
    args = parser.parse_args()
    
    # Configure UTF-8 encoding for Windows console printing
    try:
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    if not os.path.exists(args.dataset):
        print(f"Error: {args.dataset} not found. Please download it or generate one first.")
        return
        
    # 1. Load dataset
    print(f"Loading sequence dataset for task '{args.task}' (lookback={args.lookback}, step={args.step})...")
    full_dataset = SHSequenceDataset(args.dataset, lookback=args.lookback, step=args.step, task=args.task)
    
    # Train / Val / Test split (80% / 10% / 10%) at sequence level
    train_set, val_set, test_set = SHSequenceDataset.split_by_sequence(
        full_dataset, train_ratio=0.8, val_ratio=0.1, seed=42
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    print(f"Dataset split: Train={train_len}, Val={val_len}, Test={test_len}")
    
    # 2. Setup model and loss criterion
    if args.task == 'predict':
        model = WavefrontLSTM(input_dim=20, hidden_dim=128, output_dim=20)
        criterion = nn.MSELoss()
    elif args.task == 'classify':
        model = TurbulenceClassifierLSTM(input_dim=20, hidden_dim=64, num_classes=3)
        criterion = nn.CrossEntropyLoss()
    elif args.task == 'parameter':
        model = TurbulenceParameterEstimator(input_dim=254, hidden_dim=128, output_dim=1)
        criterion = nn.MSELoss()
        
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.out_dir, f"best_sequence_{args.task}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            
    # 3. Final evaluation
    checkpoint_path = os.path.join(args.out_dir, f"best_sequence_{args.task}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss on Best Checkpoint: {test_loss:.6f}")
    
    # Post-training task-specific analysis
    if args.task == 'predict':
        # Evaluate against persistence baseline on test set
        # persistence loader needs to use CPU and raw tensors
        test_loader_cpu = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        persistence_mse = run_persistence_baseline(test_loader_cpu, args.step)
        print(f"Persistence Baseline Test MSE: {persistence_mse:.6f}")
        improvement = (persistence_mse - test_loss) / persistence_mse * 100
        print(f"LSTM Improvement over Persistence: {improvement:.2f}%")
        
        # Spot check single sequence prediction
        with torch.no_grad():
            inputs, targets = next(iter(test_loader))
            inputs = inputs.to(device)
            preds = model(inputs).cpu().numpy()
            targets = targets.numpy()
            print("\nWavefront Prediction Spot Check (First 3 modes of sample 1):")
            print(f"  True future: {targets[0][:3]}")
            print(f"  LSTM Pred:   {preds[0][:3]}")
            print(f"  Persistence: {inputs[0, -1, :3].cpu().numpy()}")
            
    elif args.task == 'classify':
        # Compute accuracy
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        accuracy = correct / total * 100
        print(f"Turbulence Classification Test Accuracy: {accuracy:.2f}%")
        
    elif args.task == 'parameter':
        # Compute R-squared correlation
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).squeeze(1)
                all_targets.extend(targets.numpy())
                all_preds.extend(outputs.cpu().numpy())
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        
        # R2 calculation
        y_mean = np.mean(all_targets)
        ss_tot = np.sum((all_targets - y_mean) ** 2)
        ss_res = np.sum((all_targets - all_preds) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        print(f"Turbulence Parameter (D/r_0) Regression R^2 Score: {r2:.4f}")
        print(f"Average target D/r_0: {np.mean(all_targets):.2f}, RMSE: {np.sqrt(np.mean((all_targets - all_preds) ** 2)):.4f}")

if __name__ == "__main__":
    main()
