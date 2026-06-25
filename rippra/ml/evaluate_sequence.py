# ml/evaluate_sequence.py - Evaluate Phase 5 sequential model performance on the test dataset
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add current directory to sys.path
sys.path.append(os.path.dirname(__file__))

from sequence_models import WavefrontLSTM, TurbulenceClassifierLSTM, TurbulenceParameterEstimator
from train_sequence import SHSequenceDataset, run_persistence_baseline

def main():
    # Configure UTF-8 encoding for Windows console printing
    try:
        import sys
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sequence Evaluation running on device: {device}")
    
    # Paths setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data_ai", "dataset.npz")
    checkpoint_dir = os.path.join(base_dir, "ml_checkpoints", "kaggle")
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please place the downloaded dataset.npz there.")
        return
        
    print(f"Loading local dataset from {dataset_path}...")
    
    # 1. Evaluate Wavefront Predictor LSTM
    predict_ckpt = os.path.join(checkpoint_dir, "best_sequence_predict.pt")
    if os.path.exists(predict_ckpt):
        print("\n" + "="*80)
        print(" 1. FUTURE WAVEFRONT PREDICTION EVALUATION")
        print("="*80)
        
        # Load dataset split
        dataset = SHSequenceDataset(dataset_path, lookback=10, step=1, task='predict')
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len
        _, _, test_set = random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
        
        # Load model
        model = WavefrontLSTM(input_dim=20, hidden_dim=128, output_dim=20).to(device)
        checkpoint = torch.load(predict_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluate MSE
        running_loss = 0.0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
        lstm_mse = running_loss / len(test_set)
        
        # Evaluate Persistence Baseline
        test_loader_cpu = DataLoader(test_set, batch_size=128, shuffle=False)
        persistence_mse = run_persistence_baseline(test_loader_cpu, step=1)
        improvement = (persistence_mse - lstm_mse) / persistence_mse * 100
        
        print(f"  LSTM Prediction Test MSE:  {lstm_mse:.6f}")
        print(f"  Persistence Baseline MSE:  {persistence_mse:.6f}")
        print(f"  LSTM Improvement:          {improvement:.2f}%")
        
        # Spot check samples
        with torch.no_grad():
            inputs, targets = next(iter(test_loader))
            inputs = inputs.to(device)
            preds = model(inputs).cpu().numpy()
            targets = targets.numpy()
            print("\n  Wavefront Prediction Spot Check (First 3 modes of sample 1):")
            print(f"    True future: {targets[0][:3]}")
            print(f"    LSTM Pred:   {preds[0][:3]}")
            print(f"    Persistence: {inputs[0, -1, :3].cpu().numpy()}")
    else:
        print(f"\nSkip Wavefront Prediction: Checkpoint {predict_ckpt} not found.")

    # 2. Evaluate Turbulence Classifier LSTM
    classify_ckpt = os.path.join(checkpoint_dir, "best_sequence_classify.pt")
    if os.path.exists(classify_ckpt):
        print("\n" + "="*80)
        print(" 2. TURBULENCE REGIME CLASSIFICATION EVALUATION")
        print("="*80)
        
        dataset = SHSequenceDataset(dataset_path, lookback=10, step=1, task='classify')
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len
        _, _, test_set = random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
        
        model = TurbulenceClassifierLSTM(input_dim=20, hidden_dim=64, num_classes=3).to(device)
        checkpoint = torch.load(classify_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        correct = 0
        total = 0
        conf_matrix = np.zeros((3, 3), dtype=int) # rows=true, cols=pred
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                
                # Update confusion matrix
                for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy()):
                    conf_matrix[t, p] += 1
                    
        accuracy = correct / total * 100
        print(f"  Classification Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
        print("\n  Confusion Matrix (Rows=True, Cols=Predicted):")
        print("               Weak  Mod  Strong")
        print(f"    Weak:    {conf_matrix[0, 0]:5d} {conf_matrix[0, 1]:4d} {conf_matrix[0, 2]:7d}")
        print(f"    Mod:     {conf_matrix[1, 0]:5d} {conf_matrix[1, 1]:4d} {conf_matrix[1, 2]:7d}")
        print(f"    Strong:  {conf_matrix[2, 0]:5d} {conf_matrix[2, 1]:4d} {conf_matrix[2, 2]:7d}")
    else:
        print(f"\nSkip Turbulence Classification: Checkpoint {classify_ckpt} not found.")

    # 3. Evaluate Turbulence Parameter Regression LSTM
    param_ckpt = os.path.join(checkpoint_dir, "best_sequence_parameter.pt")
    if os.path.exists(param_ckpt):
        print("\n" + "="*80)
        print(" 3. TURBULENCE PARAMETER REGRESSION EVALUATION (D/r_0)")
        print("="*80)
        
        dataset = SHSequenceDataset(dataset_path, lookback=10, step=1, task='parameter')
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len
        _, _, test_set = random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
        
        model = TurbulenceParameterEstimator(input_dim=254, hidden_dim=128, output_dim=1).to(device)
        checkpoint = torch.load(param_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
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
        
        y_mean = np.mean(all_targets)
        ss_tot = np.sum((all_targets - y_mean) ** 2)
        ss_res = np.sum((all_targets - all_preds) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean((all_targets - all_preds) ** 2))
        mae = np.mean(np.abs(all_targets - all_preds))
        
        print(f"  R-squared Correlation Score (R²): {r2:.4f}")
        print(f"  Root Mean Squared Error (RMSE):    {rmse:.4f}")
        print(f"  Mean Absolute Error (MAE):         {mae:.4f}")
        print(f"  Average Target D/r_0 value:        {y_mean:.2f}")
        
        print("\n  Physical Parameter Regression Spot Check:")
        for idx in range(min(5, len(all_targets))):
            true_idx = idx * 1000 # Sample from different sequences
            if true_idx < len(all_targets):
                print(f"    True D/r_0: {all_targets[true_idx]:5.2f} | Predicted D/r_0: {all_preds[true_idx]:5.2f}")
    else:
        print(f"\nSkip Parameter Regression: Checkpoint {param_ckpt} not found.")
        
    print("\n" + "="*80)
    print("                     EVALUATION COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    main()
