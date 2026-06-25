# ml/sequence_models.py - LSTM architectures for Phase 5 wavefront prediction & classification
import torch
import torch.nn as nn

class WavefrontLSTM(nn.Module):
    """
    LSTM Model for Future Wavefront Prediction (Checkpoint 5.1).
    Takes a history sequence of Zernike coefficients (shape: [batch, seq_len, 20])
    and predicts the Zernike coefficients at a future step (shape: [batch, 20]).
    """
    def __init__(self, input_dim=20, hidden_dim=128, output_dim=20, num_layers=2, dropout=0.1):
        super(WavefrontLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        lstm_out, _ = self.lstm(x)
        
        # Take the hidden state of the last time step
        last_hidden = lstm_out[:, -1, :] # shape: [batch_size, hidden_dim]
        
        # Predict future coefficients
        out = self.fc(last_hidden) # shape: [batch_size, output_dim]
        return out


class TurbulenceClassifierLSTM(nn.Module):
    """
    LSTM Model for Turbulence Classification (Checkpoint 5.2).
    Takes a sequence of Zernike coefficients (shape: [batch, seq_len, 20])
    and outputs logits for Weak, Moderate, or Strong turbulence regimes.
    """
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=3, num_layers=2):
        super(TurbulenceClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :] # shape: [batch_size, hidden_dim]
        logits = self.fc(last_hidden)     # shape: [batch_size, num_classes]
        return logits


class TurbulenceParameterEstimator(nn.Module):
    """
    LSTM Model for Turbulence Parameter Estimation (Checkpoint 5.3).
    Takes a history sequence of spot displacements (shape: [batch, seq_len, 254])
    and performs regression to predict the physical parameters [D/r_0, tau_0].
    """
    def __init__(self, input_dim=254, hidden_dim=128, output_dim=2, num_layers=2):
        super(TurbulenceParameterEstimator, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :] # shape: [batch_size, hidden_dim]
        out = self.fc(last_hidden)        # shape: [batch_size, output_dim]
        return out
