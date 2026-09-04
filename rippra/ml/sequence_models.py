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


class SlopeCompletionLSTM(nn.Module):
    """
    Slope-domain sequence completion model (umbrella #90 / AI fallback).

    Reconstructs missing or corrupted sub-aperture slopes (dx, dy) from
    temporal context.  Input is a [batch, L+1, 4*N] tensor:
        * observed_dx, observed_dy   zero-filled observed slopes
        * mask_x, mask_y             1 = valid spot, 0 = lost/corrupted
    Output is a [batch, 2*N] complete slope vector for the *current* frame.

    Mask-conditioning is built into the input; the model learns where it is
    allowed to be uncertain and propagates that through the recurrence.
    """
    def __init__(self, nspots=137, hidden_dim=128, num_layers=2, dropout=0.1):
        super(SlopeCompletionLSTM, self).__init__()
        self.nspots = nspots
        input_dim = 4 * nspots  # observed dx/dy + masks for both axes
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Output head predicts residual (delta) around a simple temporal prior.
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 2 * nspots),
        )

    def forward(self, x):
        # x: [batch, L+1, 4*nspots]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]       # [batch, hidden_dim]
        delta = self.fc(last_hidden)            # [batch, 2*nspots]
        # Add persistence prior: use the last frame's observed slopes as baseline.
        # The last input row contains observed_dx, observed_dy, mask, inv_mask.
        n = self.nspots
        prior = x[:, -1, :2*n].clone()
        return prior + delta
