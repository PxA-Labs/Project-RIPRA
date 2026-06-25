# ml/models.py - PyTorch Neural Network Architectures for Wavefront Reconstruction
import torch
import torch.nn as nn

class WavefrontMLP(nn.Module):
    """
    Fully Connected Multi-Layer Perceptron Baseline model.
    Maps 1D vector of displacements directly to Zernike coefficients.
    """
    def __init__(self, input_dim=254, output_dim=20, hidden_dims=[512, 256, 128], dropout=0.1):
        super(WavefrontMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    """
    Standard Residual block with BatchNorm and Relu activation.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class WavefrontCNN(nn.Module):
    """
    Spatial 2D Convolutional Neural Network Reconstructor.
    Maps 2-channel 2D grid of spot displacements (dx, dy) to Zernike coefficients.
    """
    def __init__(self, output_dim=20):
        super(WavefrontCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        self.res1 = ResBlock(32)
        self.res2 = ResBlock(32)
        
        # Downsample using strided convolution (13x13 -> 7x7)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.res3 = ResBlock(64)
        
        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.res3(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
