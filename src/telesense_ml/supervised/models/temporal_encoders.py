"""
Multi-scale temporal encoders for temporal localization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ShortTermEncoder(nn.Module):
    """
    Short-term temporal encoder using 3D CNN (1-5 second windows).
    """
    
    def __init__(self, input_dim=2048, hidden_dim=512, kernel_size=(3, 3, 3)):
        """
        Initialize the ShortTermEncoder.
        Input:
            input_dim - input channel dimension
            hidden_dim - output feature dimension
            kernel_size - 3D convolution kernel size (time, height, width)
        """
        super().__init__()
        # Simplified 3D CNN for short-term patterns
        self.conv3d = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim, kernel_size=kernel_size, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Pool spatial dims
        
    def forward(self, x):
        """
        Forward.
        Input:
            x - Input features of shape (batch, channels, time, height, width)

        Output:
            Temporal features of shape (batch, time, hidden_dim)
        """
        # x: (batch, C, T, H, W)
        x = self.conv3d(x)  # (batch, hidden_dim, T, H', W')
        x = self.pool(x)  # (batch, hidden_dim, T, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (batch, hidden_dim, T)
        x = x.transpose(1, 2)  # (batch, T, hidden_dim)
        return x


class TemporalConvolutionalNetwork(nn.Module):
    """
    Temporal Convolutional Network for medium-term patterns (5-30 seconds).
    """
    
    def __init__(self, input_dim, hidden_dim=512, num_layers=4, kernel_size=3, dropout=0.1):
        """
        Initialize the TemporalConvolutionalNetwork.
        Input:
            input_dim - input feature dimension
            hidden_dim - output feature dimension
            num_layers - number of dilated convolution layers
            kernel_size - convolution kernel size
            dropout - dropout probability
        """
        super().__init__()
        layers = []
        
        # Dilated convolutions for multi-scale temporal modeling
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        input_dim if i == 0 else hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        padding=(kernel_size - 1) * dilation // 2
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Forward.
        Input:
            x - Input features of shape (batch, time, input_dim)

        Output:
            Temporal features of shape (batch, time, hidden_dim)
        """
        # Transpose for conv1d: (batch, time, dim) -> (batch, dim, time)
        x = x.transpose(1, 2)
        
        for layer in self.layers:
            residual = x
            x = layer(x)
            # Residual connection if dimensions match
            if x.shape == residual.shape:
                x = x + residual
        
        # Transpose back: (batch, dim, time) -> (batch, time, dim)
        x = x.transpose(1, 2)
        x = self.norm(x)
        
        return x


class LongTermTransformerEncoder(nn.Module):
    """
    Long-term temporal encoder using Transformer (30+ seconds).
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        max_seq_len=1000
    ):
        """
        Initialize the LongTermTransformerEncoder.
        Input:
            input_dim - input feature dimension
            hidden_dim - output feature dimension
            num_heads - number of attention heads
            num_layers - number of transformer layers
            dropout - dropout probability
            max_seq_len - maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Projection to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward.
        Input:
            x - Input features of shape (batch, time, input_dim)

        Output:
            Temporal features of shape (batch, time, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        return x


class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-scale temporal encoder combining short, medium, and long-term patterns.
    """
    
    def __init__(
        self,
        input_dim=2048,
        hidden_dim=512,
        short_term_dim=512,
        medium_term_dim=512,
        long_term_dim=512,
        dropout=0.1
    ):
        """
        Initialize the MultiScaleTemporalEncoder.
        Input:
            input_dim - input feature dimension
            hidden_dim - fused output dimension
            short_term_dim - short-term branch dimension
            medium_term_dim - TCN (medium-term) branch dimension
            long_term_dim - transformer (long-term) branch dimension
            dropout - dropout probability
        """
        super().__init__()
        
        # Short-term encoder (for 2D features, we'll use TCN directly)
        # If we have 3D video, use ShortTermEncoder
        self.use_3d = False  # Set to True if using 3D video input
        
        # Medium-term: TCN
        self.medium_term = TemporalConvolutionalNetwork(
            input_dim=input_dim,
            hidden_dim=medium_term_dim,
            dropout=dropout
        )
        
        # Long-term: Transformer
        self.long_term = LongTermTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=long_term_dim,
            dropout=dropout
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(medium_term_dim + long_term_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Forward.
        Input:
            x - Input features of shape (batch, time, input_dim)

        Output:
            Fused temporal features of shape (batch, time, hidden_dim)
        """
        # Medium-term features
        medium_features = self.medium_term(x)
        
        # Long-term features
        long_features = self.long_term(x)
        
        # Concatenate and fuse
        combined = torch.cat([medium_features, long_features], dim=-1)
        fused = self.fusion(combined)
        
        return fused

