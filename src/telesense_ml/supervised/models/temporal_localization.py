"""
Temporal localization head for dense frame-level prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for highlighting important timepoints.
    """
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        """
        Initialize the TemporalAttention.
        Input:
            hidden_dim - feature dimension
            num_heads - number of attention heads
            dropout - dropout probability
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward.
        Input:
            x - Input features of shape (batch, time, hidden_dim)

        Output:
            Attended features of shape (batch, time, hidden_dim)
        """
        residual = x
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(residual + self.dropout(attn_out))
        return x


class TemporalLocalizationHead(nn.Module):
    """
    Dense prediction head for frame-level concern detection.
    """
    
    def __init__(
        self,
        input_dim,
        num_concern_types=7,
        hidden_dims=[256, 128],
        dropout=0.2
    ):
        """
        Initialize the TemporalLocalizationHead.
        Input:
            input_dim - input feature dimension
            num_concern_types - number of concern classes predicted per frame
            hidden_dims - sizes of the hidden MLP layers
            dropout - dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final prediction layer (per frame, per concern type)
        layers.append(nn.Linear(prev_dim, num_concern_types))
        
        self.predictor = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward.
        Input:
            x - Temporal features of shape (batch, time, input_dim)

        Output:
            Frame-level scores of shape (batch, time, num_concern_types)
        """
        return self.predictor(x)


class TemporalSmoothing(nn.Module):
    """
    Temporal smoothing to reduce prediction jitter.
    """
    
    def __init__(self, window_size=5, mode='gaussian'):
        """
        Initialize the TemporalSmoothing.
        Input:
            window_size - smoothing window in frames
            mode - smoothing kernel ('gaussian' or moving average)
        """
        super().__init__()
        self.window_size = window_size
        self.mode = mode
        
        if mode == 'gaussian':
            # Create Gaussian kernel
            kernel = torch.exp(-torch.arange(-window_size//2, window_size//2 + 1).float() ** 2 / 2)
            kernel = kernel / kernel.sum()
            self.register_buffer('kernel', kernel.view(1, 1, -1))
        else:
            # Uniform kernel
            self.register_buffer('kernel', torch.ones(1, 1, window_size) / window_size)
    
    def forward(self, x):
        """
        Forward.
        Input:
            x - Predictions of shape (batch, time, num_classes)

        Output:
            Smoothed predictions of same shape
        """
        # Apply 1D convolution for temporal smoothing
        # x: (batch, time, num_classes) -> (batch, num_classes, time)
        x = x.transpose(1, 2)
        
        # Pad for same-length output
        padding = self.window_size // 2
        x = F.pad(x, (padding, padding), mode='replicate')
        
        # Apply convolution
        x = F.conv1d(x, self.kernel.expand(x.shape[1], -1, -1), groups=x.shape[1])
        
        # Transpose back: (batch, num_classes, time) -> (batch, time, num_classes)
        x = x.transpose(1, 2)
        
        return x


class TemporalLocalizationModel(nn.Module):
    """
    Complete temporal localization model.
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        num_concern_types=7,
        use_attention=True,
        use_smoothing=True,
        smoothing_window=5,
        dropout=0.1
    ):
        """
        Initialize the TemporalLocalizationModel.
        Input:
            input_dim - input feature dimension
            hidden_dim - hidden feature dimension
            num_concern_types - number of concern classes
            use_attention - apply temporal self-attention when True
            use_smoothing - smooth frame scores when True
            smoothing_window - smoothing window in frames
            dropout - dropout probability
        """
        super().__init__()
        
        # Temporal attention (optional)
        if use_attention:
            self.temporal_attention = TemporalAttention(
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else:
            self.temporal_attention = None
        
        # Localization head
        self.localization_head = TemporalLocalizationHead(
            input_dim=hidden_dim,
            num_concern_types=num_concern_types,
            dropout=dropout
        )
        
        # Temporal smoothing (optional)
        if use_smoothing:
            self.smoothing = TemporalSmoothing(
                window_size=smoothing_window,
                mode='gaussian'
            )
        else:
            self.smoothing = None
    
    def forward(self, temporal_features):
        """
        Forward.
        Input:
            temporal_features - Features of shape (batch, time, hidden_dim)

        Output:
            Frame-level concern scores of shape (batch, time, num_concern_types)
        """
        # Apply temporal attention if enabled
        if self.temporal_attention is not None:
            temporal_features = self.temporal_attention(temporal_features)
        
        # Dense prediction
        frame_scores = self.localization_head(temporal_features)
        
        # Apply temporal smoothing if enabled
        if self.smoothing is not None:
            frame_scores = self.smoothing(frame_scores)
        
        return frame_scores

