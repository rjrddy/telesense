"""
Visual Encoder using ResNet/DenseNet with temporal attention pooling.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class TemporalAttentionPooling(nn.Module):
    """
    Temporal attention pooling for video sequences.
    """
    
    def __init__(self, hidden_size, dropout=0.1):
        """
        Initialize the TemporalAttentionPooling.
        Input:
            hidden_size - feature dimension
            dropout - dropout probability
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Apply temporal attention pooling.
        Input:
            x - Input tensor of shape (batch, seq_len, hidden_size)

        Output:
            Pooled tensor of shape (batch, hidden_size)
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        pooled = torch.sum(attn_weights * x, dim=1)  # (batch, hidden_size)
        
        return pooled


class TemporalTransformer(nn.Module):
    """
    Lightweight transformer for temporal modeling.
    """
    
    def __init__(self, hidden_size, num_layers, num_heads, dropout=0.1):
        """
        Initialize the TemporalTransformer.
        Input:
            hidden_size - feature dimension
            num_layers - number of transformer layers
            num_heads - number of attention heads
            dropout - dropout probability
        """
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        """
        Forward pass through temporal transformer layers.
        Input:
            x - Input tensor of shape (batch, seq_len, hidden_size)

        Output:
            Output tensor of same shape
        """
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class VisualEncoder(nn.Module):
    """
    Visual encoder using ResNet/DenseNet with temporal modeling.
    """
    
    def __init__(
        self,
        encoder="resnet50",
        pretrained=True,
        hidden_size=2048,
        num_temporal_layers=2,
        dropout=0.1,
        freeze_base=False
    ):
        """
        Initialize the VisualEncoder.
        Input:
            encoder - backbone name ('resnet50' or 'densenet121')
            pretrained - load ImageNet weights when True
            hidden_size - output feature dimension
            num_temporal_layers - transformer layers over the frame sequence (0 disables)
            dropout - dropout probability
            freeze_base - freeze the backbone weights when True
        """
        super().__init__()
        
        # Load pre-trained backbone
        if encoder == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            # Remove final fully connected layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_dim = 2048
        elif encoder == "densenet121":
            backbone = models.densenet121(pretrained=pretrained)
            # Remove classifier
            self.backbone = backbone.features
            backbone_dim = 1024
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection to desired hidden size
        self.projection = nn.Linear(backbone_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            hidden_size=hidden_size,
            num_layers=num_temporal_layers,
            num_heads=8,
            dropout=dropout
        )
        
        # Temporal attention pooling
        self.temporal_pooling = TemporalAttentionPooling(
            hidden_size=hidden_size,
            dropout=dropout
        )
        
        self.hidden_size = hidden_size
        
    def forward(self, video_frames, return_sequence=False):
        """
        Encode video frames.
        Input:
            video_frames - Video frames tensor of shape (batch, seq_len, C, H, W)
            return_sequence - If True, return full sequence; else return pooled

        Output:
            If return_sequence: (batch, seq_len, hidden_size)
            Else: (batch, hidden_size)
        """
        batch_size, seq_len = video_frames.shape[:2]
        
        # Reshape for batch processing
        video_frames = video_frames.view(batch_size * seq_len, *video_frames.shape[2:])
        
        # Extract features from each frame
        with torch.set_grad_enabled(self.training):
            features = self.backbone(video_frames)
            # Global average pooling
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(batch_size * seq_len, -1)
        
        # Project to hidden size
        features = self.projection(features)
        features = self.dropout(features)
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, self.hidden_size)
        
        # Apply temporal transformer
        features = self.temporal_transformer(features)
        
        if return_sequence:
            return features
        else:
            # Apply temporal attention pooling
            pooled = self.temporal_pooling(features)
            return pooled
    
    def get_output_dim(self):
        """
        Get output dimension.
        Input:
            None

        Output:
            Output feature dimension (int)
        """
        return self.hidden_size

