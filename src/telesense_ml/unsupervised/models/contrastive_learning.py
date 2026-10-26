"""
Self-supervised contrastive learning for temporal representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveTemporalEncoder(nn.Module):
    """
    Temporal encoder with contrastive learning.
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        projection_dim=128,
        num_temporal_layers=3,
        dropout=0.1,
        temperature=0.07
    ):
        """
        Initialize the ContrastiveTemporalEncoder.
        Input:
            input_dim - input feature dimension
            hidden_dim - encoder hidden dimension
            projection_dim - dimension of the contrastive projection
            num_temporal_layers - number of transformer layers
            dropout - dropout probability
            temperature - softmax temperature for the contrastive loss
        """
        super().__init__()
        
        self.temperature = temperature
        
        # Temporal encoder
        layers = []
        prev_dim = input_dim
        
        for _ in range(num_temporal_layers):
            layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.temporal_encoder = nn.Sequential(*layers)
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
    def forward(self, x):
        """
        Forward.
        Input:
            x - Input features of shape (batch, time, input_dim)

        Output:
            Projected features of shape (batch, time, projection_dim)
        """
        # Transpose for conv1d: (batch, time, input_dim) -> (batch, input_dim, time)
        x = x.transpose(1, 2)
        
        # Encode
        x = self.temporal_encoder(x)
        
        # Transpose back: (batch, hidden_dim, time) -> (batch, time, hidden_dim)
        x = x.transpose(1, 2)
        
        # Project
        x = self.projection(x)
        
        # L2 normalize
        x = F.normalize(x, p=2, dim=-1)
        
        return x
    
    def contrastive_loss(self, anchor, positive, negative):
        """
        Compute contrastive loss.
        Input:
            anchor - Anchor features (batch, time, projection_dim)
            positive - Positive features (batch, time, projection_dim)
            negative - Negative features (batch, time, projection_dim)

        Output:
            Contrastive loss
        """
        # Compute similarities
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)  # (batch, time)
        neg_sim = F.cosine_similarity(anchor, negative, dim=-1)  # (batch, time)
        
        # Apply temperature
        pos_sim = pos_sim / self.temperature
        neg_sim = neg_sim / self.temperature
        
        # Contrastive loss: -log(exp(pos) / (exp(pos) + exp(neg)))
        loss = -torch.log(torch.sigmoid(pos_sim - neg_sim) + 1e-8)
        
        return loss.mean()



