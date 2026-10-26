"""
Cross-Modal Attention Fusion Mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing multiple modalities.
    """
    
    def __init__(
        self,
        hidden_size=768,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    ):
        """
        Initialize the CrossModalAttention.
        Input:
            hidden_size - shared feature dimension (must be divisible by num_heads)
            num_heads - number of attention heads
            num_layers - number of cross-attention layers
            dropout - dropout probability
        """
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Multi-head attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'audio_to_visual': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'visual_to_audio': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'audio_to_text': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'text_to_audio': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'visual_to_text': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'text_to_visual': nn.MultiheadAttention(
                    embed_dim=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'norm1': nn.LayerNorm(hidden_size),
                'norm2': nn.LayerNorm(hidden_size),
                'norm3': nn.LayerNorm(hidden_size),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * 4, hidden_size),
                    nn.Dropout(dropout)
                )
            })
            for _ in range(num_layers)
        ])
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, audio_features, visual_features, text_features):
        """
        Fuse features from three modalities using cross-modal attention.
        Input:
            audio_features - Audio features of shape (batch, seq_len_a, hidden_size)
            visual_features - Visual features of shape (batch, seq_len_v, hidden_size)
            text_features - Text features of shape (batch, seq_len_t, hidden_size)

        Output:
            Fused features of shape (batch, hidden_size)
        """
        # If visual_features is 2D (already pooled), add sequence dimension
        if len(visual_features.shape) == 2:
            visual_features = visual_features.unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Apply cross-modal attention layers
        audio_out = audio_features
        visual_out = visual_features
        text_out = text_features
        
        for layer in self.layers:
            # Audio-Visual cross-attention
            audio_attn, _ = layer['audio_to_visual'](
                audio_out, visual_out, visual_out
            )
            audio_out = layer['norm1'](audio_out + audio_attn)
            audio_out = layer['norm1'](audio_out + layer['ffn'](audio_out))
            
            visual_attn, _ = layer['visual_to_audio'](
                visual_out, audio_out, audio_out
            )
            visual_out = layer['norm2'](visual_out + visual_attn)
            visual_out = layer['norm2'](visual_out + layer['ffn'](visual_out))
            
            # Audio-Text cross-attention
            audio_attn, _ = layer['audio_to_text'](
                audio_out, text_out, text_out
            )
            audio_out = layer['norm1'](audio_out + audio_attn)
            
            text_attn, _ = layer['text_to_audio'](
                text_out, audio_out, audio_out
            )
            text_out = layer['norm3'](text_out + text_attn)
            text_out = layer['norm3'](text_out + layer['ffn'](text_out))
            
            # Visual-Text cross-attention
            visual_attn, _ = layer['visual_to_text'](
                visual_out, text_out, text_out
            )
            visual_out = layer['norm2'](visual_out + visual_attn)
            
            text_attn, _ = layer['text_to_visual'](
                text_out, visual_out, visual_out
            )
            text_out = layer['norm3'](text_out + text_attn)
        
        # Pool sequences (mean pooling with attention)
        audio_pooled = self._pool_sequence(audio_out)
        visual_pooled = self._pool_sequence(visual_out)
        text_pooled = self._pool_sequence(text_out)
        
        # Concatenate and fuse
        concatenated = torch.cat([audio_pooled, visual_pooled, text_pooled], dim=-1)
        fused = self.fusion(concatenated)
        
        return fused
    
    def _pool_sequence(self, x):
        """
        Pool sequence to single vector using mean pooling.
        Input:
            x - Input tensor of shape (batch, seq_len, hidden_size)

        Output:
            Pooled tensor of shape (batch, hidden_size)
        """
        return torch.mean(x, dim=1)

