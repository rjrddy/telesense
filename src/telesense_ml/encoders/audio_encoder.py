"""
Audio Encoder using Wav2Vec2.0 with temporal transformer layers.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


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


class AudioEncoder(nn.Module):
    """
    Audio encoder using Wav2Vec2.0 with temporal transformer.
    """
    
    def __init__(
        self,
        model_name="facebook/wav2vec2-base",
        hidden_size=768,
        num_temporal_layers=2,
        dropout=0.1,
        freeze_base=False
    ):
        """
        Initialize the AudioEncoder.
        Input:
            model_name - HuggingFace Wav2Vec2 checkpoint name
            hidden_size - output feature dimension
            num_temporal_layers - transformer layers stacked on Wav2Vec2 (0 disables)
            dropout - dropout probability
            freeze_base - freeze the pretrained Wav2Vec2 weights when True
        """
        super().__init__()
        
        # Load pre-trained Wav2Vec2.0
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_size = hidden_size
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
        
        # Temporal transformer for sequence modeling
        self.temporal_transformer = TemporalTransformer(
            hidden_size=hidden_size,
            num_layers=num_temporal_layers,
            num_heads=8,  # hidden_size must be divisible by num_heads
            dropout=dropout
        )
        
        # Projection layer if needed
        if self.wav2vec2.config.hidden_size != hidden_size:
            self.projection = nn.Linear(
                self.wav2vec2.config.hidden_size,
                hidden_size
            )
        else:
            self.projection = nn.Identity()
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, audio_waveform, attention_mask=None):
        """
        Encode audio waveform.
        Input:
            audio_waveform - Raw audio waveform tensor of shape (batch, seq_len)
            attention_mask - Optional attention mask

        Output:
            Encoded features of shape (batch, seq_len, hidden_size)
        """
        # Extract features using Wav2Vec2.0
        outputs = self.wav2vec2(
            input_values=audio_waveform,
            attention_mask=attention_mask
        )
        
        # Get hidden states (batch, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state
        
        # Project to desired hidden size
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Apply temporal transformer
        hidden_states = self.temporal_transformer(hidden_states)
        
        return hidden_states
    
    def get_output_dim(self):
        """
        Get output dimension.
        Input:
            None

        Output:
            Output feature dimension (int)
        """
        return self.hidden_size

