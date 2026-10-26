"""
Temporal Variational Autoencoder for unsupervised behavior pattern learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TemporalEncoder(nn.Module):
    """
    Encoder for temporal VAE.
    """
    
    def __init__(
        self,
        input_dim,
        latent_dim=128,
        hidden_dims=[256, 512],
        num_temporal_layers=3,
        dropout=0.1
    ):
        """
        Initialize the TemporalEncoder.
        Input:
            input_dim - input feature dimension
            latent_dim - latent space dimension
            hidden_dims - sizes of the projection layers
            num_temporal_layers - number of transformer layers
            dropout - dropout probability
        """
        super().__init__()
        
        # Temporal convolutional layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Transformer for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=prev_dim,
            nhead=8,
            dim_feedforward=prev_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_temporal_layers)
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x):
        """
        Forward.
        Input:
            x - Input features of shape (batch, time, input_dim)

        Output:
            mu: Mean of latent distribution (batch, time, latent_dim)
            logvar: Log variance of latent distribution (batch, time, latent_dim)
        """
        # x: (batch, time, input_dim) -> (batch, input_dim, time)
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Transpose back: (batch, hidden_dim, time) -> (batch, time, hidden_dim)
        x = x.transpose(1, 2)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Compute mean and log variance
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class TemporalDecoder(nn.Module):
    """
    Decoder for temporal VAE.
    """
    
    def __init__(
        self,
        latent_dim=128,
        output_dim=512,
        hidden_dims=[512, 256],
        num_temporal_layers=3,
        dropout=0.1
    ):
        """
        Initialize the TemporalDecoder.
        Input:
            latent_dim - latent space dimension
            output_dim - reconstructed feature dimension
            hidden_dims - sizes of the projection layers
            num_temporal_layers - number of transformer layers
            dropout - dropout probability
        """
        super().__init__()
        
        # Transformer for temporal modeling
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=8,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_temporal_layers)
        
        # Temporal convolutional layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(prev_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Conv1d(prev_dim, output_dim, kernel_size=3, padding=1))
        
        self.conv_layers = nn.Sequential(*layers)
        
    def forward(self, z):
        """
        Forward.
        Input:
            z - Latent code of shape (batch, time, latent_dim)

        Output:
            Reconstructed features of shape (batch, time, output_dim)
        """
        # Apply transformer (self-attention)
        z = self.transformer(z, z)
        
        # Transpose for conv1d: (batch, time, latent_dim) -> (batch, latent_dim, time)
        z = z.transpose(1, 2)
        
        # Apply convolutional layers
        x = self.conv_layers(z)
        
        # Transpose back: (batch, output_dim, time) -> (batch, time, output_dim)
        x = x.transpose(1, 2)
        
        return x


class TemporalVAE(nn.Module):
    """
    Temporal Variational Autoencoder for learning normal behavior patterns.
    """
    
    def __init__(
        self,
        input_dim,
        latent_dim=128,
        encoder_hidden_dims=[256, 512],
        decoder_hidden_dims=[512, 256],
        num_temporal_layers=3,
        dropout=0.1
    ):
        """
        Initialize the TemporalVAE.
        Input:
            input_dim - input feature dimension
            latent_dim - latent space dimension
            encoder_hidden_dims - encoder projection layer sizes
            decoder_hidden_dims - decoder projection layer sizes
            num_temporal_layers - transformer layers in encoder and decoder
            dropout - dropout probability
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = TemporalEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            num_temporal_layers=num_temporal_layers,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = TemporalDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=decoder_hidden_dims,
            num_temporal_layers=num_temporal_layers,
            dropout=dropout
        )
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick.
        Input:
            mu - latent mean (batch, time, latent_dim)
            logvar - latent log-variance (batch, time, latent_dim)

        Output:
            Sampled latent code z = mu + eps * std
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Forward.
        Input:
            x - Input features of shape (batch, time, input_dim)

        Output:
            reconstructed: Reconstructed features (batch, time, input_dim)
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            z: Sampled latent code
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, logvar, z
    
    def encode(self, x):
        """
        Encode input to latent space.
        Input:
            x - input features (batch, time, input_dim)

        Output:
            Tuple (z, mu, logvar)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        """
        Decode latent code to features.
        Input:
            z - latent code (batch, time, latent_dim)

        Output:
            Reconstructed features (batch, time, output_dim)
        """
        return self.decoder(z)
    
    def compute_reconstruction_error(self, x, reconstructed):
        """
        Compute reconstruction error per frame.
        Input:
            x - Original features (batch, time, input_dim)
            reconstructed - Reconstructed features (batch, time, input_dim)

        Output:
            Reconstruction error per frame (batch, time)
        """
        # MSE per frame
        error = F.mse_loss(reconstructed, x, reduction='none')
        # Sum over feature dimension, keep time dimension
        error = error.sum(dim=-1)  # (batch, time)
        return error

