"""
Complete unsupervised temporal localization model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from telesense_ml.encoders import AudioEncoder, VisualEncoder, TextEncoder
from .temporal_vae import TemporalVAE
from .contrastive_learning import ContrastiveTemporalEncoder


class FeatureExtractor(nn.Module):
    """
    Extract features from raw modalities.
    """
    
    def __init__(self, config):
        """
        Initialize the FeatureExtractor.
        Input:
            config - configuration dictionary (model section configures the encoders)
        """
        super().__init__()
        
        # Initialize encoders
        self.visual_encoder = VisualEncoder(
            encoder=config['model']['visual']['encoder'],
            pretrained=config['model']['visual']['pretrained'],
            hidden_size=config['model']['visual']['hidden_size'],
            num_temporal_layers=0,
            dropout=config['model']['visual']['dropout']
        )
        
        self.audio_encoder = AudioEncoder(
            model_name=config['model']['audio']['model_name'],
            hidden_size=config['model']['audio']['hidden_size'],
            num_temporal_layers=0,
            dropout=config['model']['audio']['dropout']
        )
        
        self.text_encoder = TextEncoder(
            model_name=config['model']['text']['model_name'],
            hidden_size=config['model']['text']['hidden_size'],
            max_length=config['model']['text']['max_length'],
            dropout=config['model']['text']['dropout']
        )
        
        # Projection to common dimension
        self.feature_dim = config['model']['unsupervised']['feature_dim']
        
        self.visual_proj = nn.Linear(
            self.visual_encoder.get_output_dim(),
            self.feature_dim
        )
        self.audio_proj = nn.Linear(
            self.audio_encoder.get_output_dim(),
            self.feature_dim
        )
        self.text_proj = nn.Linear(
            self.text_encoder.get_output_dim(),
            self.feature_dim
        )
    
    def forward(self, video_frames=None, audio_waveform=None, audio_mask=None,
                text_input_ids=None, text_mask=None):
        """
        Extract and combine features from modalities.
        Input:
            video_frames - video frames (batch, num_frames, C, H, W)
            audio_waveform - audio waveform (batch, samples)
            audio_mask - audio attention mask
            text_input_ids - text token IDs (batch, seq_len)
            text_mask - text attention mask

        Output:
            Concatenated per-frame features (batch, time, combined_dim)
        """
        features_list = []
        
        # Visual features
        if video_frames is not None:
            batch_size, num_frames = video_frames.shape[:2]
            visual_features = self.visual_encoder(video_frames, return_sequence=True)
            visual_features = self.visual_proj(visual_features)
            features_list.append(visual_features)
        
        # Audio features
        if audio_waveform is not None:
            audio_features = self.audio_encoder(audio_waveform, audio_mask)
            # Interpolate to match video frames if needed
            if video_frames is not None:
                num_frames = video_frames.shape[1]
                audio_features = F.interpolate(
                    audio_features.transpose(1, 2),
                    size=num_frames,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            audio_features = self.audio_proj(audio_features)
            features_list.append(audio_features)
        
        # Text features (expand to temporal dimension)
        if text_input_ids is not None:
            text_features = self.text_encoder(text_input_ids, text_mask)
            # Mean pool and expand
            if text_mask is not None:
                text_mask_expanded = text_mask.unsqueeze(-1).float()
                text_pooled = (text_features * text_mask_expanded).sum(dim=1) / text_mask_expanded.sum(dim=1)
            else:
                text_pooled = text_features.mean(dim=1)
            
            # Expand to temporal dimension
            if video_frames is not None:
                num_frames = video_frames.shape[1]
            elif audio_waveform is not None:
                num_frames = audio_features.shape[1]
            else:
                num_frames = 1
            
            text_pooled = text_pooled.unsqueeze(1).expand(-1, num_frames, -1)
            text_features = self.text_proj(text_pooled)
            features_list.append(text_features)
        
        # Combine features
        if len(features_list) == 0:
            raise ValueError("At least one modality must be provided")
        
        combined_features = torch.cat(features_list, dim=-1)
        return combined_features


class UnsupervisedTemporalModel(nn.Module):
    """
    Unsupervised temporal localization model using VAE and contrastive learning.

    Architecture:
    1. Feature extraction from modalities
    2. Contrastive pre-training (optional)
    3. Temporal VAE for normal pattern learning
    4. Reconstruction error-based anomaly detection
    """
    
    def __init__(self, config):
        """
        Initialize the UnsupervisedTemporalModel.
        Input:
            config - configuration dictionary (model and anomaly-detection sections)
        """
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(config)
        
        # Contrastive encoder (for pre-training)
        unsupervised_config = config['model']['unsupervised']
        self.contrastive_encoder = ContrastiveTemporalEncoder(
            input_dim=unsupervised_config['feature_dim'] * 3,  # Combined features
            hidden_dim=unsupervised_config['hidden_dim'],
            projection_dim=unsupervised_config['projection_dim'],
            num_temporal_layers=unsupervised_config['num_temporal_layers'],
            dropout=unsupervised_config['dropout'],
            temperature=unsupervised_config.get('temperature', 0.07)
        )
        
        # Temporal VAE
        self.vae = TemporalVAE(
            input_dim=unsupervised_config['feature_dim'] * 3,
            latent_dim=unsupervised_config['latent_dim'],
            encoder_hidden_dims=unsupervised_config['encoder_hidden_dims'],
            decoder_hidden_dims=unsupervised_config['decoder_hidden_dims'],
            num_temporal_layers=unsupervised_config['num_temporal_layers'],
            dropout=unsupervised_config['dropout']
        )
        
        self.config = config
    
    def forward(
        self,
        video_frames=None,
        audio_waveform=None,
        audio_mask=None,
        text_input_ids=None,
        text_mask=None,
        mode='vae'
    ):
        """
        Forward pass.
        Input:
            video_frames - Video frames tensor
            audio_waveform - Audio waveform tensor
            audio_mask - Audio attention mask
            text_input_ids - Text token IDs
            text_mask - Text attention mask
            mode - 'vae' for VAE training, 'contrastive' for contrastive learning

        Output:
            If mode='vae': (reconstructed, mu, logvar, z, features)
            If mode='contrastive': (projected_features,)
        """
        # Extract features
        features = self.feature_extractor(
            video_frames=video_frames,
            audio_waveform=audio_waveform,
            audio_mask=audio_mask,
            text_input_ids=text_input_ids,
            text_mask=text_mask
        )
        
        if mode == 'vae':
            # VAE forward pass
            reconstructed, mu, logvar, z = self.vae(features)
            return reconstructed, mu, logvar, z, features
        elif mode == 'contrastive':
            # Contrastive learning forward pass
            projected = self.contrastive_encoder(features)
            return projected, features
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def compute_anomaly_scores(self, features):
        """
        Compute anomaly scores based on reconstruction error.
        Input:
            features - Input features (batch, time, feature_dim)

        Output:
            Anomaly scores per frame (batch, time)
        """
        # Encode and decode
        reconstructed, _, _, _ = self.vae(features)
        
        # Compute reconstruction error
        anomaly_scores = self.vae.compute_reconstruction_error(features, reconstructed)
        
        return anomaly_scores
    
    def predict_anomalies(self, features, threshold=None):
        """
        Predict anomalous timepoints.
        Input:
            features - Input features (batch, time, feature_dim)
            threshold - Anomaly threshold (if None, use adaptive)

        Output:
            Dictionary with:
                - anomaly_scores: (batch, time) tensor
                - flagged_frames: (batch, time) boolean tensor
                - segments: List of anomaly segments
        """
        # Compute anomaly scores
        anomaly_scores = self.compute_anomaly_scores(features)
        
        # Determine threshold
        if threshold is None:
            # Adaptive threshold: use percentile
            threshold_percentile = self.config.get('inference', {}).get('threshold_percentile', 95)
            threshold = torch.quantile(anomaly_scores, threshold_percentile / 100.0)
        
        # Flag anomalies
        flagged_frames = anomaly_scores > threshold
        
        # Convert to segments
        segments = self._frames_to_segments(flagged_frames, anomaly_scores)
        
        return {
            'anomaly_scores': anomaly_scores,
            'flagged_frames': flagged_frames,
            'threshold': threshold,
            'segments': segments
        }
    
    def _frames_to_segments(self, flagged_frames, anomaly_scores):
        """
        Convert flagged frames to temporal segments.
        Input:
            flagged_frames - boolean tensor of flagged frames (batch, num_frames)
            anomaly_scores - per-frame anomaly scores (batch, num_frames)

        Output:
            Per-sample lists of (start_frame, end_frame, mean_score) segments
        """
        batch_size, num_frames = flagged_frames.shape
        segments_list = []
        
        for b in range(batch_size):
            segments = []
            in_segment = False
            start_frame = None
            
            for t in range(num_frames):
                if flagged_frames[b, t] and not in_segment:
                    start_frame = t
                    in_segment = True
                elif not flagged_frames[b, t] and in_segment:
                    end_frame = t - 1
                    confidence = anomaly_scores[b, start_frame:end_frame+1].mean().item()
                    segments.append((start_frame, end_frame, confidence))
                    in_segment = False
            
            # Handle segment at end
            if in_segment:
                end_frame = num_frames - 1
                confidence = anomaly_scores[b, start_frame:end_frame+1].mean().item()
                segments.append((start_frame, end_frame, confidence))
            
            segments_list.append(segments)
        
        return segments_list

