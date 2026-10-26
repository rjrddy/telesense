"""
Temporal localization model for flagging behavioral concerns at specific timepoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from telesense_ml.encoders import AudioEncoder, VisualEncoder, TextEncoder
from .temporal_encoders import MultiScaleTemporalEncoder
from .temporal_localization import TemporalLocalizationModel


class TemporalMultimodalBehavioralModel(nn.Module):
    """
    Temporal localization model for detecting behavioral concerns at specific timepoints.

    Architecture:
    1. Modality-specific encoders (audio, visual, text)
    2. Multi-scale temporal encoder (TCN + Transformer)
    3. Temporal localization head (dense frame-level prediction)
    """
    
    def __init__(self, config):
        """
        Initialize the TemporalMultimodalBehavioralModel.
        Input:
            config - configuration dictionary (model section configures encoders and temporal head)
        """
        super().__init__()
        
        # Modality encoders
        self.visual_encoder = VisualEncoder(
            encoder=config['model']['visual']['encoder'],
            pretrained=config['model']['visual']['pretrained'],
            hidden_size=config['model']['visual']['hidden_size'],
            num_temporal_layers=0,  # We'll use multi-scale encoder instead
            dropout=config['model']['visual']['dropout']
        )
        
        self.audio_encoder = AudioEncoder(
            model_name=config['model']['audio']['model_name'],
            hidden_size=config['model']['audio']['hidden_size'],
            num_temporal_layers=0,  # We'll use multi-scale encoder instead
            dropout=config['model']['audio']['dropout']
        )
        
        self.text_encoder = TextEncoder(
            model_name=config['model']['text']['model_name'],
            hidden_size=config['model']['text']['hidden_size'],
            max_length=config['model']['text']['max_length'],
            dropout=config['model']['text']['dropout']
        )
        
        # Project all modalities to common dimension
        fusion_dim = config['model']['temporal']['hidden_dim']
        
        self.visual_proj = nn.Linear(
            self.visual_encoder.get_output_dim(),
            fusion_dim
        )
        self.audio_proj = nn.Linear(
            self.audio_encoder.get_output_dim(),
            fusion_dim
        )
        self.text_proj = nn.Linear(
            self.text_encoder.get_output_dim(),
            fusion_dim
        )
        
        # Multi-scale temporal encoder
        self.temporal_encoder = MultiScaleTemporalEncoder(
            input_dim=fusion_dim * 3,  # Combined from all modalities
            hidden_dim=fusion_dim,
            dropout=config['model']['temporal'].get('dropout', 0.1)
        )
        
        # Temporal localization head
        self.localization_model = TemporalLocalizationModel(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim,
            num_concern_types=config['model']['temporal']['num_concern_types'],
            use_attention=config['model']['temporal'].get('use_attention', True),
            use_smoothing=config['model']['temporal'].get('use_smoothing', True),
            smoothing_window=config['model']['temporal'].get('smoothing_window', 5),
            dropout=config['model']['temporal'].get('dropout', 0.1)
        )
        
    def forward(
        self,
        audio_waveform=None,
        audio_mask=None,
        video_frames=None,
        text_input_ids=None,
        text_mask=None
    ):
        """
        Forward pass for temporal localization.
        Input:
            audio_waveform - Audio waveform tensor (batch, seq_len)
            audio_mask - Audio attention mask (batch, seq_len)
            video_frames - Video frames tensor (batch, num_frames, C, H, W)
            text_input_ids - Text token IDs (batch, seq_len)
            text_mask - Text attention mask (batch, seq_len)

        Output:
            Frame-level concern scores of shape (batch, time, num_concern_types)
        """
        batch_size = None
        temporal_features_list = []
        
        # Encode visual modality (frame-level)
        if video_frames is not None:
            batch_size = video_frames.shape[0]
            num_frames = video_frames.shape[1]
            
            # Process frames: (batch, num_frames, C, H, W)
            # Visual encoder expects (batch, seq_len, C, H, W) format
            visual_features = self.visual_encoder(video_frames, return_sequence=True)
            # visual_features is now (batch, num_frames, hidden_size)
            visual_features = self.visual_proj(visual_features)  # (batch, num_frames, fusion_dim)
            temporal_features_list.append(visual_features)
        
        # Encode audio modality (temporal)
        if audio_waveform is not None:
            if batch_size is None:
                batch_size = audio_waveform.shape[0]
            
            # Get temporal audio features
            audio_features = self.audio_encoder(audio_waveform, audio_mask)
            # Downsample/interpolate to match video frames if needed
            if video_frames is not None:
                num_frames = video_frames.shape[1]
                audio_features = F.interpolate(
                    audio_features.transpose(1, 2),
                    size=num_frames,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            audio_features = self.audio_proj(audio_features)
            temporal_features_list.append(audio_features)
        
        # Encode text modality (expand to temporal dimension)
        if text_input_ids is not None:
            if batch_size is None:
                batch_size = text_input_ids.shape[0]
            
            text_features = self.text_encoder(text_input_ids, text_mask)
            # Get sequence-level representation (mean pooling)
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
            
            text_pooled = text_pooled.unsqueeze(1)  # (batch, 1, hidden_dim)
            text_features = text_pooled.expand(-1, num_frames, -1)
            text_features = self.text_proj(text_features)
            temporal_features_list.append(text_features)
        
        # Handle missing modalities
        if len(temporal_features_list) == 0:
            raise ValueError("At least one modality must be provided")
        
        # Concatenate all modalities
        combined_features = torch.cat(temporal_features_list, dim=-1)  # (batch, time, fusion_dim * 3)
        
        # Multi-scale temporal encoding
        temporal_encoded = self.temporal_encoder(combined_features)  # (batch, time, fusion_dim)
        
        # Temporal localization (dense prediction)
        frame_scores = self.localization_model(temporal_encoded)  # (batch, time, num_concern_types)
        
        return frame_scores
    
    def predict_segments(self, frame_scores, threshold=0.5, min_duration=1):
        """
        Convert frame-level scores to temporal segments.
        Input:
            frame_scores - Frame-level scores (batch, time, num_concern_types)
            threshold - Threshold for concern detection
            min_duration - Minimum segment duration in frames

        Output:
            List of segments per sample: [(start_frame, end_frame, concern_type, confidence), ...]
        """
        batch_size, num_frames, num_types = frame_scores.shape
        segments_list = []
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(frame_scores)
        
        for b in range(batch_size):
            segments = []
            
            for concern_type in range(num_types):
                # Get binary mask for this concern type
                mask = (probs[b, :, concern_type] > threshold).cpu().numpy()
                
                # Find contiguous segments
                in_segment = False
                start_frame = None
                
                for t in range(num_frames):
                    if mask[t] and not in_segment:
                        # Start of new segment
                        start_frame = t
                        in_segment = True
                    elif not mask[t] and in_segment:
                        # End of segment
                        end_frame = t - 1
                        if end_frame - start_frame >= min_duration:
                            confidence = probs[b, start_frame:end_frame+1, concern_type].mean().item()
                            segments.append((start_frame, end_frame, concern_type, confidence))
                        in_segment = False
                
                # Handle segment at end
                if in_segment:
                    end_frame = num_frames - 1
                    if end_frame - start_frame >= min_duration:
                        confidence = probs[b, start_frame:end_frame+1, concern_type].mean().item()
                        segments.append((start_frame, end_frame, concern_type, confidence))
            
            segments_list.append(segments)
        
        return segments_list

