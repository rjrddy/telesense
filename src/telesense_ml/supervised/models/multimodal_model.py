"""
Complete Multimodal Behavioral Model.
"""

import torch
import torch.nn as nn
from telesense_ml.encoders import AudioEncoder, VisualEncoder, TextEncoder
from .fusion import CrossModalAttention


class ClassificationHead(nn.Module):
    """
    Classification head for emotion/behavioral categories.
    """
    
    def __init__(self, input_size, num_classes, hidden_dims=[512, 256], dropout=0.2):
        """
        Initialize the ClassificationHead.
        Input:
            input_size - fused feature dimension
            num_classes - number of output classes
            hidden_dims - sizes of the hidden MLP layers
            dropout - dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass.
        Input:
            x - Input features of shape (batch, input_size)

        Output:
            Logits of shape (batch, num_classes)
        """
        return self.classifier(x)


class MultimodalBehavioralModel(nn.Module):
    """
    Complete multimodal behavioral analysis model.
    """
    
    def __init__(self, config):
        """
        Initialize the MultimodalBehavioralModel.
        Input:
            config - configuration dictionary (model section configures encoders, fusion and head)
        """
        super().__init__()
        
        # Initialize encoders
        self.audio_encoder = AudioEncoder(
            model_name=config['model']['audio']['model_name'],
            hidden_size=config['model']['audio']['hidden_size'],
            num_temporal_layers=config['model']['audio']['num_temporal_layers'],
            dropout=config['model']['audio']['dropout']
        )
        
        self.visual_encoder = VisualEncoder(
            encoder=config['model']['visual']['encoder'],
            pretrained=config['model']['visual']['pretrained'],
            hidden_size=config['model']['visual']['hidden_size'],
            num_temporal_layers=config['model']['visual']['num_temporal_layers'],
            dropout=config['model']['visual']['dropout']
        )
        
        self.text_encoder = TextEncoder(
            model_name=config['model']['text']['model_name'],
            hidden_size=config['model']['text']['hidden_size'],
            max_length=config['model']['text']['max_length'],
            dropout=config['model']['text']['dropout']
        )
        
        # Ensure all encoders output same hidden size for fusion
        fusion_hidden_size = config['model']['fusion']['hidden_size']
        
        # Projection layers if needed
        if self.audio_encoder.get_output_dim() != fusion_hidden_size:
            self.audio_proj = nn.Linear(
                self.audio_encoder.get_output_dim(),
                fusion_hidden_size
            )
        else:
            self.audio_proj = nn.Identity()
            
        if self.visual_encoder.get_output_dim() != fusion_hidden_size:
            self.visual_proj = nn.Linear(
                self.visual_encoder.get_output_dim(),
                fusion_hidden_size
            )
        else:
            self.visual_proj = nn.Identity()
            
        if self.text_encoder.get_output_dim() != fusion_hidden_size:
            self.text_proj = nn.Linear(
                self.text_encoder.get_output_dim(),
                fusion_hidden_size
            )
        else:
            self.text_proj = nn.Identity()
        
        # Cross-modal fusion
        self.fusion = CrossModalAttention(
            hidden_size=fusion_hidden_size,
            num_heads=config['model']['fusion']['num_heads'],
            num_layers=config['model']['fusion']['num_layers'],
            dropout=config['model']['fusion']['dropout']
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            input_size=fusion_hidden_size,
            num_classes=config['model']['classifier']['num_classes'],
            hidden_dims=config['model']['classifier']['hidden_dims'],
            dropout=config['model']['classifier']['dropout']
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
        Forward pass through the complete model.
        Input:
            audio_waveform - Audio waveform tensor (batch, seq_len)
            audio_mask - Audio attention mask (batch, seq_len)
            video_frames - Video frames tensor (batch, seq_len, C, H, W)
            text_input_ids - Text token IDs (batch, seq_len)
            text_mask - Text attention mask (batch, seq_len)

        Output:
            Logits of shape (batch, num_classes)
        """
        # Encode each modality
        audio_features = None
        visual_features = None
        text_features = None
        
        if audio_waveform is not None:
            audio_features = self.audio_encoder(audio_waveform, audio_mask)
            audio_features = self.audio_proj(audio_features)
        
        if video_frames is not None:
            visual_features = self.visual_encoder(video_frames, return_sequence=True)
            visual_features = self.visual_proj(visual_features)
        
        if text_input_ids is not None:
            text_features = self.text_encoder(text_input_ids, text_mask)
            text_features = self.text_proj(text_features)
        
        # Handle missing modalities (for flexibility during training)
        if audio_features is None:
            batch_size = visual_features.shape[0] if visual_features is not None else text_features.shape[0]
            hidden_size = visual_features.shape[-1] if visual_features is not None else text_features.shape[-1]
            audio_features = torch.zeros(batch_size, 1, hidden_size, device=next(self.parameters()).device)
        
        if visual_features is None:
            batch_size = audio_features.shape[0] if audio_features is not None else text_features.shape[0]
            hidden_size = audio_features.shape[-1] if audio_features is not None else text_features.shape[-1]
            visual_features = torch.zeros(batch_size, 1, hidden_size, device=next(self.parameters()).device)
        
        if text_features is None:
            batch_size = audio_features.shape[0] if audio_features is not None else visual_features.shape[0]
            hidden_size = audio_features.shape[-1] if audio_features is not None else visual_features.shape[-1]
            text_features = torch.zeros(batch_size, 1, hidden_size, device=next(self.parameters()).device)
        
        # Fuse modalities
        fused_features = self.fusion(audio_features, visual_features, text_features)
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits

