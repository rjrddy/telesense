"""
Inference pipeline for unsupervised temporal localization.
"""

import torch
from typing import Dict, Optional
import os

from ..models import UnsupervisedTemporalModel
from telesense_ml.data.preprocessing import AudioPreprocessor, VideoPreprocessor, TextPreprocessor


class UnsupervisedPredictor:
    """
    Predictor for unsupervised temporal anomaly detection.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize predictor.
        Input:
            checkpoint_path - path to the trained model checkpoint
            device - torch device for inference (auto-selected when None)
            config - configuration dictionary (read from the checkpoint when None)
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get config
        if config is None:
            config = checkpoint.get('config', {})
        self.config = config
        
        # Device setup
        if device is None:
            device_type = config.get('device', {}).get('type', 'mps')
            if device_type == 'mps' and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif device_type == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # Initialize model
        self.model = UnsupervisedTemporalModel(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessors
        data_config = config.get('data', {})
        audio_config = data_config.get('audio', {})
        video_config = data_config.get('video', {})
        text_config = data_config.get('text', {})
        
        self.audio_preprocessor = AudioPreprocessor(
            sample_rate=audio_config.get('sample_rate', 16000),
            max_length=audio_config.get('max_length', 60),
            normalize=audio_config.get('normalize', True)
        )
        
        self.video_preprocessor = VideoPreprocessor(
            fps=video_config.get('fps', 30),
            frame_size=video_config.get('frame_size', 224),
            max_frames=video_config.get('max_frames', 1800),
            face_detection=video_config.get('face_detection', True)
        )
        
        self.text_preprocessor = TextPreprocessor(
            max_length=text_config.get('max_length', 512),
            lowercase=text_config.get('lowercase', True)
        )
        
        # Text tokenizer
        from transformers import DistilBertTokenizer
        self.text_tokenizer = DistilBertTokenizer.from_pretrained(
            config.get('model', {}).get('text', {}).get('model_name', 'distilbert-base-uncased')
        )
    
    def predict(
        self,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        text: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        Make unsupervised anomaly detection prediction.
        Input:
            audio_path - Path to audio file
            video_path - Path to video file
            text - Text string
            threshold - Anomaly threshold (if None, use adaptive)

        Output:
            Dictionary with anomaly scores, flagged frames, and segments
        """
        # Preprocess inputs
        audio_waveform = None
        audio_mask = None
        video_frames = None
        text_input_ids = None
        text_mask = None
        
        if audio_path is not None and os.path.exists(audio_path):
            audio_waveform = self.audio_preprocessor(audio_path)
            audio_mask = self.audio_preprocessor.create_attention_mask(audio_waveform)
            audio_waveform = audio_waveform.unsqueeze(0).to(self.device)
            audio_mask = audio_mask.unsqueeze(0).to(self.device)
        
        if video_path is not None and os.path.exists(video_path):
            video_frames = self.video_preprocessor(video_path)
            video_frames = video_frames.unsqueeze(0).to(self.device)
        
        if text is not None:
            text = self.text_preprocessor(text)
            tokenized = self.text_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.text_preprocessor.max_length,
                return_tensors="pt"
            )
            text_input_ids = tokenized['input_ids'].to(self.device)
            text_mask = tokenized['attention_mask'].to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.feature_extractor(
                video_frames=video_frames,
                audio_waveform=audio_waveform,
                audio_mask=audio_mask,
                text_input_ids=text_input_ids,
                text_mask=text_mask
            )
            
            # Predict anomalies
            results = self.model.predict_anomalies(features, threshold=threshold)
        
        # Format output
        return {
            'anomaly_scores': results['anomaly_scores'][0].cpu().numpy(),
            'flagged_frames': results['flagged_frames'][0].cpu().numpy(),
            'threshold': results['threshold'].item() if isinstance(results['threshold'], torch.Tensor) else results['threshold'],
            'segments': results['segments'][0] if results['segments'] else [],
            'num_frames': features.shape[1]
        }

