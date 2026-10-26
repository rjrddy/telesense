"""
Inference pipeline for multimodal behavioral model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union
import os

from ..models import MultimodalBehavioralModel
from telesense_ml.data.preprocessing import AudioPreprocessor, VideoPreprocessor, TextPreprocessor
from telesense_ml.data.collators import MultimodalCollator


class Predictor:
    """
    Predictor for multimodal behavioral analysis.
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
            checkpoint_path - Path to model checkpoint
            device - Device to run inference on
            config - Model configuration (loaded from checkpoint if not provided)
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
        self.model = MultimodalBehavioralModel(config)
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
            max_length=audio_config.get('max_length', 10),
            normalize=audio_config.get('normalize', True)
        )
        
        self.video_preprocessor = VideoPreprocessor(
            fps=video_config.get('fps', 30),
            frame_size=video_config.get('frame_size', 224),
            max_frames=video_config.get('max_frames', 300),
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
        
        # Get class names (if available in config)
        self.class_names = config.get('class_names', None)
    
    def predict(
        self,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        text: Optional[str] = None
    ) -> Dict:
        """
        Make prediction on input data.
        Input:
            audio_path - Path to audio file
            video_path - Path to video file
            text - Text string

        Output:
            Dictionary with predictions and probabilities
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
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(
                audio_waveform=audio_waveform,
                audio_mask=audio_mask,
                video_frames=video_frames,
                text_input_ids=text_input_ids,
                text_mask=text_mask
            )
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probs[0, predicted_class].item()
        
        # Format output
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probs[0].cpu().numpy().tolist()
        }
        
        if self.class_names:
            result['predicted_class_name'] = self.class_names[predicted_class]
        
        return result
    
    def predict_batch(
        self,
        samples: list[Dict[str, Union[str, None]]]
    ) -> list[Dict]:
        """
        Make predictions on a batch of samples.
        Input:
            samples - List of dictionaries with 'audio_path', 'video_path', 'text' keys

        Output:
            List of prediction dictionaries
        """
        results = []
        for sample in samples:
            result = self.predict(
                audio_path=sample.get('audio_path'),
                video_path=sample.get('video_path'),
                text=sample.get('text')
            )
            results.append(result)
        return results

