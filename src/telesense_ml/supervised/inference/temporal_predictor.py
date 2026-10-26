"""
Inference pipeline for temporal localization model.
"""

import torch
from typing import Dict, Optional, List, Tuple
import os

from ..models import TemporalMultimodalBehavioralModel
from telesense_ml.data.preprocessing import AudioPreprocessor, VideoPreprocessor, TextPreprocessor


class TemporalPredictor:
    """
    Predictor for temporal localization of behavioral concerns.
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
            config - Model configuration
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
        self.model = TemporalMultimodalBehavioralModel(config)
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
        
        # Inference settings
        inference_config = config.get('inference', {})
        self.threshold = inference_config.get('threshold', 0.5)
        self.min_duration = inference_config.get('min_duration', 30)
        
        # Get class names (if available)
        self.class_names = config.get('class_names', None)
    
    def predict(
        self,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        text: Optional[str] = None
    ) -> Dict:
        """
        Make temporal localization prediction.
        Input:
            audio_path - Path to audio file
            video_path - Path to video file
            text - Text string

        Output:
            Dictionary with:
                - frame_scores: (num_frames, num_concern_types) array
                - segments: List of [(start_frame, end_frame, concern_type, confidence), ...]
                - frame_predictions: Per-frame concern predictions
                - seconds_per_frame: Seconds of media covered by each output frame
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
            frame_scores = self.model(
                audio_waveform=audio_waveform,
                audio_mask=audio_mask,
                video_frames=video_frames,
                text_input_ids=text_input_ids,
                text_mask=text_mask
            )
            
            # Get probabilities
            frame_probs = torch.sigmoid(frame_scores)
        
        # Convert to segments
        segments = self.model.predict_segments(
            frame_scores,
            threshold=self.threshold,
            min_duration=self.min_duration
        )
        
        # Inputs are padded to a fixed window (max_frames of video, or max_length seconds of
        # audio), so each output frame covers window / num_frames seconds
        num_frames = frame_scores.shape[1]
        if video_frames is not None:
            window_s = self.video_preprocessor.max_frames * self.video_preprocessor.frame_interval_s
        else:
            window_s = self.audio_preprocessor.max_length
        
        # Format output
        result = {
            'frame_scores': frame_scores[0].cpu().numpy(),
            'frame_probs': frame_probs[0].cpu().numpy(),
            'segments': segments[0] if segments else [],
            'num_frames': num_frames,
            'seconds_per_frame': window_s / num_frames if num_frames else None
        }
        
        # Add frame-level predictions
        frame_predictions = []
        for t in range(frame_scores.shape[1]):
            frame_pred = {
                'frame': t,
                'concerns': []
            }
            for concern_type in range(frame_scores.shape[2]):
                prob = frame_probs[0, t, concern_type].item()
                if prob > self.threshold:
                    concern_name = self.class_names[concern_type] if self.class_names else f"Concern_{concern_type}"
                    frame_pred['concerns'].append({
                        'type': concern_type,
                        'name': concern_name,
                        'confidence': prob
                    })
            frame_predictions.append(frame_pred)
        
        result['frame_predictions'] = frame_predictions
        
        return result
    
    def predict_segments_only(
        self,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
        text: Optional[str] = None
    ) -> List[Tuple[int, int, int, float]]:
        """
        Get only the temporal segments (simplified output).
        Input:
            audio_path - path to the audio file (optional)
            video_path - path to the video file (optional)
            text - transcript text (optional)

        Output:
            List of (start_frame, end_frame, concern_type, confidence) tuples
        """
        result = self.predict(audio_path, video_path, text)
        return result['segments']

