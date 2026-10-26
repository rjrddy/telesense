"""Audio emotion detection using Wav2Vec2 with sliding window processing."""

import os
import torch
import numpy as np
import librosa
from typing import List, Dict, Optional
from scipy import signal

try:
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers")


class AudioEmotionDetector:
    """
    Detect emotions from audio using sliding window approach.
    """
    
    def __init__(self, model_name: str = "superb/wav2vec2-base-superb-er", device: Optional[str] = None):
        """
        Initialize audio emotion detector.
        Input:
            model_name - Hugging Face model name for emotion recognition
            device - Device to run on ('cpu', 'cuda', 'mps')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Validate model name
        if not model_name or not isinstance(model_name, str):
            raise ValueError(f"model_name must be a non-empty string, got: {model_name}")
        
        self.processor = None
        self.model = None
        self.emotion_labels = []
        
        # Load model
        print(f"Loading audio emotion model: {model_name} on {self.device}")
        try:
            # Load feature extractor first (emotion checkpoints ship no tokenizer, so
            # Wav2Vec2Processor can't load them)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            if self.processor is None:
                raise ValueError("Processor loaded but is None")
            
            # Load model
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            if self.model is None:
                raise ValueError("Model loaded but is None")
            
            # Move to device (handle MPS issues)
            try:
                if self.device == "mps":
                    # MPS might have issues, try CPU as fallback
                    try:
                        self.model.to(self.device)
                    except Exception as mps_error:
                        print(f"Warning: MPS device failed, using CPU: {mps_error}")
                        self.device = "cpu"
                        self.model.to(self.device)
                else:
                    self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"Warning: Could not move model to {self.device}, using CPU: {e}")
                self.device = "cpu"
                self.model.to(self.device)
                self.model.eval()
            
            # Get emotion labels from model config
            if hasattr(self.model, 'config') and self.model.config:
                if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                    self.emotion_labels = list(self.model.config.id2label.values())
                elif hasattr(self.model.config, 'label2id') and self.model.config.label2id:
                    # Reverse label2id to get id2label
                    self.emotion_labels = [k for k, v in sorted(self.model.config.label2id.items(), key=lambda x: x[1])]
            
            # Fallback to default labels if none found
            if not self.emotion_labels:
                self.emotion_labels = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
            
            # Only print success if we actually have a working model
            if self.processor is not None and self.model is not None:
                print(f"Successfully loaded audio emotion model with {len(self.emotion_labels)} emotion classes")
            else:
                raise ValueError("Model or processor is None after loading")
        except Exception as e:
            print(f"Error loading audio emotion model: {e}")
            import traceback
            traceback.print_exc()
            # Don't raise - allow graceful degradation
            self.processor = None
            self.model = None
            self.emotion_labels = []
            # Don't print success message - initialization failed
    
    def detect_emotions(self, audio_path: str, window_size: float = 3.0, 
                       hop_size: float = 0.5, sr: int = 16000) -> List[Dict]:
        """
        Detect emotions using sliding window approach.
        Input:
            audio_path - Path to audio file
            window_size - Size of sliding window in seconds
            hop_size - Hop size between windows in seconds
            sr - Sample rate for audio loading

        Output:
            List of dictionaries with:
                - start_time: Window start time
                - end_time: Window end time
                - emotions: Dict of emotion -> probability
                - dominant_emotion: Emotion with highest probability
                - confidence: Confidence score
        """
        if self.processor is None or self.model is None:
            print("Warning: Audio emotion detector not properly initialized")
            return []
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        audio, sample_rate = librosa.load(audio_path, sr=sr)
        duration = len(audio) / sample_rate
        
        # Convert window sizes to samples
        window_samples = int(window_size * sample_rate)
        hop_samples = int(hop_size * sample_rate)
        
        results = []
        start_sample = 0
        
        while start_sample + window_samples <= len(audio):
            # Extract window
            window_audio = audio[start_sample:start_sample + window_samples]
            
            # Process window
            emotion_result = self._process_window(window_audio, sample_rate)
            
            if emotion_result:
                start_time = start_sample / sample_rate
                end_time = (start_sample + window_samples) / sample_rate
                
                results.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "emotions": emotion_result["emotions"],
                    "dominant_emotion": emotion_result["dominant_emotion"],
                    "confidence": emotion_result["confidence"]
                })
            
            # Move to next window
            start_sample += hop_samples
        
        # Process final window if there's remaining audio
        if start_sample < len(audio):
            window_audio = audio[start_sample:]
            if len(window_audio) >= window_samples * 0.5:  # At least 50% of window size
                emotion_result = self._process_window(window_audio, sample_rate)
                if emotion_result:
                    start_time = start_sample / sample_rate
                    end_time = duration
                    
                    results.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "emotions": emotion_result["emotions"],
                        "dominant_emotion": emotion_result["dominant_emotion"],
                        "confidence": emotion_result["confidence"]
                    })
        
        return results
    
    def _process_window(self, audio_segment: np.ndarray, sr: int) -> Optional[Dict]:
        """
        Process a single audio window.
        Input:
            audio_segment - Audio samples for the window
            sr - Sample rate

        Output:
            Dictionary with emotion probabilities and dominant emotion
        """
        try:
            if self.processor is None or self.model is None:
                return None
            
            # Process audio - ensure audio_segment is valid
            if audio_segment is None or len(audio_segment) == 0:
                return None
            
            # Process audio
            inputs = self.processor(audio_segment, sampling_rate=sr, return_tensors="pt", padding=True)
            if inputs is None:
                return None
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Get probabilities for each emotion
            probs_np = probs[0].cpu().numpy()
            emotions = {}
            
            for i, label in enumerate(self.emotion_labels):
                if i < len(probs_np):
                    emotions[label] = float(probs_np[i])
            
            # Find dominant emotion
            dominant_idx = np.argmax(probs_np)
            dominant_emotion = self.emotion_labels[dominant_idx] if dominant_idx < len(self.emotion_labels) else "neutral"
            confidence = float(probs_np[dominant_idx])
            
            return {
                "emotions": emotions,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence
            }
        except Exception as e:
            print(f"Error processing audio window: {e}")
            return None
    
    def smooth_emotions(self, emotion_results: List[Dict], method: str = "moving_average",
                       window_size: float = 2.0, hop_size: float = 0.5) -> List[Dict]:
        """
        Apply temporal smoothing to emotion results.
        Input:
            emotion_results - List of emotion detection results
            method - Smoothing method ('moving_average' or 'ema')
            window_size - Smoothing window size in seconds
            hop_size - Original hop size (for EMA alpha calculation)

        Output:
            Smoothed emotion results
        """
        if not emotion_results:
            return []
        
        if method == "moving_average":
            return self._moving_average_smooth(emotion_results, window_size)
        elif method == "ema":
            return self._ema_smooth(emotion_results, hop_size)
        else:
            return emotion_results
    
    def _moving_average_smooth(self, results: List[Dict], window_size: float) -> List[Dict]:
        """
        Apply moving average smoothing.
        Input:
            results - list of per-window/frame emotion dictionaries
            window_size - number of neighbours averaged on each side

        Output:
            List of emotion dictionaries with smoothed scores
        """
        if len(results) <= 1:
            return results
        
        smoothed = []
        
        for i, result in enumerate(results):
            # Find results within window
            window_start = result["start_time"] - window_size / 2
            window_end = result["start_time"] + window_size / 2
            
            window_results = [
                r for r in results
                if window_start <= r["start_time"] < window_end
            ]
            
            if not window_results:
                smoothed.append(result)
                continue
            
            # Average emotion probabilities
            avg_emotions = {}
            for emotion in self.emotion_labels:
                avg_emotions[emotion] = np.mean([
                    r["emotions"].get(emotion, 0.0) for r in window_results
                ])
            
            # Find new dominant emotion
            dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0]
            confidence = avg_emotions[dominant_emotion]
            
            smoothed.append({
                "start_time": result["start_time"],
                "end_time": result["end_time"],
                "emotions": avg_emotions,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence
            })
        
        return smoothed
    
    def _ema_smooth(self, results: List[Dict], hop_size: float) -> List[Dict]:
        """
        Apply exponential moving average smoothing.
        Input:
            results - list of per-window emotion dictionaries
            hop_size - window hop in seconds (sets the smoothing factor)

        Output:
            List of emotion dictionaries with smoothed scores
        """
        if len(results) <= 1:
            return results
        
        # Calculate alpha based on hop size (smaller hop = more smoothing)
        alpha = 0.3  # Default smoothing factor
        
        smoothed = []
        prev_emotions = None
        
        for result in results:
            if prev_emotions is None:
                prev_emotions = result["emotions"].copy()
                smoothed.append(result)
                continue
            
            # EMA update
            smoothed_emotions = {}
            for emotion in self.emotion_labels:
                current_val = result["emotions"].get(emotion, 0.0)
                prev_val = prev_emotions.get(emotion, 0.0)
                smoothed_emotions[emotion] = alpha * current_val + (1 - alpha) * prev_val
            
            # Find dominant emotion
            dominant_emotion = max(smoothed_emotions.items(), key=lambda x: x[1])[0]
            confidence = smoothed_emotions[dominant_emotion]
            
            smoothed.append({
                "start_time": result["start_time"],
                "end_time": result["end_time"],
                "emotions": smoothed_emotions,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence
            })
            
            prev_emotions = smoothed_emotions
        
        return smoothed
