"""Video emotion detection using PyTorch-based alternatives to DeepFace."""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional
import torch

# Try FER first (PyTorch-based, lightweight)
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False


class VideoEmotionDetectorPyTorch:
    """
    Detect emotions from video frames using PyTorch-compatible libraries.
    """
    
    def __init__(self, method: str = "fer", device: Optional[str] = None):
        """
        Initialize video emotion detector.
        Input:
            method - Detection method (only 'fer' is supported)
            device - Device to run on ('cpu', 'cuda', 'mps')
        """
        self.method = method
        self.detector = None
        
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
        
        # Emotion labels (standard across methods)
        self.emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        
        # Map emotions to behavioral concerns
        self.emotion_to_concern = {
            "sad": "Depression",
            "angry": "Agitation",
            "fear": "Anxiety",
            "disgust": "Negative_Emotion",
            "happy": "Positive_Emotion",
            "surprise": "Neutral",
            "neutral": "Neutral"
        }
        
        # Initialize detector based on method
        if method == "fer" and FER_AVAILABLE:
            try:
                # FER uses MTCNN by default, but can fall back to OpenCV
                # Try MTCNN first, fall back to OpenCV if it fails
                try:
                    self.detector = FER(mtcnn=True)  # Use MTCNN for better face detection
                    print(f"Initialized FER (Facial Expression Recognition) detector with MTCNN on {self.device}")
                except Exception as e:
                    print(f"Warning: MTCNN initialization failed, trying OpenCV backend: {e}")
                    self.detector = FER(mtcnn=False)  # Fall back to OpenCV
                    print(f"Initialized FER with OpenCV backend on {self.device}")
            except Exception as e:
                print(f"Warning: Could not initialize FER: {e}")
                print("Note: Install with 'pip install fer --no-build-isolation' in conda environments")
                self.detector = None
        # Only FER classifies emotions; without it video emotion is disabled rather than
        # filled with placeholder values
        if self.detector is None:
            print("Warning: No video emotion model available (install 'fer'). Video emotion will be disabled.")
    
    def detect_emotions(self, video_path: str, sample_rate: float = 1.0,
                       fps: Optional[float] = None) -> List[Dict]:
        """
        Detect emotions from video frames.
        Input:
            video_path - Path to video file
            sample_rate - Frames per second to analyze (1.0 = 1 frame per second)
            fps - Video FPS (if None, will be detected from video)

        Output:
            List of dictionaries with:
                - frame_idx: Frame index
                - timestamp: Time in seconds
                - emotions: Dict of emotion -> probability
                - dominant_emotion: Emotion with highest probability
                - confidence: Confidence score
                - faces_detected: Number of faces detected
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if self.detector is None:
            return []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        video_fps = fps if fps is not None else cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / video_fps if video_fps > 0 else 0
        
        # Calculate frame interval for sampling
        frame_interval = max(1, int(video_fps / sample_rate)) if video_fps > 0 else 1
        
        results = []
        frame_idx = 0
        
        print(f"Processing video: {frame_count} frames at {video_fps} fps, sampling every {frame_interval} frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / video_fps if video_fps > 0 else 0
                
                # Analyze frame
                emotion_result = self._analyze_frame(frame)
                
                if emotion_result:
                    results.append({
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                        **emotion_result
                    })
            
            frame_idx += 1
        
        cap.release()
        
        print(f"Processed {len(results)} frames from video")
        return results
    
    def _analyze_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Analyze a single frame for emotions.
        Input:
            frame - Video frame (BGR format from OpenCV)

        Output:
            Dictionary with emotion analysis results, or None if no face or emotion model
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.method == "fer" and self.detector is not None:
                return self._analyze_with_fer(frame_rgb)
            return None
        except Exception as e:
            # Skip frames that fail to analyze rather than recording a placeholder
            return None
    
    def _analyze_with_fer(self, frame_rgb: np.ndarray) -> Optional[Dict]:
        """
        Analyze frame using FER library.
        Input:
            frame_rgb - RGB frame as a numpy array (H, W, 3)

        Output:
            Dictionary with emotion scores, dominant emotion, confidence and number of faces
            detected, or None if no face is found
        """
        try:
            # FER returns top emotion with bounding box
            result = self.detector.detect_emotions(frame_rgb)
            
            if not result or len(result) == 0:
                return None
            
            # Get emotions from first face (most prominent)
            emotions_dict = result[0]['emotions']
            faces_detected = len(result)
            
            # Normalize emotions to 0-1 range (FER returns 0-1 already)
            emotions = {}
            for emotion, score in emotions_dict.items():
                emotions[emotion.lower()] = float(score)
            
            # Find dominant emotion
            if emotions:
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                confidence = emotions[dominant_emotion]
            else:
                dominant_emotion = "neutral"
                confidence = 0.0
            
            return {
                "emotions": emotions,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "faces_detected": faces_detected
            }
        except Exception as e:
            print(f"Error in FER analysis: {e}")
            return None
    
    def smooth_emotions(self, emotion_results: List[Dict], method: str = "moving_average",
                       window_size: float = 2.0) -> List[Dict]:
        """
        Apply temporal smoothing to emotion results.
        Input:
            emotion_results - list of per-frame emotion dictionaries
            method - 'moving_average' or 'ema'
            window_size - window size for the moving average

        Output:
            List of emotion dictionaries with smoothed scores
        """
        if not emotion_results or len(emotion_results) <= 1:
            return emotion_results
        
        if method == "moving_average":
            return self._moving_average_smooth(emotion_results, window_size)
        elif method == "ema":
            return self._ema_smooth(emotion_results)
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
        smoothed = []
        
        for i, result in enumerate(results):
            # Find results within window
            window_start = result["timestamp"] - window_size / 2
            window_end = result["timestamp"] + window_size / 2
            
            window_results = [
                r for r in results
                if window_start <= r["timestamp"] < window_end
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
                "frame_idx": result["frame_idx"],
                "timestamp": result["timestamp"],
                "emotions": avg_emotions,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "faces_detected": result.get("faces_detected", 0)
            })
        
        return smoothed
    
    def _ema_smooth(self, results: List[Dict], alpha: float = 0.3) -> List[Dict]:
        """
        Apply exponential moving average smoothing.
        Input:
            results - list of per-frame emotion dictionaries
            alpha - smoothing factor in (0, 1]; higher follows the signal more closely

        Output:
            List of emotion dictionaries with smoothed scores
        """
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
                "frame_idx": result["frame_idx"],
                "timestamp": result["timestamp"],
                "emotions": smoothed_emotions,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "faces_detected": result.get("faces_detected", 0)
            })
            
            prev_emotions = smoothed_emotions
        
        return smoothed
