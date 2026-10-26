"""Video emotion detection using DeepFace or PyTorch alternatives."""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

# Try DeepFace first
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Try PyTorch-based alternatives
try:
    from .video_emotion_detector_pytorch import VideoEmotionDetectorPyTorch
    PYTORCH_DETECTOR_AVAILABLE = True
except ImportError:
    PYTORCH_DETECTOR_AVAILABLE = False
    VideoEmotionDetectorPyTorch = None


class VideoEmotionDetector:
    """
    Detect emotions from video frames using DeepFace.
    """
    
    def __init__(self, model_name: str = "VGG-Face", backend: str = "opencv", 
                 use_pytorch_alternative: bool = True):
        """
        Initialize video emotion detector.
        Input:
            model_name - DeepFace model name (VGG-Face, Facenet, etc.) - only used if DeepFace available
            backend - Backend for face detection ('opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface')
            use_pytorch_alternative - If True, use PyTorch-based alternative when DeepFace unavailable
        """
        self.model_name = None
        self.backend = None
        self.pytorch_detector = None
        self.use_deepface = False
        
        if DEEPFACE_AVAILABLE:
            # Use DeepFace if available
            try:
                self.model_name = model_name
                self.backend = backend
                self.use_deepface = True
                print(f"Initialized DeepFace video emotion detector (model: {model_name}, backend: {backend})")
            except Exception as e:
                print(f"Warning: DeepFace initialization failed: {e}")
                self.use_deepface = False
        elif use_pytorch_alternative and PYTORCH_DETECTOR_AVAILABLE and VideoEmotionDetectorPyTorch is not None:
            # Use PyTorch-based alternative
            try:
                self.pytorch_detector = VideoEmotionDetectorPyTorch(method="fer")
                if self.pytorch_detector.detector is None:
                    self.pytorch_detector = None  # FER not installed: no emotion model
                else:
                    print("Initialized PyTorch-based video emotion detector (FER)")
            except Exception as e:
                print(f"Warning: PyTorch-based detector initialization failed: {e}")
                self.pytorch_detector = None
        else:
            print("Warning: No video emotion detector available.")
            print("Install 'fer' for PyTorch-based emotion detection: pip install fer")
            print("Or install 'deepface' for TensorFlow-based detection (may conflict with PyTorch)")
        
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
        # Use appropriate detector
        if self.use_deepface and self.model_name:
            # Use DeepFace
            return self._detect_with_deepface(video_path, sample_rate, fps)
        elif self.pytorch_detector:
            # Use PyTorch-based alternative
            return self.pytorch_detector.detect_emotions(video_path, sample_rate, fps)
        else:
            # No detector available
            return []
    
    def _detect_with_deepface(self, video_path: str, sample_rate: float = 1.0,
                              fps: Optional[float] = None) -> List[Dict]:
        """
        Detect emotions using DeepFace.
        Input:
            video_path - path to the video file
            sample_rate - analyze every Nth frame
            fps - video frame rate

        Output:
            List of per-frame emotion dictionaries
        """
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
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
            Dictionary with emotion analysis results, or None if no face is detected
        """
        try:
            # Convert BGR to RGB for DeepFace
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Analyze with DeepFace
            # DeepFace can handle multiple faces, but we'll get the dominant one
            analysis = DeepFace.analyze(
                img_path=frame_rgb,
                actions=['emotion'],
                model_name=self.model_name,
                detector_backend=self.backend,
                enforce_detection=False,  # Don't fail if no face detected
                silent=True
            )
            
            # Handle multiple faces (analysis can be a list)
            if isinstance(analysis, list):
                # Use the first face (largest/most prominent)
                analysis = analysis[0]
            
            # Extract emotions
            emotions = {}
            if 'emotion' in analysis:
                for emotion, score in analysis['emotion'].items():
                    emotions[emotion.lower()] = float(score) / 100.0  # DeepFace returns 0-100
            
            # Find dominant emotion
            if emotions:
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                confidence = emotions[dominant_emotion]
            else:
                dominant_emotion = "neutral"
                confidence = 0.0
            
            # Count faces (if available)
            faces_detected = 1 if emotions else 0
            
            return {
                "emotions": emotions,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "faces_detected": faces_detected
            }
        except Exception as e:
            # DeepFace fails when no face is detected; skip the frame rather than
            # recording a placeholder
            return None
    
    def smooth_emotions(self, emotion_results: List[Dict], method: str = "moving_average",
                       window_size: float = 2.0) -> List[Dict]:
        """
        Apply temporal smoothing to emotion results.
        Input:
            emotion_results - List of emotion detection results
            method - Smoothing method ('moving_average' or 'ema')
            window_size - Smoothing window size in seconds

        Output:
            Smoothed emotion results
        """
        # Use PyTorch detector's smoothing if available, otherwise use DeepFace smoothing
        if self.pytorch_detector:
            return self.pytorch_detector.smooth_emotions(emotion_results, method, window_size)
        
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
