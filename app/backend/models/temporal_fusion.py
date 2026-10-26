"""Temporal fusion and smoothing for multimodal emotion scores."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import signal
from scipy.ndimage import uniform_filter1d


class TemporalFusion:
    """
    Fuse and smooth multimodal emotion scores over time.
    """
    
    def __init__(self, audio_weight: float = 0.4, text_weight: float = 0.4, 
                 video_weight: float = 0.2):
        """
        Initialize temporal fusion.
        Input:
            audio_weight - Weight for audio emotion scores
            text_weight - Weight for text emotion scores
            video_weight - Weight for video emotion scores
        """
        # Normalize weights
        total_weight = audio_weight + text_weight + video_weight
        self.audio_weight = audio_weight / total_weight
        self.text_weight = text_weight / total_weight
        self.video_weight = video_weight / total_weight
    
    def fuse_modalities(self, audio_scores: List[Dict], text_scores: List[Dict],
                       video_scores: List[Dict], duration: float,
                       time_resolution: float = 0.1) -> np.ndarray:
        """
        Fuse scores from all modalities into a single concern score over time.
        Input:
            audio_scores - List of audio emotion results with timestamps
            text_scores - List of text emotion results with timestamps
            video_scores - List of video emotion results with timestamps
            duration - Total duration in seconds
            time_resolution - Time resolution for output (seconds per sample)

        Output:
            Array of concern scores over time (shape: [num_samples])
        """
        # Create time grid
        num_samples = int(duration / time_resolution) + 1
        time_grid = np.linspace(0, duration, num_samples)
        fused_scores = np.zeros(num_samples)
        
        # Convert emotions to concern scores first (before time series conversion)
        audio_concern_scores = self._emotions_to_concern_score(audio_scores)
        text_concern_scores = self._emotions_to_concern_score(text_scores)
        video_concern_scores = self._emotions_to_concern_score(video_scores)
        
        # Convert concern scores to time series
        audio_concern = self._concern_scores_to_time_series(audio_concern_scores, audio_scores, time_grid, "start_time")
        text_concern = self._concern_scores_to_time_series(text_concern_scores, text_scores, time_grid, "start_time")
        video_concern = self._concern_scores_to_time_series(video_concern_scores, video_scores, time_grid, "timestamp")
        
        # Weighted fusion
        fused_scores = (
            self.audio_weight * audio_concern +
            self.text_weight * text_concern +
            self.video_weight * video_concern
        )
        
        return fused_scores
    
    def _scores_to_time_series(self, scores: List[Dict], time_grid: np.ndarray,
                               time_key: str) -> np.ndarray:
        """
        Convert score results to time series.
        Input:
            scores - List of score results with timestamps
            time_grid - Time grid to interpolate onto
            time_key - Key in dict that contains timestamp

        Output:
            Array of confidence scores over time
        """
        if not scores:
            return np.zeros(len(time_grid))
        
        # Extract timestamps and confidences
        times = np.array([s.get(time_key, 0.0) for s in scores])
        confidences = np.array([s.get("confidence", 0.0) for s in scores])
        
        # Interpolate to time grid
        if len(times) > 1:
            # Use linear interpolation
            series = np.interp(time_grid, times, confidences, left=0.0, right=0.0)
        else:
            # Single value - spread it
            series = np.zeros(len(time_grid))
            if len(times) == 1:
                idx = np.argmin(np.abs(time_grid - times[0]))
                series[idx] = confidences[0]
        
        return series
    
    def _concern_scores_to_time_series(self, concern_scores: np.ndarray, 
                                       original_scores: List[Dict], 
                                       time_grid: np.ndarray,
                                       time_key: str) -> np.ndarray:
        """
        Convert concern scores to time series.
        Input:
            concern_scores - Array of concern scores (one per original score)
            original_scores - List of original score results with timestamps
            time_grid - Time grid to interpolate onto
            time_key - Key in dict that contains timestamp

        Output:
            Array of concern scores over time
        """
        if len(concern_scores) == 0 or not original_scores:
            return np.zeros(len(time_grid))
        
        # Extract timestamps
        times = np.array([s.get(time_key, 0.0) for s in original_scores])
        
        # Interpolate to time grid
        if len(times) > 1:
            # Use linear interpolation
            series = np.interp(time_grid, times, concern_scores, left=0.0, right=0.0)
        else:
            # Single value - spread it
            series = np.zeros(len(time_grid))
            if len(times) == 1:
                idx = np.argmin(np.abs(time_grid - times[0]))
                series[idx] = concern_scores[0] if len(concern_scores) > 0 else 0.0
        
        return series
    
    def _emotions_to_concern_score(self, emotion_scores: List[Dict]) -> np.ndarray:
        """
        Convert emotion scores to concern scores.

        Maps emotions to concern levels:
        - High concern: sadness, anger, fear, disgust
        - Medium concern: confusion, embarrassment
        - Low concern: neutral, positive emotions
        Input:
            emotion_scores - List of emotion detection results (can be empty list)

        Output:
            Array of concern scores (0-1), one per input score
        """
        if not emotion_scores or len(emotion_scores) == 0:
            return np.array([])
        
        concern_scores = []
        
        for score in emotion_scores:
            if not isinstance(score, dict):
                # Skip invalid entries
                concern_scores.append(0.0)
                continue
                
            emotions = score.get("emotions", {})
            dominant = score.get("dominant_emotion", "neutral")
            confidence = score.get("confidence", 0.0)
            
            # Map emotion to concern level
            high_concern_emotions = ["sad", "sadness", "grief", "angry", "anger", 
                                     "fear", "fearful", "disgust", "disgusted"]
            medium_concern_emotions = ["confusion", "embarrassment", "nervousness"]
            
            if dominant in high_concern_emotions:
                concern_level = confidence * 1.0  # Full weight
            elif dominant in medium_concern_emotions:
                concern_level = confidence * 0.6  # Reduced weight
            elif dominant in ["neutral"]:
                concern_level = confidence * 0.2  # Very low weight
            else:
                # Positive emotions
                concern_level = confidence * 0.1  # Minimal concern
            
            concern_scores.append(concern_level)
        
        return np.array(concern_scores)
    
    def smooth_scores(self, scores: np.ndarray, method: str = "moving_average",
                     window_size: float = 2.0, time_resolution: float = 0.1) -> np.ndarray:
        """
        Apply temporal smoothing to scores.
        Input:
            scores - Array of scores over time
            method - Smoothing method ('moving_average', 'ema', 'gaussian')
            window_size - Smoothing window size in seconds
            time_resolution - Time resolution of scores (seconds per sample)

        Output:
            Smoothed scores
        """
        if len(scores) == 0:
            return scores
        
        window_samples = int(window_size / time_resolution)
        window_samples = max(1, window_samples)  # At least 1 sample
        
        if method == "moving_average":
            # Simple moving average
            smoothed = uniform_filter1d(scores, size=window_samples, mode='nearest')
        elif method == "ema":
            # Exponential moving average
            alpha = 2.0 / (window_samples + 1)
            smoothed = np.zeros_like(scores)
            smoothed[0] = scores[0]
            for i in range(1, len(scores)):
                smoothed[i] = alpha * scores[i] + (1 - alpha) * smoothed[i-1]
        elif method == "gaussian":
            # Gaussian smoothing
            sigma = window_samples / 3.0
            smoothed = signal.gaussian(len(scores), std=sigma)
            smoothed = smoothed / np.sum(smoothed)
            smoothed = np.convolve(scores, smoothed, mode='same')
        else:
            smoothed = scores
        
        return smoothed
    
    def generate_concern_flags(self, smoothed_scores: np.ndarray, time_grid: np.ndarray,
                              threshold: float = 0.6, min_duration: float = 2.0) -> List[Dict]:
        """
        Generate concern flags from smoothed scores.
        Input:
            smoothed_scores - Smoothed concern scores over time
            time_grid - Time grid corresponding to scores
            threshold - Score threshold for flagging
            min_duration - Minimum duration in seconds for a flag

        Output:
            List of flag dictionaries with start_time, end_time, confidence, severity
        """
        flags = []
        
        if len(smoothed_scores) == 0:
            return flags
        
        # Find segments above threshold
        above_threshold = smoothed_scores >= threshold
        
        if not np.any(above_threshold):
            return flags
        
        # Find contiguous segments
        in_segment = False
        segment_start = None
        
        for i, is_above in enumerate(above_threshold):
            if is_above and not in_segment:
                # Start of new segment
                segment_start = i
                in_segment = True
            elif not is_above and in_segment:
                # End of segment
                segment_end = i - 1
                duration = time_grid[segment_end] - time_grid[segment_start]
                
                if duration >= min_duration:
                    # Calculate average confidence for segment
                    segment_scores = smoothed_scores[segment_start:segment_end+1]
                    avg_confidence = np.mean(segment_scores)
                    max_confidence = np.max(segment_scores)
                    
                    # Determine severity
                    if max_confidence >= 0.8:
                        severity = "high"
                    elif max_confidence >= 0.65:
                        severity = "medium"  # High Concern
                    else:
                        severity = "low"
                    
                    flags.append({
                        "start_time": float(time_grid[segment_start]),
                        "end_time": float(time_grid[segment_end]),
                        "confidence": float(avg_confidence),
                        "max_confidence": float(max_confidence),
                        "severity": severity,
                        "duration": float(duration)
                    })
                
                in_segment = False
        
        # Handle segment that extends to end
        if in_segment:
            segment_end = len(smoothed_scores) - 1
            duration = time_grid[segment_end] - time_grid[segment_start]
            
            if duration >= min_duration:
                segment_scores = smoothed_scores[segment_start:segment_end+1]
                avg_confidence = np.mean(segment_scores)
                max_confidence = np.max(segment_scores)
                
                if max_confidence >= 0.8:
                    severity = "high"
                elif max_confidence >= 0.65:
                    severity = "medium"  # High Concern
                else:
                    severity = "low"
                
                flags.append({
                    "start_time": float(time_grid[segment_start]),
                    "end_time": float(time_grid[segment_end]),
                    "confidence": float(avg_confidence),
                    "max_confidence": float(max_confidence),
                    "severity": severity,
                    "duration": float(duration)
                })
        
        return flags
    
    def apply_vad_weighting(self, scores: np.ndarray, time_grid: np.ndarray,
                           speech_segments: List[Dict]) -> np.ndarray:
        """
        Apply VAD-based weighting to scores.

        Reduces scores during non-speech segments.
        Input:
            scores - Concern scores over time
            time_grid - Time grid corresponding to scores
            speech_segments - List of speech segments from VAD

        Output:
            Weighted scores
        """
        weighted_scores = scores.copy()
        
        for i, time in enumerate(time_grid):
            # Check if time is in speech segment
            in_speech = False
            for segment in speech_segments:
                if segment["start_time"] <= time <= segment["end_time"]:
                    in_speech = True
                    break
            
            if not in_speech:
                # Reduce weight for non-speech
                weighted_scores[i] *= 0.3
        
        return weighted_scores
