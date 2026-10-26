"""Integration with existing behavioral analysis models."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import librosa


try:
    from telesense_ml.supervised.inference.temporal_predictor import TemporalPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    TemporalPredictor = None

from .llm_severity_analyzer import LLMSeverityAnalyzer


class BehavioralAnalyzer:
    """
    Analyze video using existing temporal behavioral models.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, config: Optional[Dict] = None,
                 use_llm_severity: bool = True, llm_api_key: Optional[str] = None):
        """
        Initialize analyzer.
        Input:
            checkpoint_path - Path to model checkpoint
            config - Model configuration dict
            use_llm_severity - Whether to use LLM for severity analysis
            llm_api_key - API key for LLM service
        """
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.predictor = None
        self.load_error = None
        self.use_llm_severity = use_llm_severity
        
        # Initialize LLM severity analyzer
        if use_llm_severity:
            try:
                self.llm_analyzer = LLMSeverityAnalyzer(api_key=llm_api_key)
            except Exception as e:
                print(f"Warning: Could not initialize LLM severity analyzer: {e}")
                print("Falling back to rule-based severity analysis.")
                self.llm_analyzer = None
                self.use_llm_severity = False
        else:
            self.llm_analyzer = None
        
        if not MODEL_AVAILABLE:
            self.load_error = "telesense_ml package is not importable (add src/ to PYTHONPATH or pip install -e .)"
        elif checkpoint_path:
            try:
                self.predictor = TemporalPredictor(checkpoint_path, config=config)
            except Exception as e:
                self.load_error = str(e)
                print(f"Warning: Could not load model from {checkpoint_path}: {e}")
    
    def analyze(self, video_path: str, audio_path: str, transcript_text: str,
                fps: float = 30.0, duration: Optional[float] = None,
                transcript_segments: Optional[List[Dict]] = None) -> Dict:
        """
        Analyze video/audio/text for behavioral concerns.
        Input:
            video_path - Path to video file
            audio_path - Path to audio file
            transcript_text - Transcribed text
            fps - Video frames per second
            duration - media length in seconds, used to map model frames to time (optional)
            transcript_segments - timestamped transcript segments, used to pick each flag's text (optional)

        Output:
            Dictionary with analysis results:
                - visual_flags: List of visual concern flags
                - audio_flags: List of audio concern flags
                - text_flags: List of text concern flags
                - combined_flags: List of combined flags
                - frame_predictions: Per-frame predictions
                - error: message if the model failed (only present on failure)
        """
        results = {
            "visual_flags": [],
            "audio_flags": [],
            "text_flags": [],
            "combined_flags": [],
            "frame_predictions": []
        }
        
        if not self.predictor:
            # Return empty results if model not available
            return results
        
        try:
            # Run prediction
            prediction = self.predictor.predict(
                audio_path=audio_path,
                video_path=video_path,
                text=transcript_text
            )
            
            # Convert frame predictions to time-based flags
            frame_probs = prediction.get("frame_probs", np.array([]))
            segments = prediction.get("segments", [])
            frame_predictions = prediction.get("frame_predictions", [])
            num_frames = prediction.get("num_frames", 0)
            
            # Model frames don't match video frames (inputs are resampled and padded), so use
            # the predictor's own frame spacing to map model frames to time
            seconds_per_frame = prediction.get("seconds_per_frame") or (
                duration / num_frames if duration and num_frames else 1.0 / fps)
            
            # Get class names
            class_names = self.predictor.class_names or [
                f"Concern_{i}" for i in range(frame_probs.shape[1] if len(frame_probs.shape) > 1 else 7)
            ]
            
            # Audio features for severity analysis. Visual descriptors are not extracted yet,
            # so none are passed (rather than fixed placeholder values)
            audio_features = self._extract_audio_features(audio_path) if audio_path else None
            
            # Convert segments to flag format with LLM severity analysis
            for segment in segments:
                if len(segment) >= 4:
                    start_frame, end_frame, concern_type, confidence = segment[:4]
                    
                    start_time = start_frame * seconds_per_frame
                    end_time = (end_frame + 1) * seconds_per_frame
                    if duration:
                        if start_time >= duration:
                            continue  # segment lies in the padding past the end of the media
                        end_time = min(end_time, duration)
                    concern_name = class_names[int(concern_type)] if int(concern_type) < len(class_names) else f"Concern_{concern_type}"
                    
                    # Extract text segment for this flag
                    segment_text = self._extract_text_segment(transcript_text, transcript_segments, start_time, end_time)
                    
                    # Audio features for this segment (no visual features yet)
                    seg_audio_features = self._get_segment_audio_features(audio_features, start_time, end_time) if audio_features else None
                    seg_visual_features = None
                    
                    # Determine severity using LLM or fallback
                    if self.use_llm_severity and self.llm_analyzer:
                        try:
                            severity_result = self.llm_analyzer.analyze_severity(
                                text=segment_text,
                                audio_features=seg_audio_features,
                                visual_features=seg_visual_features,
                                concern_type=concern_name
                            )
                            severity = severity_result["severity"]
                            severity_reasoning = severity_result.get("reasoning", "")
                            # Combine model confidence with LLM confidence
                            combined_confidence = (confidence + severity_result.get("confidence", 0.5)) / 2
                        except Exception as e:
                            print(f"Error in LLM severity analysis: {e}")
                            # Fallback to rule-based
                            severity = self._determine_severity_fallback(confidence, segment_text, seg_audio_features, seg_visual_features)
                            severity_reasoning = ""
                            combined_confidence = confidence
                    else:
                        # Rule-based severity
                        severity = self._determine_severity_fallback(confidence, segment_text, seg_audio_features, seg_visual_features)
                        severity_reasoning = ""
                        combined_confidence = confidence
                    
                    flag = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_frame": int(start_time * fps),
                        "end_frame": int(end_time * fps),
                        "concern_type": int(concern_type),
                        "concern_name": concern_name,
                        "confidence": float(combined_confidence),
                        "severity": severity,
                        "severity_reasoning": severity_reasoning,
                        "modality": "combined"  # Model uses all modalities
                    }
                    
                    results["combined_flags"].append(flag)
            
            # Store frame predictions
            results["frame_predictions"] = frame_predictions
            
            # For now, we'll use combined flags for all modalities
            # In a more sophisticated system, you'd separate by modality
            results["visual_flags"] = results["combined_flags"].copy()
            results["audio_flags"] = results["combined_flags"].copy()
            results["text_flags"] = results["combined_flags"].copy()
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            results["error"] = str(e)
        
        return results
    
    def _extract_audio_features(self, audio_path: str) -> Optional[Dict]:
        """
        Extract audio features for analysis.
        Input:
            audio_path - path to the audio file

        Output:
            Dictionary of tone, pitch and energy statistics, or None on failure
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Extract features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            pitch = librosa.yin(y, fmin=50, fmax=400)
            pitch_mean = np.mean(pitch[~np.isnan(pitch)]) if len(pitch) > 0 else 0
            pitch_std = np.std(pitch[~np.isnan(pitch)]) if len(pitch) > 0 else 0
            
            # Determine tone characteristics
            if pitch_std < 20:
                tone = "monotone"
            elif pitch_std > 60:
                tone = "varied"
            else:
                tone = "normal"
            
            # Speech rate (approximate)
            duration = len(y) / sr
            speech_rate = 1.0  # Default, could be improved with VAD
            
            return {
                "tone": tone,
                "pitch_mean": float(pitch_mean) if not np.isnan(pitch_mean) else 0,
                "pitch_variance": float(pitch_std) if not np.isnan(pitch_std) else 0,
                "speech_rate": speech_rate,
                "tempo": float(tempo) if not np.isnan(tempo) else 0,
                "volume": float(np.mean(np.abs(y)))
            }
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return None
    
    def _get_segment_audio_features(self, audio_features: Dict, start_time: float, end_time: float) -> Optional[Dict]:
        """
        Get audio features for a specific segment.
        Input:
            audio_features - whole-session audio features
            start_time - segment start in seconds
            end_time - segment end in seconds

        Output:
            Audio features for the segment, or None
        """
        if not audio_features:
            return None
        # For now, return global features
        # In full implementation, extract features for specific time range
        return audio_features
    
    def _extract_text_segment(self, transcript_text: str, transcript_segments: Optional[List[Dict]],
                              start_time: float, end_time: float) -> str:
        """
        Extract the transcript text spoken during a time range.
        Input:
            transcript_text - full transcript
            transcript_segments - timestamped transcript segments (optional)
            start_time - segment start in seconds
            end_time - segment end in seconds

        Output:
            Text of the transcript segments overlapping the range (limited to 500 characters);
            the start of the full transcript when no timestamps are available
        """
        if not transcript_segments:
            return transcript_text[:500]
        overlapping = [seg.get("text", "").strip() for seg in transcript_segments
                       if seg.get("start_time", 0.0) < end_time and seg.get("end_time", 0.0) > start_time]
        return " ".join(t for t in overlapping if t)[:500]
    
    def _determine_severity_fallback(
        self,
        confidence: float,
        text: str,
        audio_features: Optional[Dict],
        visual_features: Optional[Dict]
    ) -> str:
        """
        Fallback rule-based severity determination.
        Input:
            confidence - model confidence for the segment
            text - segment transcript
            audio_features - segment audio features
            visual_features - segment visual features

        Output:
            Severity level ('low', 'medium' or 'high')
        """
        severity_score = confidence
        
        # Adjust based on text content
        text_lower = text.lower()
        if any(word in text_lower for word in ["suicide", "kill", "die", "end it", "hopeless"]):
            severity_score = min(1.0, severity_score + 0.3)
        elif any(word in text_lower for word in ["depressed", "anxious", "overwhelmed", "struggling"]):
            severity_score = min(1.0, severity_score + 0.15)
        
        # Adjust based on audio
        if audio_features:
            if audio_features.get("tone") == "monotone":
                severity_score = min(1.0, severity_score + 0.1)
        
        # Determine severity
        if severity_score >= 0.75:
            return "high"
        elif severity_score >= 0.5:
            return "medium"  # High Concern
        else:
            return "low"
    
    def is_available(self) -> bool:
        """
        Check if model is available.
        Input:
            None

        Output:
            True if a trained model is loaded
        """
        return self.predictor is not None
