"""Free public models for behavioral analysis using enhanced emotion detection."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import librosa

# Import new emotion detection modules (make them optional)
# Each module is imported separately so missing dependencies don't block others
AUDIO_DETECTOR_AVAILABLE = False
TEXT_DETECTOR_AVAILABLE = False
VIDEO_DETECTOR_AVAILABLE = False
VAD_DETECTOR_AVAILABLE = False
TEMPORAL_FUSION_AVAILABLE = False

AudioEmotionDetector = None
TextEmotionDetector = None
VideoEmotionDetector = None
VADDetector = None
TemporalFusion = None

try:
    from .audio_emotion_detector import AudioEmotionDetector
    AUDIO_DETECTOR_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Info: Audio emotion detector not available: {e}")

try:
    from .text_emotion_detector import TextEmotionDetector
    TEXT_DETECTOR_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Info: Text emotion detector not available: {e}")

try:
    from .video_emotion_detector import VideoEmotionDetector
    # Check if it can actually be instantiated (DeepFace might not be available)
    VIDEO_DETECTOR_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Info: Video emotion detector not available: {e}")

try:
    from .vad_detector import VADDetector
    VAD_DETECTOR_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Info: VAD detector not available: {e}")

try:
    from .temporal_fusion import TemporalFusion
    TEMPORAL_FUSION_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Info: Temporal fusion not available: {e}")

try:
    from .psychiatric_keyword_detector import PsychiatricKeywordDetector
    KEYWORD_DETECTOR_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Info: Psychiatric keyword detector not available: {e}")
    KEYWORD_DETECTOR_AVAILABLE = False

# At least one new module should be available to use enhanced mode
NEW_MODULES_AVAILABLE = (AUDIO_DETECTOR_AVAILABLE or TEXT_DETECTOR_AVAILABLE or 
                        VIDEO_DETECTOR_AVAILABLE or VAD_DETECTOR_AVAILABLE or 
                        TEMPORAL_FUSION_AVAILABLE)

# Try to import transformers (for fallback)
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class FreeBehavioralAnalyzer:
    """
    Behavioral analyzer using enhanced free public models with temporal fusion.
    """
    
    def __init__(self, device: Optional[str] = None, use_enhanced: bool = True,
                 audio_window_size: float = 3.0, audio_hop_size: float = 0.5,
                 video_sample_rate: float = 1.0, fusion_weights: Optional[Dict] = None,
                 smoothing_method: str = "moving_average", smoothing_window: float = 2.0,
                 concern_threshold: float = 0.6, min_flag_duration: float = 2.0):
        """
        Initialize analyzer with enhanced models.
        Input:
            device - Device to run models on ('cpu', 'cuda', 'mps')
            use_enhanced - Whether to use enhanced emotion detection modules
            audio_window_size - Sliding window size for audio emotion (seconds)
            audio_hop_size - Hop size for audio emotion (seconds)
            video_sample_rate - Frames per second to analyze for video (1.0 = 1 fps)
            fusion_weights - Dict with 'audio', 'text', 'video' weights (default: 0.4, 0.4, 0.2)
            smoothing_method - Smoothing method ('moving_average', 'ema', 'gaussian')
            smoothing_window - Smoothing window size (seconds)
            concern_threshold - Threshold for concern flagging (0-1)
            min_flag_duration - Minimum duration for flags (seconds)
        """
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
        
        self.use_enhanced = use_enhanced and NEW_MODULES_AVAILABLE
        
        print(f"Initializing free models on device: {self.device}")
        print(f"Using enhanced emotion detection: {self.use_enhanced}")
        
        if self.use_enhanced:
            # Initialize enhanced modules (only if available)
            self.audio_detector = None
            self.text_detector = None
            self.video_detector = None
            self.vad_detector = None
            
            if AUDIO_DETECTOR_AVAILABLE:
                try:
                    self.audio_detector = AudioEmotionDetector(
                        model_name="superb/wav2vec2-base-superb-er",
                        device=self.device
                    )
                    # Only print success if detector is actually initialized
                    if self.audio_detector.processor is not None and self.audio_detector.model is not None:
                        self.audio_window_size = audio_window_size
                        self.audio_hop_size = audio_hop_size
                        print("✓ Audio emotion detector initialized")
                    else:
                        print("Warning: Audio emotion detector failed to initialize properly")
                        self.audio_detector = None
                except Exception as e:
                    print(f"Warning: Could not initialize audio emotion detector: {e}")
                    self.audio_detector = None
            
            if TEXT_DETECTOR_AVAILABLE:
                try:
                    self.text_detector = TextEmotionDetector(
                        model_name="SamLowe/roberta-base-go_emotions",
                        device=self.device
                    )
                    print("✓ Text emotion detector initialized")
                except Exception as e:
                    print(f"Warning: Could not initialize text emotion detector: {e}")
                    self.text_detector = None
            
            if VIDEO_DETECTOR_AVAILABLE:
                try:
                    self.video_detector = VideoEmotionDetector(
                        model_name="VGG-Face",
                        backend="opencv",
                        use_pytorch_alternative=True  # Use FER if DeepFace unavailable
                    )
                    # Check if it actually initialized (either DeepFace or PyTorch alternative)
                    if (self.video_detector.use_deepface or 
                        self.video_detector.pytorch_detector is not None):
                        self.video_sample_rate = video_sample_rate
                        if self.video_detector.use_deepface:
                            print("✓ Video emotion detector initialized (DeepFace)")
                        else:
                            print("✓ Video emotion detector initialized (FER - PyTorch)")
                    else:
                        print("Info: Video emotion detector unavailable")
                        self.video_detector = None
                except Exception as e:
                    print(f"Warning: Could not initialize video emotion detector: {e}")
                    self.video_detector = None
            
            if VAD_DETECTOR_AVAILABLE:
                try:
                    self.vad_detector = VADDetector()
                    print("✓ VAD detector initialized")
                except Exception as e:
                    print(f"Warning: Could not initialize VAD detector: {e}")
                    self.vad_detector = None
            
            # Initialize temporal fusion (only if available)
            if TEMPORAL_FUSION_AVAILABLE and TemporalFusion is not None:
                if fusion_weights:
                    weights = fusion_weights
                else:
                    # Adjust weights based on available detectors
                    audio_w = 0.4 if self.audio_detector else 0.0
                    text_w = 0.4 if self.text_detector else 0.0
                    video_w = 0.2 if self.video_detector else 0.0
                    
                    # Normalize if some are missing
                    total = audio_w + text_w + video_w
                    if total > 0:
                        weights = {
                            "audio": audio_w / total,
                            "text": text_w / total,
                            "video": video_w / total
                        }
                    else:
                        weights = {"audio": 0.4, "text": 0.4, "video": 0.2}
                
                try:
                    self.temporal_fusion = TemporalFusion(
                        audio_weight=weights.get("audio", 0.4),
                        text_weight=weights.get("text", 0.4),
                        video_weight=weights.get("video", 0.2)
                    )
                    print("✓ Temporal fusion initialized")
                except Exception as e:
                    print(f"Warning: Could not initialize temporal fusion: {e}")
                    self.temporal_fusion = None
            
            # Initialize keyword detector
            if KEYWORD_DETECTOR_AVAILABLE:
                try:
                    from .psychiatric_keyword_detector import PsychiatricKeywordDetector
                    self.keyword_detector = PsychiatricKeywordDetector()
                    print("✓ Psychiatric keyword detector initialized")
                except Exception as e:
                    print(f"Warning: Could not initialize keyword detector: {e}")
                    self.keyword_detector = None
            else:
                self.keyword_detector = None
            
            if not (self.audio_detector or self.text_detector or self.video_detector or self.vad_detector or self.temporal_fusion):
                # Fallback to old implementation
                print("Warning: No enhanced emotion detection modules could be initialized. Falling back to old implementation.")
                self.use_enhanced = False
                self._init_fallback_models()
            else:
                self.temporal_fusion = None
                print("Info: Temporal fusion not available")
            
            self.smoothing_method = smoothing_method
            self.smoothing_window = smoothing_window
        self.concern_threshold = concern_threshold
        self.min_flag_duration = min_flag_duration
        
        # Initialize keyword detector if available
        self.keyword_detector = None
        if KEYWORD_DETECTOR_AVAILABLE:
            try:
                self.keyword_detector = PsychiatricKeywordDetector()
                print("✓ Psychiatric keyword detector initialized")
            except Exception as e:
                print(f"Warning: Could not initialize keyword detector: {e}")
                self.keyword_detector = None
        else:
            # Fallback to old implementation
            self._init_fallback_models()
    
    def _init_fallback_models(self):
        """
        Initialize fallback models (old implementation).
        Input:
            None

        Output:
            None
        """
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers not available, analysis will be limited")
            return
        
        # Initialize old text model
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
            self.text_model = AutoModelForSequenceClassification.from_pretrained(
                "j-hartmann/emotion-english-distilroberta-base"
            )
            if self.device == "mps":
                self.text_model = self.text_model.to("cpu")
                self.text_device = "cpu"
            else:
                self.text_model = self.text_model.to(self.device)
                self.text_device = self.device
            self.text_model.eval()
        except Exception as e:
            print(f"Warning: Could not load fallback text model: {e}")
            self.text_model = None
        
        # Initialize old audio pipeline
        try:
            self.audio_pipeline = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Warning: Could not load fallback audio pipeline: {e}")
            self.audio_pipeline = None
    
    def analyze(self, video_path: str, audio_path: str, transcript_text: str,
                transcript_segments: Optional[List[Dict]] = None,
                fps: float = 30.0) -> Dict:
        """
        Analyze video/audio/text for behavioral concerns.
        Input:
            video_path - Path to video file
            audio_path - Path to audio file
            transcript_text - Full transcribed text
            transcript_segments - List of transcript segments with timestamps
            fps - Video frames per second

        Output:
            Dictionary with analysis results:
                - visual_flags: List of visual concern flags
                - audio_flags: List of audio concern flags
                - text_flags: List of text concern flags
                - combined_flags: List of combined flags
        """
        if self.use_enhanced:
            return self._analyze_enhanced(video_path, audio_path, transcript_text, 
                                        transcript_segments, fps)
        else:
            return self._analyze_fallback(video_path, audio_path, transcript_text,
                                         transcript_segments, fps)
    
    def _analyze_enhanced(self, video_path: str, audio_path: str, transcript_text: str,
                         transcript_segments: Optional[List[Dict]], fps: float) -> Dict:
        """
        Enhanced analysis using new modules with temporal fusion.
        Input:
            video_path - path to the video file (None for audio-only)
            audio_path - path to the extracted audio
            transcript_text - full transcript
            transcript_segments - timestamped transcript segments
            fps - video frame rate

        Output:
            Analysis results dictionary (flags per modality and per-modality emotion_series)
        """
        print("Using enhanced emotion detection pipeline...")
        results = {
            "visual_flags": [],
            "audio_flags": [],
            "text_flags": [],
            "combined_flags": [],
            "frame_predictions": []
        }
        
        # Get video duration (only if video_path is provided and valid)
        import cv2
        duration = 0.0
        video_fps = fps
        
        if video_path and video_path.strip() and os.path.exists(video_path):
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    video_fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else fps
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / video_fps if video_fps > 0 else 0
                cap.release()
            except Exception as e:
                print(f"Warning: Could not get video info: {e}")
                duration = 0.0
        elif not video_path or not video_path.strip():
            print("No video path provided (audio-only file), estimating duration from audio")
        
        # If no video or duration is 0, estimate from audio
        if duration == 0 and audio_path and os.path.exists(audio_path):
            try:
                import librosa
                y, sr = librosa.load(audio_path, sr=16000)
                duration = len(y) / sr
            except Exception as e:
                print(f"Warning: Could not load audio to estimate duration: {e}")
                duration = 10.0  # Default fallback
        
        # Step 1: Voice Activity Detection
        speech_segments = []
        if self.vad_detector and audio_path and os.path.exists(audio_path):
            try:
                speech_segments = self.vad_detector.detect_speech_segments(audio_path)
            except Exception as e:
                print(f"Warning: VAD failed: {e}")
        
        # Step 2: Audio Emotion Detection (sliding windows)
        audio_emotions = []
        if self.audio_detector and audio_path and os.path.exists(audio_path):
            try:
                audio_emotions = self.audio_detector.detect_emotions(
                    audio_path,
                    window_size=self.audio_window_size,
                    hop_size=self.audio_hop_size
                )
                # Apply smoothing
                audio_emotions = self.audio_detector.smooth_emotions(
                    audio_emotions,
                    method=self.smoothing_method,
                    window_size=self.smoothing_window
                )
            except Exception as e:
                print(f"Warning: Audio emotion detection failed: {e}")
        
        # Step 3: Text Emotion Detection (multi-label) and Keyword Detection
        text_emotions = []
        text_emotion_scores = []  # unfiltered, for the emotion timeline
        keyword_flags = []
        
        if transcript_segments:
            # Text emotion detection with higher threshold
            if self.text_detector:
                try:
                    text_emotion_scores = self.text_detector.detect_emotions_for_segments(
                        transcript_segments,
                        threshold=0.4  # Increased from 0.3 to reduce false positives
                    )
                    # Filter to only include high-confidence emotions
                    text_emotions = [e for e in text_emotion_scores if e.get("confidence", 0) >= 0.5]
                    print(f"Detected {len(text_emotions)} text emotion scores (filtered).")
                except Exception as e:
                    print(f"Warning: Text emotion detection failed: {e}")
            
            # Keyword-based detection (takes priority)
            if self.keyword_detector:
                try:
                    keyword_flags = self.keyword_detector.detect_keywords_for_segments(
                        transcript_segments,
                        min_confidence=0.5
                    )
                    print(f"Detected {len(keyword_flags)} keyword-based concerns.")
                except Exception as e:
                    print(f"Warning: Keyword detection failed: {e}")
        
        # Step 4: Video Emotion Detection (only if video_path is valid)
        video_emotions = []
        if (self.video_detector and video_path and video_path.strip() and 
            os.path.exists(video_path) and os.path.getsize(video_path) > 0):
            try:
                video_emotions = self.video_detector.detect_emotions(
                    video_path,
                    sample_rate=self.video_sample_rate,
                    fps=video_fps
                )
                # Apply smoothing
                video_emotions = self.video_detector.smooth_emotions(
                    video_emotions,
                    method=self.smoothing_method,
                    window_size=self.smoothing_window
                )
                print(f"Detected {len(video_emotions)} video emotion scores.")
            except Exception as e:
                print(f"Warning: Video emotion detection failed: {e}")
        elif not video_path or not video_path.strip():
            print("Skipping video emotion detection (audio-only file)")
        
        # Step 5: Temporal Fusion (if available)
        if self.temporal_fusion:
            time_resolution = 0.1  # 100ms resolution
            fused_scores = self.temporal_fusion.fuse_modalities(
                audio_emotions,
                text_emotions,
                video_emotions,
                duration,
                time_resolution
            )
            
            # Apply VAD weighting
            if speech_segments:
                time_grid = np.linspace(0, duration, len(fused_scores))
                fused_scores = self.temporal_fusion.apply_vad_weighting(
                    fused_scores,
                    time_grid,
                    speech_segments
                )
            
            # Apply temporal smoothing
            smoothed_scores = self.temporal_fusion.smooth_scores(
                fused_scores,
                method=self.smoothing_method,
                window_size=self.smoothing_window,
                time_resolution=time_resolution
            )
            
            # Step 6: Generate flags with increased threshold
            time_grid = np.linspace(0, duration, len(smoothed_scores))
            concern_flags = self.temporal_fusion.generate_concern_flags(
                smoothed_scores,
                time_grid,
                threshold=max(self.concern_threshold, 0.7),  # Increased to reduce false positives
                min_duration=self.min_flag_duration
            )
        else:
            # Fallback: combine flags from individual modalities
            concern_flags = []
            all_emotions = audio_emotions + text_emotions + video_emotions
            # Simple combination without fusion
            for emotion in all_emotions:
                start_time = emotion.get("start_time", emotion.get("timestamp", 0.0))
                end_time = emotion.get("end_time", start_time + 1.0)
                confidence = emotion.get("confidence", 0.5)
                
                if confidence >= self.concern_threshold:
                    concern_flags.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "confidence": confidence,
                        "max_confidence": confidence,
                        "severity": "high" if confidence >= 0.8 else ("medium" if confidence >= 0.65 else "low"),  # High Concern reintroduced
                        "duration": end_time - start_time
                    })
        
        # Convert to flag format and integrate keyword flags
        for flag in concern_flags:
            # Check if keyword flags overlap with this flag (keyword flags take priority)
            concern_name = "Behavioral_Concern"
            max_confidence = flag["confidence"]
            severity = flag["severity"]
            matched_keywords = []
            
            # Check keyword flags first (they're more specific)
            for kw_flag in keyword_flags:
                if (kw_flag["start_time"] <= flag["end_time"] and 
                    kw_flag["end_time"] >= flag["start_time"]):
                    concern_name = kw_flag["concern_name"]
                    max_confidence = max(max_confidence, kw_flag["confidence"])
                    severity = kw_flag["severity"]  # Keyword severity takes priority
                    matched_keywords = kw_flag.get("matched_keywords", [])
                    break
            
            # If no keyword match, try to get from emotions
            if concern_name == "Behavioral_Concern":
                for emotion_list in [audio_emotions, text_emotions, video_emotions]:
                    for emotion in emotion_list:
                        emotion_time = emotion.get("start_time", emotion.get("timestamp", 0))
                        if flag["start_time"] <= emotion_time <= flag["end_time"]:
                            dominant = emotion.get("dominant_emotion", "")
                            # Map to specific concern
                            concern_map = {
                                "sad": "Depression", "sadness": "Depression", "grief": "Depression",
                                "angry": "Agitation", "anger": "Agitation",
                                "fear": "Anxiety", "fearful": "Anxiety", "nervousness": "Anxiety",
                                "disgust": "Negative_Emotion", "disgusted": "Negative_Emotion"
                            }
                            if dominant in concern_map:
                                concern_name = concern_map[dominant]
                                break
            
            concern_flag = {
                "start_time": flag["start_time"],
                "end_time": flag["end_time"],
                "start_frame": int(flag["start_time"] * fps),
                "end_frame": int(flag["end_time"] * fps),
                "concern_type": 0,
                "concern_name": concern_name,
                "confidence": max_confidence,
                "severity": severity,
                "modality": "combined",
                "severity_reasoning": f"Keyword-based: {', '.join(matched_keywords)}" if matched_keywords else None
            }
            results["combined_flags"].append(concern_flag)
        
        # Add keyword flags that don't overlap with emotion-based flags
        for kw_flag in keyword_flags:
            overlaps = False
            for existing_flag in results["combined_flags"]:
                if (kw_flag["start_time"] <= existing_flag["end_time"] and 
                    kw_flag["end_time"] >= existing_flag["start_time"]):
                    overlaps = True
                    # Update existing flag if keyword is more specific/higher confidence
                    if kw_flag["confidence"] > existing_flag["confidence"]:
                        existing_flag["concern_name"] = kw_flag["concern_name"]
                        existing_flag["confidence"] = kw_flag["confidence"]
                        existing_flag["severity"] = kw_flag["severity"]
                        existing_flag["severity_reasoning"] = f"Keyword-based: {', '.join(kw_flag.get('matched_keywords', []))}"
                    break
            
            if not overlaps:
                # Add as new flag
                results["combined_flags"].append({
                    "start_time": kw_flag["start_time"],
                    "end_time": kw_flag["end_time"],
                    "start_frame": int(kw_flag["start_time"] * fps),
                    "end_frame": int(kw_flag["end_time"] * fps),
                    "concern_type": 0,
                    "concern_name": kw_flag["concern_name"],
                    "confidence": kw_flag["confidence"],
                    "severity": kw_flag["severity"],
                    "modality": "text",
                    "severity_reasoning": f"Keyword-based detection: {', '.join(kw_flag.get('matched_keywords', []))}"
                })
        
        # Create modality-specific flags (for backward compatibility)
        # Map emotions to flags for each modality
        results["audio_flags"] = self._emotions_to_flags(audio_emotions, fps, "audio")
        results["text_flags"] = self._emotions_to_flags(text_emotions, fps, "text")
        results["visual_flags"] = self._emotions_to_flags(video_emotions, fps, "visual")

        # Raw per-modality emotion scores, used to build the emotion timeline
        results["emotion_series"] = {
            "audio": audio_emotions,
            "text": text_emotion_scores,
            "video": video_emotions
        }

        return results
    
    def _emotions_to_flags(self, emotions: List[Dict], fps: float, modality: str) -> List[Dict]:
        """
        Convert emotion results to flag format.
        Input:
            emotions - list of timestamped emotion results
            fps - video frame rate
            modality - 'audio', 'video' or 'text'

        Output:
            List of flag dictionaries
        """
        flags = []
        
        for emotion in emotions:
            start_time = emotion.get("start_time", emotion.get("timestamp", 0.0))
            end_time = emotion.get("end_time", start_time + 1.0)
            confidence = emotion.get("confidence", 0.5)
            dominant = emotion.get("dominant_emotion", "neutral")
            
            # Map to concern
            concern_map = {
                "sad": "Depression", "sadness": "Depression", "grief": "Depression",
                "angry": "Agitation", "anger": "Agitation",
                "fear": "Anxiety", "fearful": "Anxiety",
                "disgust": "Negative_Emotion", "disgusted": "Negative_Emotion"
            }
            concern_name = concern_map.get(dominant, "Emotional_Concern")
            
            # Determine severity
            severity = "high" if confidence >= 0.75 else ("medium" if confidence >= 0.65 else "low")  # High Concern reintroduced
            
            if severity != "low" or confidence > 0.5:
                flags.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "start_frame": int(start_time * fps),
                    "end_frame": int(end_time * fps),
                    "concern_type": 0,
                    "concern_name": concern_name,
                    "confidence": confidence,
                    "severity": severity,
                    "modality": modality
                })
        
        return flags
    
    def _analyze_fallback(self, video_path: str, audio_path: str, transcript_text: str,
                         transcript_segments: Optional[List[Dict]], fps: float) -> Dict:
        """
        Fallback analysis using old implementation.
        Input:
            video_path - path to the video file (None for audio-only)
            audio_path - path to the extracted audio
            transcript_text - full transcript
            transcript_segments - timestamped transcript segments
            fps - video frame rate

        Output:
            Analysis results dictionary
        """
        # This is the old implementation - kept for backward compatibility
        results = {
            "visual_flags": [],
            "audio_flags": [],
            "text_flags": [],
            "combined_flags": [],
            "frame_predictions": []
        }
        
        # Simple text analysis
        if transcript_segments:
            for segment in transcript_segments:
                text = segment.get("text", "")
                if text and hasattr(self, 'text_model') and self.text_model:
                    # Use old text model
                    pass  # Simplified for now
        
        return results
    
    def is_available(self) -> bool:
        """
        Check if models are available.
        Input:
            None

        Output:
            True if at least one detector loaded
        """
        if self.use_enhanced:
            return (self.audio_detector is not None or 
                   self.text_detector is not None or 
                   self.video_detector is not None)
        else:
            return (hasattr(self, 'text_model') and self.text_model is not None) or \
                   (hasattr(self, 'audio_pipeline') and self.audio_pipeline is not None)
