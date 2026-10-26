"""Voice Activity Detection using pyannote.audio."""

import os
from typing import List, Dict, Optional

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote.audio library not available. Install with: pip install pyannote.audio")


class VADDetector:
    """
    Detect voice activity in audio using pyannote.audio.
    """
    
    def __init__(self, model_name: str = "pyannote/voice-activity-detection", 
                 use_auth_token: Optional[str] = None):
        """
        Initialize VAD detector.
        Input:
            model_name - Hugging Face model name for VAD
            use_auth_token - Hugging Face token (required for some models)
        """
        if not PYANNOTE_AVAILABLE:
            # Don't raise error - allow graceful degradation
            print("Warning: pyannote.audio not available. VAD will be disabled.")
            self.pipeline = None
            return
        
        self.model_name = model_name
        self.use_auth_token = use_auth_token or os.getenv("HUGGINGFACE_TOKEN")
        
        # Load pipeline (suppress torchcodec warnings)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*torchcodec.*")
            
            print(f"Loading VAD model: {model_name}")
            try:
                if self.use_auth_token:
                    self.pipeline = Pipeline.from_pretrained(
                        model_name,
                        use_auth_token=self.use_auth_token
                    )
                else:
                    # Try without token (may work for some models)
                    self.pipeline = Pipeline.from_pretrained(model_name)
                
                # Move to GPU if available
                if hasattr(self.pipeline, 'to'):
                    import torch
                    if torch.cuda.is_available():
                        self.pipeline = self.pipeline.to(torch.device("cuda"))
                
                print("Successfully loaded VAD model")
            except Exception as e:
                print(f"Warning: Could not load VAD model: {e}")
                print("VAD will be disabled. Will assume entire audio is speech.")
                self.pipeline = None
    
    def detect_speech_segments(self, audio_path: str) -> List[Dict]:
        """
        Detect speech segments in audio.
        Input:
            audio_path - Path to audio file

        Output:
            List of dictionaries with:
                - start_time: Start time in seconds
                - end_time: End time in seconds
                - duration: Duration in seconds
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.pipeline is None:
            # Fallback: assume entire audio is speech
            import librosa
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            return [{
                "start_time": 0.0,
                "end_time": duration,
                "duration": duration
            }]
        
        try:
            # Run VAD
            vad_output = self.pipeline(audio_path)
            
            # Extract segments
            segments = []
            for segment in vad_output.get_timeline():
                start_time = segment.start
                end_time = segment.end
                duration = end_time - start_time
                
                segments.append({
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "duration": float(duration)
                })
            
            # Sort by start time
            segments.sort(key=lambda x: x["start_time"])
            
            print(f"Detected {len(segments)} speech segments")
            return segments
        except Exception as e:
            print(f"Error in VAD detection: {e}")
            # Fallback: assume entire audio is speech
            import librosa
            y, sr = librosa.load(audio_path, sr=16000)
            duration = len(y) / sr
            return [{
                "start_time": 0.0,
                "end_time": duration,
                "duration": duration
            }]
    
    def is_speech_at_time(self, time: float, segments: List[Dict], 
                         tolerance: float = 0.1) -> bool:
        """
        Check if there is speech at a given time.
        Input:
            time - Time in seconds to check
            segments - List of speech segments
            tolerance - Tolerance in seconds for edge cases

        Output:
            True if speech is detected at the given time
        """
        for segment in segments:
            if segment["start_time"] - tolerance <= time <= segment["end_time"] + tolerance:
                return True
        return False
    
    def get_speech_weight(self, time: float, segments: List[Dict]) -> float:
        """
        Get weight for a given time based on speech presence.
        Input:
            time - Time in seconds
            segments - List of speech segments

        Output:
            Weight between 0.0 (no speech) and 1.0 (speech present)
        """
        if self.is_speech_at_time(time, segments):
            return 1.0
        else:
            return 0.3  # Lower weight for non-speech segments
    
    def filter_by_speech(self, results: List[Dict], segments: List[Dict],
                        time_key: str = "start_time") -> List[Dict]:
        """
        Filter results to only include those during speech segments.
        Input:
            results - List of results with timestamps
            segments - List of speech segments
            time_key - Key in result dict that contains timestamp

        Output:
            Filtered results
        """
        filtered = []
        for result in results:
            time = result.get(time_key, 0.0)
            if self.is_speech_at_time(time, segments):
                filtered.append(result)
        return filtered
    
    def merge_segments(self, segments: List[Dict], gap_threshold: float = 0.5) -> List[Dict]:
        """
        Merge nearby speech segments.
        Input:
            segments - List of speech segments
            gap_threshold - Maximum gap in seconds to merge segments

        Output:
            Merged segments
        """
        if not segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x["start_time"])
        
        merged = [sorted_segments[0].copy()]
        
        for segment in sorted_segments[1:]:
            last_segment = merged[-1]
            
            # Check if gap is small enough to merge
            gap = segment["start_time"] - last_segment["end_time"]
            
            if gap <= gap_threshold:
                # Merge segments
                last_segment["end_time"] = segment["end_time"]
                last_segment["duration"] = last_segment["end_time"] - last_segment["start_time"]
            else:
                # Add as new segment
                merged.append(segment.copy())
        
        return merged
