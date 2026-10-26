"""Speech transcription using Whisper."""

import os
import torch
from typing import List, Dict, Optional
from pathlib import Path

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class Transcriber:
    """
    Speech-to-text transcription using Whisper.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize transcriber.
        Input:
            model_size - Whisper model size (tiny, base, small, medium, large)
            device - Device to run on (cuda, cpu, mps)
        """
        self.model_size = model_size
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """
        Load Whisper model.
        Input:
            None

        Output:
            None
        """
        # Prefer transformers over openai-whisper to avoid OpenMP and MPS conflicts
        if TRANSFORMERS_AVAILABLE:
            # Use transformers library (avoids OpenMP conflicts and MPS float64 issues)
            try:
                print(f"Loading Whisper model using transformers library (size: {self.model_size})...")
                model_name = f"openai/whisper-{self.model_size}"
                self.processor = WhisperProcessor.from_pretrained(model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(model_name, attn_implementation="eager")
                # MPS works fine with transformers
                self.model.to(self.device)
                self.use_openai_whisper = False
                print(f"Successfully loaded transformers Whisper model on {self.device}")
                return
            except Exception as e:
                print(f"Warning: Failed to load transformers Whisper: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to openai-whisper...")
        
        if WHISPER_AVAILABLE:
            # Fallback to openai-whisper library (may have OpenMP and MPS issues)
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Workaround for OpenMP conflict
            
            # Force CPU for openai-whisper on MPS to avoid float64 issues
            original_device = self.device
            if self.device == "mps":
                print("Warning: Using CPU for openai-whisper to avoid MPS float64 issues")
                self.device = "cpu"
            
            print(f"Loading Whisper model using openai-whisper library (size: {self.model_size}, device: {self.device})...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.use_openai_whisper = True
            print(f"Successfully loaded openai-whisper model on {self.device}")
        else:
            raise ImportError(
                "Neither 'whisper' nor 'transformers' library is available. "
                "Please install one: pip install openai-whisper or pip install transformers"
            )
    
    def transcribe(self, audio_path: str, language: Optional[str] = None, 
                   word_timestamps: bool = True) -> Dict:
        """
        Transcribe audio file.
        Input:
            audio_path - Path to audio file
            language - Language code (e.g., 'en'), None for auto-detect
            word_timestamps - Whether to return word-level timestamps

        Output:
            Dictionary with:
                - text: Full transcript
                - segments: List of segments with timestamps
                - words: List of words with timestamps (if word_timestamps=True)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.use_openai_whisper:
            return self._transcribe_openai_whisper(audio_path, language, word_timestamps)
        else:
            return self._transcribe_transformers(audio_path, language, word_timestamps)
    
    def _transcribe_openai_whisper(self, audio_path: str, language: Optional[str], 
                                   word_timestamps: bool) -> Dict:
        """
        Transcribe using openai-whisper library.
        Input:
            audio_path - path to the audio file
            language - language code to force (auto-detected when None)
            word_timestamps - include word-level timestamps when True

        Output:
            Dictionary with full text, segments, words and detected language
        """
        # Disable word timestamps on MPS to avoid float64 issues
        if self.device == "mps" and word_timestamps:
            print("Warning: Disabling word timestamps for openai-whisper on MPS to avoid float64 issues")
            word_timestamps = False
        
        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            verbose=False
        )
        
        # Format output
        words = []
        if word_timestamps and "words" in result:
            for word_info in result["words"]:
                words.append({
                    "word": word_info["word"],
                    "start_time": word_info["start"],
                    "end_time": word_info["end"],
                    "confidence": word_info.get("probability", None)
                })
        
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start_time": seg["start"],
                "end_time": seg["end"],
                "text": seg["text"].strip()
            })
        
        return {
            "text": result["text"].strip(),
            "segments": segments,
            "words": words,
            "language": result.get("language", "unknown")
        }
    
    def _transcribe_transformers(self, audio_path: str, language: Optional[str],
                                 word_timestamps: bool) -> Dict:
        from .whisper_transcription import transcribe_audio
        return transcribe_audio(
            self.model, self.processor, self.device, audio_path, language, word_timestamps
        )
