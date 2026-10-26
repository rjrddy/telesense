"""Video-to-text transcription using multimodal models that process video directly."""

import os
import torch
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import cv2

try:
    from transformers import (
        AutoProcessor, AutoModelForSpeechSeq2Seq,
        WhisperProcessor, WhisperForConditionalGeneration
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class VideoTranscriber:
    """
    Video-to-text transcription using video as input (multimodal approach).
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None, use_visual_cues: bool = True):
        """
        Initialize video transcriber.
        Input:
            model_size - Model size (tiny, base, small, medium, large)
            device - Device to run on (cuda, cpu, mps)
            use_visual_cues - Whether to use visual cues from video frames (lip reading, gestures)
        """
        self.model_size = model_size
        self.use_visual_cues = use_visual_cues
        
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
        Load transcription model.
        Input:
            None

        Output:
            None
        """
        # For now, we'll use Whisper but process video frames for visual context
        # In the future, this could use video-specific ASR models like:
        # - Wav2Lip-based models
        # - Video-LLaMA
        # - Multimodal ASR models
        
        if TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading Whisper model for video transcription (size: {self.model_size})...")
                model_name = f"openai/whisper-{self.model_size}"
                self.processor = WhisperProcessor.from_pretrained(model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
                self.model.to(self.device)
                self.use_openai_whisper = False
                print(f"Successfully loaded transformers Whisper model on {self.device}")
                return
            except Exception as e:
                print(f"Warning: Failed to load transformers Whisper: {e}")
                print("Falling back to openai-whisper...")
        
        if WHISPER_AVAILABLE:
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            if self.device == "mps":
                print("Warning: Using CPU for openai-whisper to avoid MPS float64 issues")
                self.device = "cpu"
            
            print(f"Loading Whisper model using openai-whisper (size: {self.model_size}, device: {self.device})...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.use_openai_whisper = True
            print(f"Successfully loaded openai-whisper model on {self.device}")
        else:
            raise ImportError(
                "Neither 'whisper' nor 'transformers' library is available. "
                "Please install one: pip install openai-whisper or pip install transformers"
            )
    
    def transcribe(self, video_path: str, language: Optional[str] = None,
                   word_timestamps: bool = True, extract_audio: bool = True) -> Dict:
        """
        Transcribe video file directly.
        Input:
            video_path - Path to video file
            language - Language code (e.g., 'en'), None for auto-detect
            word_timestamps - Whether to return word-level timestamps
            extract_audio - Whether to extract audio from video (required for transcription)

        Output:
            Dictionary with:
                - text: Full transcript
                - segments: List of segments with timestamps
                - words: List of words with timestamps (if word_timestamps=True)
                - visual_cues: Visual information extracted from video (if use_visual_cues=True)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract audio from video (required for transcription)
        if extract_audio:
            audio_path = self._extract_audio_from_video(video_path)
        else:
            # Assume audio is already extracted or use video directly
            audio_path = video_path
        
        # Extract visual cues if enabled
        visual_cues = None
        if self.use_visual_cues:
            visual_cues = self._extract_visual_cues(video_path)
        
        # Transcribe audio (with potential visual context in future)
        if self.use_openai_whisper:
            result = self._transcribe_openai_whisper(audio_path, language, word_timestamps)
        else:
            result = self._transcribe_transformers(audio_path, language, word_timestamps)
        
        # Add visual cues to result
        if visual_cues:
            result["visual_cues"] = visual_cues
        
        # Clean up temporary audio file if we extracted it
        if extract_audio and audio_path != video_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
        
        return result
    
    def _extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from video file.
        Input:
            video_path - path to the video file

        Output:
            Path to the extracted audio (the video path if extraction fails)
        """
        import tempfile
        import subprocess
        
        # Create temporary audio file
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"temp_audio_{os.path.basename(video_path)}.wav")
        
        try:
            # Use ffmpeg to extract audio
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1",
                    "-y", audio_path
                ],
                check=True,
                capture_output=True
            )
            return audio_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: try using moviepy
            try:
                from moviepy.editor import VideoFileClip
                video = VideoFileClip(video_path)
                audio = video.audio
                audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le', verbose=False, logger=None)
                audio.close()
                video.close()
                return audio_path
            except Exception as e:
                print(f"Warning: Could not extract audio using moviepy: {e}")
                # Last resort: return video path and hope model can handle it
                return video_path
    
    def _extract_visual_cues(self, video_path: str) -> Dict:
        """
        Extract visual cues from video frames (lip reading, facial expressions, gestures).

        This is a placeholder for future enhancement with video-based ASR models.
        Currently extracts basic frame information.
        Input:
            video_path - path to the video file

        Output:
            Dictionary of video metadata and sampled-frame cues, or None on failure
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Sample frames for analysis
            sampled_frames = []
            sample_interval = max(1, int(fps / 2))  # Sample 2 frames per second
            
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize for efficiency
                    frame_rgb = cv2.resize(frame_rgb, (224, 224))
                    sampled_frames.append({
                        "frame_idx": frame_idx,
                        "timestamp": frame_idx / fps if fps > 0 else 0,
                        "frame": frame_rgb
                    })
                
                frame_idx += 1
                
                # Limit to reasonable number of frames
                if len(sampled_frames) >= 100:
                    break
            
            cap.release()
            
            return {
                "fps": fps,
                "total_frames": frame_count,
                "duration": duration,
                "sampled_frames": len(sampled_frames),
                "frame_timestamps": [f["timestamp"] for f in sampled_frames]
            }
        except Exception as e:
            print(f"Warning: Could not extract visual cues: {e}")
            return None
    
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
        """
        Transcribe using transformers library.
        Input:
            audio_path - path to the audio file
            language - language code to force (auto-detected when None)
            word_timestamps - include word-level timestamps when True

        Output:
            Dictionary with full text, segments, words and detected language
        """
        import librosa
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        
        # Process
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with language if specified
        generate_kwargs = {}
        if language:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
            generate_kwargs["forced_decoder_ids"] = forced_decoder_ids
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
        
        # Decode transcription
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # For word-level timestamps, estimate based on word positions
        words = []
        segments = []
        
        if word_timestamps and transcription.strip():
            word_list = transcription.strip().split()
            words_per_second = len(word_list) / duration if duration > 0 else 0
            
            current_time = 0.0
            for word in word_list:
                word_duration = 1.0 / words_per_second if words_per_second > 0 else 0.1
                words.append({
                    "word": word,
                    "start_time": current_time,
                    "end_time": min(current_time + word_duration, duration),
                    "confidence": None
                })
                current_time += word_duration
            
            # Group words into segments
            current_segment = {"start_time": 0.0, "text": "", "words": []}
            for word in words:
                if not current_segment["text"]:
                    current_segment["start_time"] = word["start_time"]
                
                if current_segment["text"]:
                    current_segment["text"] += " " + word["word"]
                else:
                    current_segment["text"] = word["word"]
                
                current_segment["end_time"] = word["end_time"]
                current_segment["words"].append(word)
                
                # End segment on punctuation
                if word["word"].endswith(('.', '!', '?', '。', '！', '？')):
                    segments.append({
                        "start_time": current_segment["start_time"],
                        "end_time": current_segment["end_time"],
                        "text": current_segment["text"]
                    })
                    current_segment = {"start_time": word["end_time"], "text": "", "words": []}
            
            # Add final segment if exists
            if current_segment["text"]:
                segments.append({
                    "start_time": current_segment["start_time"],
                    "end_time": current_segment["end_time"],
                    "text": current_segment["text"]
                })
        else:
            # No word timestamps - create single segment
            segments.append({
                "start_time": 0.0,
                "end_time": duration,
                "text": transcription
            })
        
        return {
            "text": transcription.strip(),
            "segments": segments,
            "words": words,
            "language": language or "unknown"
        }
