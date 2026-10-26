"""
Preprocessing utilities for audio, video, and text modalities.
"""

import torch
import torchaudio
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Optional, Tuple, List
import librosa


class AudioPreprocessor:
    """
    Audio preprocessing pipeline.
    """
    
    def __init__(
        self,
        sample_rate=16000,
        max_length=10,
        normalize=True
    ):
        """
        Initialize the AudioPreprocessor.
        Input:
            sample_rate - target sample rate in Hz
            max_length - maximum clip length in seconds (longer audio is truncated)
            normalize - normalize the waveform amplitude when True
        """
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.max_samples = int(sample_rate * max_length)
        self.normalize = normalize
        
    def __call__(self, audio_path: str) -> torch.Tensor:
        """
        Load and preprocess audio file.
        Input:
            audio_path - Path to audio file

        Output:
            Preprocessed audio waveform tensor of shape (seq_len,)
        """
        # Load audio
        try:
            waveform, sr = torchaudio.load(audio_path)
        except:
            # Fallback to librosa if torchaudio fails
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Remove batch dimension
        waveform = waveform.squeeze(0)
        
        # Truncate or pad to max_length
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]
        elif waveform.shape[0] < self.max_samples:
            padding = self.max_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Normalize
        if self.normalize:
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform
    
    def create_attention_mask(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask for audio.
        Input:
            waveform - Audio waveform tensor

        Output:
            Attention mask tensor
        """
        # For now, return all ones (can be extended for actual padding detection)
        return torch.ones(waveform.shape[0], dtype=torch.bool)


class VideoPreprocessor:
    """
    Video preprocessing pipeline.
    """
    
    def __init__(
        self,
        fps=30,
        frame_size=224,
        max_frames=300,
        face_detection=True
    ):
        """
        Initialize the VideoPreprocessor.
        Input:
            fps - frames per second to sample
            frame_size - output frame height and width in pixels
            max_frames - maximum number of frames kept per clip
            face_detection - crop to the detected face when True
        """
        self.fps = fps
        self.frame_size = frame_size
        self.max_frames = max_frames
        self.face_detection = face_detection
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Face detection (if enabled)
        if face_detection:
            try:
                import face_recognition
                self.face_detector = face_recognition
            except ImportError:
                # face_recognition not available, disable face detection
                self.face_detection = False
                self.face_detector = None
        else:
            self.face_detector = None
    
    def __call__(self, video_path: str) -> torch.Tensor:
        """
        Load and preprocess video file.
        Input:
            video_path - Path to video file

        Output:
            Preprocessed video frames tensor of shape (num_frames, C, H, W)
        """
        # Load video
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        # Keep every Nth frame; sources at or below the target rate keep every frame
        target_frame_interval = max(1, round(source_fps / self.fps)) if source_fps > 0 else 1
        # Seconds between kept frames, so callers can map frame indices back to time
        self.frame_interval_s = target_frame_interval / source_fps if source_fps > 0 else 1.0 / self.fps
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at target FPS
            if frame_count % target_frame_interval == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face detection if enabled
                if self.face_detection:
                    frame = self._extract_face(frame)
                
                # Apply transforms
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
            
            frame_count += 1
            
            # Limit to max_frames
            if len(frames) >= self.max_frames:
                break
        
        cap.release()
        
        # Pad or truncate to max_frames
        if len(frames) < self.max_frames:
            # Pad with last frame
            last_frame = frames[-1] if frames else torch.zeros(3, self.frame_size, self.frame_size)
            while len(frames) < self.max_frames:
                frames.append(last_frame)
        else:
            frames = frames[:self.max_frames]
        
        # Stack frames
        video_tensor = torch.stack(frames)  # (num_frames, C, H, W)
        
        return video_tensor
    
    def _extract_face(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract face region from frame.
        Input:
            frame - Input frame as numpy array

        Output:
            Face region or original frame if no face detected
        """
        if self.face_detector is None:
            return frame
        
        try:
            face_locations = self.face_detector.face_locations(frame)
            if face_locations:
                top, right, bottom, left = face_locations[0]
                # Add padding
                padding = 20
                top = max(0, top - padding)
                left = max(0, left - padding)
                bottom = min(frame.shape[0], bottom + padding)
                right = min(frame.shape[1], right + padding)
                frame = frame[top:bottom, left:right]
            return frame
        except Exception:
            return frame


class TextPreprocessor:
    """
    Text preprocessing pipeline.
    """
    
    def __init__(
        self,
        max_length=512,
        lowercase=True
    ):
        """
        Initialize the TextPreprocessor.
        Input:
            max_length - maximum number of characters/tokens kept
            lowercase - lowercase the text when True
        """
        self.max_length = max_length
        self.lowercase = lowercase
    
    def __call__(self, text: str) -> str:
        """
        Preprocess text.
        Input:
            text - Input text string

        Output:
            Preprocessed text string
        """
        if self.lowercase:
            text = text.lower()
        
        # Basic cleaning
        text = text.strip()
        
        return text
    
    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask for text.
        Input:
            input_ids - Token IDs tensor

        Output:
            Attention mask tensor (1 for real tokens, 0 for padding)
        """
        return (input_ids != 0).long()

