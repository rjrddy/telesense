"""Video and audio processing utilities."""

import os
import cv2
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


class VideoProcessor:
    """
    Process video files to extract audio and frames.
    """
    
    def __init__(self, fps: int = 30, frame_size: int = 224):
        """
        Initialize video processor.
        Input:
            fps - Target frames per second for frame extraction
            frame_size - Size to resize frames to
        """
        self.fps = fps
        self.frame_size = frame_size
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video file.
        Input:
            video_path - Path to video file
            output_path - Optional output path for audio file

        Output:
            Path to extracted audio file
        """
        if output_path is None:
            output_dir = Path(video_path).parent
            output_path = str(output_dir / "audio.wav")
        
        if MOVIEPY_AVAILABLE:
            # Use moviepy for audio extraction
            video = VideoFileClip(video_path)
            audio = video.audio
            if audio is not None:
                audio.write_audiofile(output_path, verbose=False, logger=None)
                audio.close()
            video.close()
        else:
            # Fallback to ffmpeg via opencv
            import subprocess
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                "-y", output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        
        return output_path
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata.
        Input:
            video_path - Path to video file

        Output:
            Dictionary with video info (fps, duration, width, height, frame_count)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "fps": fps,
            "duration": duration,
            "width": width,
            "height": height,
            "frame_count": frame_count
        }
    
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None, 
                      max_frames: Optional[int] = None) -> Tuple[str, list]:
        """
        Extract frames from video.
        Input:
            video_path - Path to video file
            output_dir - Optional output directory for frames
            max_frames - Maximum number of frames to extract

        Output:
            Tuple of (output_dir, list of frame paths)
        """
        if output_dir is None:
            output_dir = str(Path(video_path).parent / "frames")
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / self.fps))
        
        frame_paths = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at target FPS
            if frame_count % frame_interval == 0:
                # Resize frame
                frame_resized = cv2.resize(frame, (self.frame_size, self.frame_size))
                
                # Save frame
                frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(frame_path, frame_resized)
                frame_paths.append(frame_path)
                saved_count += 1
                
                if max_frames and saved_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        
        return output_dir, frame_paths
