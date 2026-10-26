"""File upload and handling utilities."""

import os
import uuid
import shutil
import json
from pathlib import Path
from typing import Optional, Dict
from fastapi import UploadFile, HTTPException


# Runtime storage: app/storage by default, or TELESENSE_STORAGE_DIR if set
STORAGE_DIR = Path(os.getenv("TELESENSE_STORAGE_DIR", Path(__file__).resolve().parents[2] / "storage"))
UPLOAD_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"
RESULTS_DIR = STORAGE_DIR / "results"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_id(value: str) -> str:
    """
    Reject IDs that aren't UUIDs so they can't escape the data directories.
    Input:
        value - ID from the request path

    Output:
        Normalized UUID string (raises HTTP 400 if invalid)
    """
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid ID")


def save_uploaded_file(file: UploadFile, video_id: Optional[str] = None) -> tuple[str, str]:
    """
    Save uploaded file and return video_id and file path.
    Input:
        file - Uploaded file
        video_id - Optional video ID, generates new one if not provided

    Output:
        Tuple of (video_id, file_path)
    """
    if video_id is None:
        video_id = str(uuid.uuid4())
    
    # Get file extension
    ext = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
    
    # Create video directory
    video_dir = UPLOAD_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = video_dir / f"video{ext}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return video_id, str(file_path)


def get_video_path(video_id: str) -> Optional[str]:
    """
    Get path to uploaded video or audio file.
    Input:
        video_id - Video ID

    Output:
        Path to video/audio file or None if not found
    """
    video_dir = UPLOAD_DIR / _safe_id(video_id)
    if not video_dir.exists():
        return None
    
    # Find video or audio file
    # Check video files first
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        video_path = video_dir / f"video{ext}"
        if video_path.exists():
            return str(video_path)
    
    # Check audio files
    for ext in [".wav", ".mp3", ".m4a", ".flac"]:
        audio_path = video_dir / f"video{ext}"  # Still named "video" for consistency
        if audio_path.exists():
            return str(audio_path)
    
    return None


def is_audio_file(file_path: str) -> bool:
    """
    Check if file is an audio-only file.
    Input:
        file_path - Path to file

    Output:
        True if audio file, False if video file
    """
    if not file_path:
        return False
    
    audio_extensions = ('.wav', '.mp3', '.m4a', '.flac')
    return file_path.lower().endswith(audio_extensions)


def get_processed_dir(video_id: str) -> Path:
    """
    Get directory for processed files (audio, frames, etc.).
    Input:
        video_id - Video ID

    Output:
        Path to processed directory
    """
    processed_video_dir = PROCESSED_DIR / _safe_id(video_id)
    processed_video_dir.mkdir(parents=True, exist_ok=True)
    return processed_video_dir


def get_results_path(job_id: str) -> Path:
    """
    Get path to results file.
    Input:
        job_id - Job ID

    Output:
        Path to results file
    """
    return RESULTS_DIR / f"{_safe_id(job_id)}.json"


def cleanup_video(video_id: str) -> None:
    """
    Clean up uploaded and processed files for a video.
    Input:
        video_id - Video ID

    Output:
        None
    """
    video_id = _safe_id(video_id)
    video_dir = UPLOAD_DIR / video_id
    processed_dir = PROCESSED_DIR / video_id
    
    if video_dir.exists():
        shutil.rmtree(video_dir)
    
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
