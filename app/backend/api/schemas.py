"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class UploadResponse(BaseModel):
    """
    Response after video upload.
    """
    video_id: str
    filename: str
    message: str


class FlagSegment(BaseModel):
    """
    A flagged segment with temporal information.
    """
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    concern_type: int
    concern_name: str
    confidence: float
    severity: str  # "high", "medium", "low"
    modality: str  # "visual", "audio", "text", "combined"
    severity_reasoning: Optional[str] = None  # LLM reasoning for severity level


class WordTimestamp(BaseModel):
    """
    Word with timestamp.
    """
    word: str
    start_time: float
    end_time: float
    confidence: Optional[float] = None


class TranscriptSegment(BaseModel):
    """
    Transcript segment with timestamps.
    """
    start_time: float
    end_time: float
    text: str
    words: List[WordTimestamp]
    flags: List[FlagSegment] = []


class AnalysisStatus(BaseModel):
    """
    Status of analysis job.
    """
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    created_at: datetime
    updated_at: datetime


class AnalysisResponse(BaseModel):
    """
    Complete analysis results.
    """
    job_id: str
    video_id: str
    duration: float
    fps: float
    num_frames: int
    
    # Flags by modality
    visual_flags: List[FlagSegment] = []
    audio_flags: List[FlagSegment] = []
    text_flags: List[FlagSegment] = []
    combined_flags: List[FlagSegment] = []
    
    # Transcript
    transcript: List[TranscriptSegment] = []
    
    # Frame-level predictions
    frame_predictions: List[Dict[str, Any]] = []
    
    # Audio URL for frontend
    audio_url: Optional[str] = None
    
    # Statistics
    statistics: Dict[str, Any] = {}
    
    # File type (video or audio)
    file_type: Optional[str] = None  # "video" or "audio"
    
    # Panel data for Summary Dashboard
    evaluations: Optional[List[Dict[str, Any]]] = []  # For Missing Questions, Protective Factors, Evaluation Details panels
    emotion_summary: Optional[Dict[str, Any]] = None  # For Emotion Timeline panel
    metadata: Optional[Dict[str, Any]] = None  # Files and modalities analyzed, engine notes

    # Which engine produced the flags: {"name": "telesense_model" | "pretrained", "checkpoint", "note"}
    engine: Optional[Dict[str, Any]] = None

    # Metadata
    created_at: datetime
    processing_time: float
