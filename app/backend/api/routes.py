"""API routes for video analysis."""

import os
import uuid
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse

from .schemas import (
    UploadResponse, AnalysisResponse, 
    AnalysisStatus, FlagSegment, TranscriptSegment, WordTimestamp
)
from ..models.video_processor import VideoProcessor
from ..models.transcriber import Transcriber
from ..models.video_transcriber import VideoTranscriber
from ..models.analyzer import BehavioralAnalyzer
from ..models.free_analyzer import FreeBehavioralAnalyzer
from ..models.clinical_evaluator import build_evaluations
from ..models.emotion_summary import build_emotion_summary
from ..utils.file_handler import (
    save_uploaded_file, get_video_path, get_processed_dir, 
    get_results_path, cleanup_video, is_audio_file
)

router = APIRouter()

# In-memory job storage (in production, use Redis or database)
jobs: Dict[str, AnalysisStatus] = {}
job_results: Dict[str, Dict] = {}

# Trained Telesense model: set MODEL_CHECKPOINT (and optionally MODEL_CONFIG, a YAML path) to use it;
# without a usable checkpoint the pretrained-model pipeline runs instead
DEFAULT_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", None)
DEFAULT_CONFIG = os.getenv("MODEL_CONFIG", None)


def _load_config(config_path: Optional[str]) -> Optional[Dict]:
    """
    Load a YAML model configuration.
    Input:
        config_path - path to the YAML file (None returns None)

    Output:
        Configuration dictionary, or None
    """
    if not config_path:
        return None
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@router.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video or audio file for analysis.
    Input:
        file - uploaded video or audio file

    Output:
        UploadResponse with the new video_id
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Support both video and audio files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    audio_extensions = ('.wav', '.mp3', '.m4a', '.flac')
    
    if not file.filename.lower().endswith(video_extensions + audio_extensions):
        raise HTTPException(
            status_code=400, 
            detail="Only video files (mp4, avi, mov, mkv) or audio files (wav, mp3, m4a, flac) are supported"
        )
    
    try:
        video_id, file_path = save_uploaded_file(file)
        file_type = "audio" if file.filename.lower().endswith(audio_extensions) else "video"
        
        return UploadResponse(
            video_id=video_id,
            filename=file.filename,
            message=f"{file_type.capitalize()} uploaded successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.post("/analyze/{video_id}", response_model=AnalysisStatus)
async def start_analysis(video_id: str):
    """
    Start analysis job for a video.
    Input:
        video_id - ID returned by /upload

    Output:
        AnalysisStatus for the new job (analysis runs in the background)
    """
    # Check if video exists
    video_path = get_video_path(video_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Create job
    job_id = str(uuid.uuid4())
    job_status = AnalysisStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Job created",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    jobs[job_id] = job_status
    
    # Model checkpoint and config come only from server settings, never from the request
    checkpoint_path = DEFAULT_CHECKPOINT
    config = _load_config(DEFAULT_CONFIG)
    
    # Start background task
    asyncio.create_task(process_video_analysis(
        job_id, video_id, video_path, checkpoint_path, config
    ))
    
    return job_status


@router.get("/status/{job_id}", response_model=AnalysisStatus)
async def get_status(job_id: str):
    """
    Get status of analysis job.
    Input:
        job_id - analysis job ID

    Output:
        Current AnalysisStatus of the job
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@router.get("/results/{job_id}", response_model=AnalysisResponse)
async def get_results(job_id: str):
    """
    Get analysis results.
    Input:
        job_id - analysis job ID

    Output:
        AnalysisResponse with flags, transcript and statistics
    """
    if job_id not in job_results:
        # Check if job is still processing
        if job_id in jobs and jobs[job_id].status != "completed":
            raise HTTPException(status_code=202, detail="Analysis still in progress")
        raise HTTPException(status_code=404, detail="Results not found")
    
    return AnalysisResponse(**job_results[job_id])


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time progress updates.
    Input:
        websocket - client WebSocket connection
        job_id - analysis job ID to stream

    Output:
        None
    """
    await websocket.accept()
    
    try:
        while True:
            if job_id in jobs:
                status = jobs[job_id]
                await websocket.send_json({
                    "status": status.status,
                    "progress": status.progress,
                    "message": status.message
                })
                
                if status.status in ["completed", "failed"]:
                    break
            
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


@router.get("/audio/{video_id}/audio.wav")
async def get_audio_file(video_id: str):
    """
    Get extracted audio file for a video.
    Input:
        video_id - ID of the uploaded video

    Output:
        FileResponse streaming the extracted audio.wav
    """
    from pathlib import Path
    
    processed_dir = get_processed_dir(video_id)
    audio_path = Path(processed_dir) / "audio.wav"
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found. Audio may still be processing.")
    
    return FileResponse(
        str(audio_path),
        media_type="audio/wav",
        filename=f"audio_{video_id}.wav"
    )


# A single CPU analysis at a time avoids loading multiple model stacks into RAM.
_analysis_lock = asyncio.Lock()


async def process_video_analysis(job_id, video_id, video_path, checkpoint_path, config):
    async with _analysis_lock:
        await asyncio.to_thread(
            _process_video_analysis, job_id, video_id, video_path, checkpoint_path, config
        )


def _process_video_analysis(job_id: str, video_id: str, video_path: str,
                                  checkpoint_path: Optional[str], config: Optional[Dict]):
    """
    Process video or audio analysis in background.
    Input:
        job_id - analysis job ID
        video_id - ID of the uploaded file
        video_path - path to the uploaded file
        checkpoint_path - trained model checkpoint (free pretrained models are used when None)
        config - model configuration (optional)

    Output:
        None
    """
    try:
        # Check if this is an audio-only file
        is_audio_only = is_audio_file(video_path)
        
        # Update status
        jobs[job_id].status = "processing"
        jobs[job_id].progress = 0.1
        jobs[job_id].message = "Extracting audio and video info" if not is_audio_only else "Processing audio file"
        jobs[job_id].updated_at = datetime.now()
        
        # Initialize processors
        video_processor = VideoProcessor(fps=30, frame_size=224)
        
        # Get file info and extract audio
        processed_dir = get_processed_dir(video_id)
        audio_path = str(processed_dir / "audio.wav")
        
        if is_audio_only:
            # For audio files, copy to processed directory
            import shutil
            shutil.copy2(video_path, audio_path)
            # Get audio info using librosa or similar
            try:
                import librosa
                y, sr = librosa.load(audio_path, sr=None)
                duration = len(y) / sr
            except Exception as e:
                print(f"Warning: Could not load audio with librosa: {e}")
                # Fallback: use ffprobe or estimate
                duration = 60.0  # Default estimate
            fps = 30.0  # Default FPS for audio (used for frame calculations)
        else:
            # For video files, extract audio and get video info
            video_info = video_processor.get_video_info(video_path)
            fps = video_info["fps"]
            duration = video_info["duration"]
            audio_path = video_processor.extract_audio(video_path, audio_path)
        
        # Store audio URL for frontend (relative path)
        audio_url = f"/api/audio/{video_id}/audio.wav"
        
        jobs[job_id].progress = 0.3
        jobs[job_id].message = "Transcribing audio" if is_audio_only else "Transcribing video (video-to-text)"
        jobs[job_id].updated_at = datetime.now()
        
        # Transcribe audio
        if is_audio_only:
            # Use regular transcriber for audio-only files
            transcriber = Transcriber(model_size="base")
            transcript_result = transcriber.transcribe(
                audio_path=audio_path,
                word_timestamps=True
            )
        else:
            # Use video transcriber for video files
            video_transcriber = VideoTranscriber(model_size="base", use_visual_cues=True)
            transcript_result = video_transcriber.transcribe(
                video_path=video_path,
                word_timestamps=True,
                extract_audio=True  # Will extract audio internally for transcription
            )
        
        jobs[job_id].progress = 0.5
        jobs[job_id].message = "Analyzing behavioral patterns"
        jobs[job_id].updated_at = datetime.now()
        
        # Transcript segments in the shape the analyzers and evaluator expect
        transcript_segments_for_analysis = [
            {"start_time": seg["start_time"], "end_time": seg["end_time"], "text": seg["text"]}
            for seg in transcript_result.get("segments", [])
        ]
        
        # Choose analyzer: the trained Telesense model when a usable checkpoint is configured,
        # otherwise (or if it fails) the pretrained-model pipeline
        analysis_result = None
        engine = {"name": "pretrained", "checkpoint": None, "note": "No model checkpoint configured."}
        if checkpoint_path and os.path.exists(checkpoint_path):
            use_llm = os.getenv("USE_LLM_SEVERITY", "true").lower() == "true"
            llm_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            
            analyzer = BehavioralAnalyzer(
                checkpoint_path=checkpoint_path,
                config=config,
                use_llm_severity=use_llm,
                llm_api_key=llm_api_key
            )
            if analyzer.is_available():
                # For audio-only files, pass None for video_path
                analysis_result = analyzer.analyze(
                    video_path=video_path if not is_audio_only else None,
                    audio_path=audio_path,
                    transcript_text=transcript_result["text"],
                    fps=fps,
                    duration=duration,
                    transcript_segments=transcript_segments_for_analysis
                )
                if analysis_result.get("error"):
                    engine["note"] = f"Telesense model failed during analysis: {analysis_result['error']}"
                    analysis_result = None
                else:
                    engine = {"name": "telesense_model", "checkpoint": os.path.basename(checkpoint_path), "note": None}
            else:
                engine["note"] = f"Could not load Telesense model checkpoint: {analyzer.load_error}"
        elif checkpoint_path:
            engine["note"] = f"Model checkpoint not found: {checkpoint_path}"
        
        if analysis_result is None:
            print(f"Using pretrained models for analysis. {engine['note']}")
            free_analyzer = FreeBehavioralAnalyzer()
            
            # For audio-only files, pass None for video_path
            analysis_result = free_analyzer.analyze(
                video_path=video_path if not is_audio_only else None,
                audio_path=audio_path,
                transcript_text=transcript_result["text"],
                transcript_segments=transcript_segments_for_analysis,
                fps=fps
            )
        
        jobs[job_id].progress = 0.9
        jobs[job_id].message = "Formatting results"
        jobs[job_id].updated_at = datetime.now()
        
        # Format transcript segments
        transcript_segments = []
        for seg in transcript_result.get("segments", []):
            # Find words in this segment
            words = []
            for word in transcript_result.get("words", []):
                if seg["start_time"] <= word["start_time"] < seg["end_time"]:
                    words.append(WordTimestamp(
                        word=word["word"],
                        start_time=word["start_time"],
                        end_time=word["end_time"],
                        confidence=word.get("confidence")
                    ))
            
            # Find flags in this segment
            segment_flags = []
            for flag in analysis_result.get("combined_flags", []):
                if (seg["start_time"] <= flag["start_time"] < seg["end_time"] or
                    seg["start_time"] < flag["end_time"] <= seg["end_time"]):
                    segment_flags.append(FlagSegment(**flag))
            
            transcript_segments.append(TranscriptSegment(
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                text=seg["text"],
                words=words,
                flags=segment_flags
            ))
        
        # Convert flags to FlagSegment objects
        visual_flags = [FlagSegment(**f) for f in analysis_result.get("visual_flags", [])]
        audio_flags = [FlagSegment(**f) for f in analysis_result.get("audio_flags", [])]
        text_flags = [FlagSegment(**f) for f in analysis_result.get("text_flags", [])]
        combined_flags = [FlagSegment(**f) for f in analysis_result.get("combined_flags", [])]
        
        # Calculate statistics
        total_flags = len(combined_flags)
        flags_by_severity = {
            "high": len([f for f in combined_flags if f.severity == "high"]),
            "medium": len([f for f in combined_flags if f.severity == "medium"]),  # High Concern
            "low": len([f for f in combined_flags if f.severity == "low"])
        }
        flags_by_type = {}
        for flag in combined_flags:
            flags_by_type[flag.concern_name] = flags_by_type.get(flag.concern_name, 0) + 1
        
        # Create response
        processing_start = jobs[job_id].created_at
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        # Get audio URL for frontend
        processed_dir = get_processed_dir(video_id)
        audio_file_path = processed_dir / "audio.wav"
        audio_url = f"/api/audio/{video_id}/audio.wav" if audio_file_path.exists() else None
        
        result = {
            "job_id": job_id,
            "video_id": video_id,
            "duration": duration,
            "fps": fps,
            "num_frames": int(duration * fps),
            "file_type": "audio" if is_audio_only else "video",
            "visual_flags": [f.dict() for f in visual_flags],
            "audio_flags": [f.dict() for f in audio_flags],
            "text_flags": [f.dict() for f in text_flags],
            "combined_flags": [f.dict() for f in combined_flags],
            "transcript": [s.dict() for s in transcript_segments],
            "frame_predictions": analysis_result.get("frame_predictions", []),
            "audio_url": audio_url,  # URL for frontend to load audio
            "statistics": {
                "total_flags": total_flags,
                "flags_by_severity": flags_by_severity,
                "flags_by_type": flags_by_type
            },
            "created_at": processing_start,
            "processing_time": processing_time,
            "engine": engine,
            "evaluations": build_evaluations(
                transcript_segments_for_analysis, [f.dict() for f in combined_flags]
            ),
            "emotion_summary": build_emotion_summary(
                analysis_result.get("emotion_series"), duration, transcript_segments_for_analysis
            ),
            "metadata": {
                "files_provided": ["audio"] if is_audio_only else ["video", "audio"],
                "modalities_used": (["audio", "text"] if is_audio_only else ["video", "audio", "text"]),
                "notes": [n for n in [
                    "Analyzed with the trained Telesense model." if engine["name"] == "telesense_model"
                    else "Analyzed with pretrained emotion models (no trained Telesense model in use).",
                    engine["note"]
                ] if n]
            }
        }
        
        # Save results
        job_results[job_id] = result
        results_path = get_results_path(job_id)
        with open(results_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Update status
        jobs[job_id].status = "completed"
        jobs[job_id].progress = 1.0
        jobs[job_id].message = "Analysis completed"
        jobs[job_id].updated_at = datetime.now()
        
    except Exception as e:
        jobs[job_id].status = "failed"
        jobs[job_id].message = f"Error: {str(e)}"
        jobs[job_id].updated_at = datetime.now()
        import traceback
        traceback.print_exc()
