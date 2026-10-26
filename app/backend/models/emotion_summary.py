"""Emotion timeline built from the per-modality emotion detectors.

Merges audio, text and video emotion scores onto a shared time grid with a common
label set, producing the event points and windows the Emotion Timeline panel plots.
"""

from typing import Dict, List, Optional

import numpy as np

# Detector labels -> common label set
LABEL_MAP = {
    # audio (superb-er) and video (DeepFace/FER)
    "neu": "neutral", "neutral": "neutral",
    "hap": "joy", "happy": "joy", "happiness": "joy",
    "sad": "sadness", "sadness": "sadness",
    "ang": "anger", "angry": "anger", "anger": "anger",
    "fear": "fear", "fearful": "fear",
    "surprise": "surprise", "surprised": "surprise",
    "disgust": "disgust", "disgusted": "disgust",
    # text (go_emotions)
    "joy": "joy", "amusement": "joy", "excitement": "joy", "gratitude": "joy", "love": "joy",
    "optimism": "joy", "pride": "joy", "relief": "joy", "admiration": "joy", "approval": "joy",
    "caring": "joy",
    "grief": "sadness", "disappointment": "sadness", "remorse": "sadness",
    "annoyance": "anger", "disapproval": "anger",
    "nervousness": "fear",
    "realization": "surprise", "confusion": "surprise", "curiosity": "surprise",
    "embarrassment": "disgust",
}
EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"]


def _timestamp(seconds: float) -> str:
    """
    Format seconds as MM:SS.
    Input:
        seconds - time in seconds

    Output:
        Time string in MM:SS format
    """
    minutes, secs = divmod(int(max(seconds, 0)), 60)
    return f"{minutes:02d}:{secs:02d}"


def _normalize(emotions: Dict[str, float]) -> Dict[str, float]:
    """
    Map detector-specific labels onto the common label set.
    Input:
        emotions - {label: score} from one detector

    Output:
        {common_label: score} summed per common label
    """
    out = {e: 0.0 for e in EMOTIONS}
    for label, score in (emotions or {}).items():
        common = LABEL_MAP.get(str(label).lower())
        if common:
            out[common] += float(score)
    return out


def _span(entry: Dict) -> tuple:
    """
    Get the time span of a detector result.
    Input:
        entry - detector result with start_time/end_time or a single timestamp

    Output:
        Tuple (start_s, end_s)
    """
    start = entry.get("start_time", entry.get("timestamp", 0.0)) or 0.0
    end = entry.get("end_time", start + 1.0) or start + 1.0
    return float(start), float(end)


def build_emotion_summary(emotion_series: Optional[Dict[str, List[Dict]]], duration: float,
                          transcript: Optional[List[Dict]] = None, step: float = 5.0,
                          window: float = 60.0) -> Optional[Dict]:
    """
    Merge per-modality emotion results into the Emotion Timeline data.
    Input:
        emotion_series - {"audio" | "text" | "video": [detector results]} from the analyzer
        duration - session length in seconds
        transcript - transcript segments, used for the text shown at each point (optional)
        step - spacing of event points in seconds
        window - length of summary windows in seconds

    Output:
        Dictionary with assumptions, event_points and windows, or None if there are no
        emotion results
    """
    series = {k: v for k, v in (emotion_series or {}).items() if v}
    if not series or duration <= 0:
        return None

    grid = np.arange(0.0, duration, step)
    points = []
    for t in grid:
        totals, count = {e: 0.0 for e in EMOTIONS}, 0
        for results in series.values():
            for entry in results:
                start, end = _span(entry)
                if start < t + step and end > t:
                    for e, v in _normalize(entry.get("emotions")).items():
                        totals[e] += v
                    count += 1
        if count == 0:
            continue
        emotions = {e: round(v / count, 3) for e, v in totals.items()}
        top = sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)[:2]
        text = next((s.get("text", "") for s in (transcript or [])
                     if s.get("start_time", 0) <= t < s.get("end_time", 0)), "")
        points.append({
            "time_s": float(t),
            "timestamp": _timestamp(t),
            "emotions": emotions,
            "top_emotions": [[e, v] for e, v in top],
            "evidence": {"text": text},
        })
    if not points:
        return None

    windows = []
    for w_start in np.arange(0.0, duration, window):
        w_end = min(w_start + window, duration)
        in_window = [p for p in points if w_start <= p["time_s"] < w_end]
        if not in_window:
            continue
        mean = {e: float(np.mean([p["emotions"][e] for p in in_window])) for e in EMOTIONS}
        top = sorted(mean.items(), key=lambda kv: kv[1], reverse=True)[:2]
        windows.append({
            "start_timestamp": _timestamp(w_start),
            "end_timestamp": _timestamp(w_end),
            "dominant_emotions": [[e, round(v, 3)] for e, v in top],
            "notes": "",
        })

    return {
        "assumptions": [
            f"Scores averaged across {', '.join(sorted(series))} detectors on a {step:g}s grid.",
            "Detector labels mapped onto a shared set of 7 emotions.",
        ],
        "event_points": points,
        "windows": windows,
    }
