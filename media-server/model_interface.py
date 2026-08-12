"""Single seam between the SFU and the mental-health ML model.

The real model is added later. For now every hook logs shape/size and returns
a stub certainty score. Swap the bodies (or the whole class) — the SFU only
touches this file.

Design contract:
- feed_audio(peer_id, pcm, sample_rate): raw mono float32 PCM chunks. Buffer
  internally until you have enough for Whisper (~1s+), transcribe, then call
  feed_text with the transcript.
- feed_video(peer_id, frame_bgr): numpy HxWx3 uint8 BGR frame. Run visual
  cues (facial affect, gaze, etc.) here.
- feed_text(peer_id, text, source): 'chat' or 'transcript'. Word2vec/embedding
  pipeline goes here.

Every hook is called from an asyncio task and MUST NOT block the event loop.
Heavy work belongs on a thread/process pool.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

log = logging.getLogger("model")


@dataclass
class PeerScore:
    peer_id: str
    certainty: float = 0.0
    last_update: float = field(default_factory=time.time)
    notes: dict = field(default_factory=dict)


class ModelInterface:
    """Stub. Aggregates cues per peer and produces a certainty score."""

    def __init__(self, on_score: Callable[[PeerScore], None] | None = None):
        self._scores: dict[str, PeerScore] = {}
        self._on_score = on_score
        # Placeholders for future components:
        self._whisper = None       # load whisper model here (lazy)
        self._word2vec = None      # load gensim KeyedVectors here (lazy)
        self._vision_model = None  # load vision/affect model here (lazy)

    def _score(self, peer_id: str) -> PeerScore:
        if peer_id not in self._scores:
            self._scores[peer_id] = PeerScore(peer_id=peer_id)
        return self._scores[peer_id]

    async def feed_audio(self, peer_id: str, pcm: np.ndarray, sample_rate: int) -> None:
        # TODO: buffer -> whisper.transcribe -> self.feed_text(..., source="transcript")
        score = self._score(peer_id)
        score.notes["audio_samples"] = score.notes.get("audio_samples", 0) + len(pcm)
        score.last_update = time.time()
        if score.notes["audio_samples"] % (sample_rate * 5) < len(pcm):
            log.info("peer=%s audio ~%.1fs buffered", peer_id, score.notes["audio_samples"] / sample_rate)

    async def feed_video(self, peer_id: str, frame_bgr: np.ndarray) -> None:
        # TODO: vision model inference (facial affect / gaze / motion cues)
        score = self._score(peer_id)
        score.notes["video_frames"] = score.notes.get("video_frames", 0) + 1
        score.last_update = time.time()
        if score.notes["video_frames"] % 150 == 0:
            h, w, _ = frame_bgr.shape
            log.info("peer=%s video frame #%d %dx%d", peer_id, score.notes["video_frames"], w, h)

    async def feed_text(self, peer_id: str, text: str, source: str = "chat") -> None:
        # TODO: word2vec/embedding + classifier
        score = self._score(peer_id)
        score.notes.setdefault("text_msgs", 0)
        score.notes["text_msgs"] += 1
        score.last_update = time.time()
        log.info("peer=%s text (%s): %r", peer_id, source, text[:120])
        # Stub scoring: length-based, deterministic, obviously fake.
        score.certainty = min(1.0, 0.05 * score.notes["text_msgs"])
        if self._on_score:
            self._on_score(score)

    def get_score(self, peer_id: str) -> PeerScore | None:
        return self._scores.get(peer_id)


# Convenience singletons so the SFU has a stable import.
_default: ModelInterface | None = None


def get_model(on_score: Callable[[PeerScore], None] | None = None) -> ModelInterface:
    global _default
    if _default is None:
        _default = ModelInterface(on_score=on_score)
    elif on_score is not None:
        _default._on_score = on_score
    return _default


async def _demo() -> None:
    m = get_model()
    await m.feed_text("p1", "hello world")
    print(m.get_score("p1"))


if __name__ == "__main__":
    asyncio.run(_demo())
