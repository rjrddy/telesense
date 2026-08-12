"""Consumes remote WebRTC tracks and pushes the raw media to the model.

Each incoming track spawns a long-running asyncio task that pulls frames off
the track and hands them to the model. Also republishes the track to a
MediaRelay so multiple subscribers (other peers in the room) can share it
without pulling twice.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaRelay

from model_interface import ModelInterface

log = logging.getLogger("pipeline")

relay = MediaRelay()


async def consume_audio(peer_id: str, track: MediaStreamTrack, model: ModelInterface) -> None:
    log.info("peer=%s audio consumer started", peer_id)
    try:
        while True:
            frame = await track.recv()
            # aiortc AudioFrame -> ndarray. Layout is (channels, samples) for
            # planar formats or (1, samples*channels) for packed. We take a
            # mono float32 view for the model.
            arr = frame.to_ndarray()
            if arr.ndim == 2 and arr.shape[0] > 1:
                arr = arr.mean(axis=0)
            arr = arr.astype(np.float32).reshape(-1)
            # Normalize s16 -> [-1, 1] if needed
            if arr.dtype != np.float32 or np.max(np.abs(arr)) > 1.5:
                arr = arr.astype(np.float32) / 32768.0
            await model.feed_audio(peer_id, arr, frame.sample_rate)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("peer=%s audio consumer crashed", peer_id)


async def consume_video(peer_id: str, track: MediaStreamTrack, model: ModelInterface) -> None:
    log.info("peer=%s video consumer started", peer_id)
    try:
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            await model.feed_video(peer_id, img)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("peer=%s video consumer crashed", peer_id)


def subscribe(track: MediaStreamTrack) -> MediaStreamTrack:
    """Create a fresh subscriber track that can be added to another PC."""
    return relay.subscribe(track)
