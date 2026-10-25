"""Single seam between the SFU and the multimodal behavioral model.

Per-peer sliding-window inference:
- audio ring buffer at 16 kHz mono float32  (resampled from aiortc's native rate)
- latest N video frames, cached as (224, 224, 3) uint8 RGB
- rolling text buffer of recent chat/transcript messages

Every INFER_EVERY_S seconds, if a peer has at least a MIN_AUDIO_S window of
audio and one video frame, we assemble a batch-of-1 and run
`MultimodalBehavioralModel.forward(...)` on a thread executor (to keep the
asyncio loop free). The 7-class softmax is emitted via `on_score`.

Env vars:
- MODEL_VARIANT     : "base" (default) | "temporal"
- MODEL_CHECKPOINT  : optional path to a torch state_dict for the fusion+head
- MODEL_DEVICE      : "cpu" (default) | "cuda" | "mps"
- MODEL_VIDEO_BACKBONE : "resnet50" (default) | "densenet121"
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Deque

import numpy as np

log = logging.getLogger("model")

# --- Windowing / cadence ---
TARGET_SR = 16_000
WINDOW_S = 2.0            # audio window fed to Wav2Vec2 per inference
MIN_AUDIO_S = 1.0         # don't infer until we have this much audio
INFER_EVERY_S = 2.0       # sliding-window cadence per peer
TEXT_KEEP_CHARS = 512     # rolling text window
IMAGE_SIZE = 224


@dataclass
class PeerScore:
    peer_id: str
    logits: list[float] = field(default_factory=list)          # raw model output
    probs: list[float] = field(default_factory=list)           # softmax
    top_label: str = ""
    top_confidence: float = 0.0
    last_update: float = field(default_factory=time.time)


@dataclass
class _PeerBuffer:
    peer_id: str
    audio: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=64))  # chunks of 16k float32
    audio_samples: int = 0
    latest_frame_rgb: np.ndarray | None = None
    text_buf: str = ""
    last_infer: float = 0.0
    infer_running: bool = False


class ModelInterface:
    """Buffers per-peer media, runs sliding-window inference, emits PeerScore.

    Model loading is deferred until first inference so the SFU can boot without
    paying the ~1 GB HF download upfront.
    """

    def __init__(self, on_score: Callable[[PeerScore], None] | None = None):
        self._on_score = on_score
        self._buffers: dict[str, _PeerBuffer] = {}
        self._scores: dict[str, PeerScore] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mm-infer")
        self._model = None                # torch nn.Module, lazy
        self._tokenizer = None            # DistilBertTokenizerFast, lazy
        self._audio_extractor = None      # Wav2Vec2FeatureExtractor, lazy
        self._image_transform = None      # torchvision transform, lazy
        self._resampler = None            # torchaudio resampler, lazy
        self._src_sr = None               # updated when first audio arrives
        self._labels = None
        self._device = os.environ.get("MODEL_DEVICE", "cpu")
        self._variant = os.environ.get("MODEL_VARIANT", "base")
        self._video_backbone = os.environ.get("MODEL_VIDEO_BACKBONE", "resnet50")
        self._checkpoint = os.environ.get("MODEL_CHECKPOINT")

        # Where to import your model class from. Change these if your file
        # isn't `behavioral_model.py` or your classes have different names.
        self._module = os.environ.get("MODEL_MODULE", "behavioral_model")
        self._base_cls = os.environ.get("MODEL_BASE_CLASS", "MultimodalBehavioralModel")
        self._temporal_cls = os.environ.get("MODEL_TEMPORAL_CLASS", "TemporalMultimodalBehavioralModel")
        self._labels_attr = os.environ.get("MODEL_LABELS_ATTR", "DEFAULT_LABELS")
        # Set to "0" if your ctor doesn't take a video_backbone kwarg.
        self._pass_backbone_kwarg = os.environ.get("MODEL_PASS_BACKBONE", "1") == "1"

        self._load_lock = asyncio.Lock()

    # ------------------------------------------------------------------ ingest

    def _buf(self, peer_id: str) -> _PeerBuffer:
        b = self._buffers.get(peer_id)
        if b is None:
            b = _PeerBuffer(peer_id=peer_id)
            self._buffers[peer_id] = b
        return b

    async def feed_audio(self, peer_id: str, pcm: np.ndarray, sample_rate: int) -> None:
        b = self._buf(peer_id)
        # aiortc may hand us 48 kHz; resample to 16 kHz for Wav2Vec2.
        if sample_rate != TARGET_SR:
            pcm = self._resample_np(pcm, sample_rate, TARGET_SR)
        b.audio.append(pcm.astype(np.float32, copy=False))
        b.audio_samples += pcm.shape[0]
        await self._maybe_infer(peer_id)

    async def feed_video(self, peer_id: str, frame_bgr: np.ndarray) -> None:
        # aiortc gives BGR; convert to RGB and downsize to 224x224 for the CNN.
        rgb = frame_bgr[:, :, ::-1]
        h, w = rgb.shape[:2]
        s = min(h, w)
        top = (h - s) // 2
        left = (w - s) // 2
        crop = rgb[top:top + s, left:left + s]
        # cheap resize via numpy stride slicing then handoff to torchvision at infer time
        self._buf(peer_id).latest_frame_rgb = np.ascontiguousarray(crop)
        await self._maybe_infer(peer_id)

    async def feed_text(self, peer_id: str, text: str, source: str = "chat") -> None:
        b = self._buf(peer_id)
        b.text_buf = (b.text_buf + " " + text).strip()[-TEXT_KEEP_CHARS:]
        log.info("peer=%s text (%s): %r", peer_id, source, text[:120])
        await self._maybe_infer(peer_id)

    def get_score(self, peer_id: str) -> PeerScore | None:
        return self._scores.get(peer_id)

    # ---------------------------------------------------------------- inference

    async def _maybe_infer(self, peer_id: str) -> None:
        b = self._buf(peer_id)
        if b.infer_running:
            return
        now = time.time()
        if now - b.last_infer < INFER_EVERY_S:
            return
        if b.audio_samples < int(MIN_AUDIO_S * TARGET_SR):
            return
        if b.latest_frame_rgb is None:
            return
        b.infer_running = True
        b.last_infer = now
        asyncio.create_task(self._infer(peer_id))

    async def _infer(self, peer_id: str) -> None:
        b = self._buf(peer_id)
        try:
            await self._ensure_loaded()
            audio_win = self._take_audio_window(b)
            frame = b.latest_frame_rgb
            text = b.text_buf or ""

            loop = asyncio.get_running_loop()
            logits = await loop.run_in_executor(self._executor, self._forward, audio_win, frame, text)

            probs = self._softmax(logits)
            top = int(np.argmax(probs))
            score = self._scores.setdefault(peer_id, PeerScore(peer_id=peer_id))
            score.logits = logits.tolist()
            score.probs = probs.tolist()
            score.top_label = self._labels[top] if self._labels else str(top)
            score.top_confidence = float(probs[top])
            score.last_update = time.time()
            log.info(
                "peer=%s score top=%s conf=%.2f probs=%s",
                peer_id, score.top_label, score.top_confidence,
                [f"{p:.2f}" for p in score.probs],
            )
            if self._on_score:
                try:
                    self._on_score(score)
                except Exception:
                    log.exception("on_score callback failed")
        except Exception:
            log.exception("peer=%s inference failed", peer_id)
        finally:
            b.infer_running = False

    def _take_audio_window(self, b: _PeerBuffer) -> np.ndarray:
        needed = int(WINDOW_S * TARGET_SR)
        buf = np.concatenate(list(b.audio)) if b.audio else np.zeros(0, dtype=np.float32)
        if buf.shape[0] > needed:
            buf = buf[-needed:]
        # Keep the tail in the deque so audio keeps flowing; simplest: reset to a
        # single-chunk tail equal to the window we just grabbed.
        b.audio.clear()
        b.audio.append(buf)
        b.audio_samples = buf.shape[0]
        return buf

    # ------------------------------------------------------------- model + prep

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            log.info("loading behavioral model (variant=%s, backbone=%s, device=%s)...",
                     self._variant, self._video_backbone, self._device)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._load_blocking)
            log.info("behavioral model ready")

    def _load_blocking(self) -> None:
        import torch
        import importlib
        from torchvision import transforms
        from transformers import DistilBertTokenizerFast, Wav2Vec2FeatureExtractor

        mod = importlib.import_module(self._module)
        cls_name = self._base_cls if self._variant == "base" else self._temporal_cls
        cls = getattr(mod, cls_name)
        kwargs = {"video_backbone": self._video_backbone} if self._pass_backbone_kwarg else {}
        try:
            model = cls(**kwargs)
        except TypeError:
            # Ctor doesn't accept our kwargs — try no-arg.
            log.warning("%s(**%s) failed; retrying with no kwargs", cls_name, kwargs)
            model = cls()

        if self._checkpoint and os.path.isfile(self._checkpoint):
            sd = torch.load(self._checkpoint, map_location="cpu")
            missing, unexpected = model.load_state_dict(sd, strict=False)
            log.info("loaded checkpoint %s (missing=%d, unexpected=%d)",
                     self._checkpoint, len(missing), len(unexpected))
        else:
            log.warning("no checkpoint at MODEL_CHECKPOINT=%r; running with untrained fusion+head",
                        self._checkpoint)
        model.eval().to(self._device)
        self._model = model

        self._tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self._audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        self._image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        labels = getattr(mod, self._labels_attr, None)
        if labels is None or len(labels) != 7:
            log.warning("no %s attribute (or wrong length) in %s; using numeric labels",
                        self._labels_attr, self._module)
            labels = [str(i) for i in range(7)]
        self._labels = list(labels)

    def _forward(self, audio_win: np.ndarray, frame_rgb: np.ndarray, text: str) -> np.ndarray:
        import torch

        assert self._model and self._tokenizer and self._audio_extractor and self._image_transform

        a = self._audio_extractor(audio_win, sampling_rate=TARGET_SR, return_tensors="pt")
        audio_values = a["input_values"].to(self._device)                         # (1, T_a)

        img = self._image_transform(frame_rgb).unsqueeze(0).to(self._device)      # (1, 3, 224, 224)

        tok = self._tokenizer(
            text if text else "[unk]",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        input_ids = tok["input_ids"].to(self._device)
        attn_mask = tok["attention_mask"].to(self._device)

        with torch.inference_mode():
            if self._variant == "temporal":
                # Fake a 1-window temporal batch until we actually buffer T windows.
                audio_values = audio_values.unsqueeze(1)                          # (1, 1, T_a)
                img = img.unsqueeze(1)                                            # (1, 1, 3, 224, 224)
                input_ids = input_ids.unsqueeze(1)                                # (1, 1, T_t)
                attn_mask = attn_mask.unsqueeze(1)
            logits = self._model(audio_values, img, input_ids, attn_mask)
        return logits.squeeze(0).float().cpu().numpy()

    # ------------------------------------------------------------------- utils

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _resample_np(self, pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Linear-interp resample (dependency-free). Wav2Vec2 tolerates this fine
        for short windows; swap for torchaudio.functional.resample if you want
        higher fidelity."""
        if src_sr == dst_sr or pcm.size == 0:
            return pcm
        n_out = int(round(pcm.shape[0] * dst_sr / src_sr))
        if n_out <= 1:
            return pcm[:n_out].astype(np.float32)
        x_in = np.linspace(0.0, 1.0, num=pcm.shape[0], endpoint=False)
        x_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_out, x_in, pcm).astype(np.float32)


# Module-level singleton so the SFU has a stable import.
_default: ModelInterface | None = None


def get_model(on_score: Callable[[PeerScore], None] | None = None) -> ModelInterface:
    global _default
    if _default is None:
        _default = ModelInterface(on_score=on_score)
    elif on_score is not None:
        _default._on_score = on_score
    return _default
