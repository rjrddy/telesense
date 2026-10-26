"""Multimodal behavioral models: base + temporal variants.

Two models with shared encoders:
- MultimodalBehavioralModel: single-window inference; cross-modal attention fusion.
- TemporalMultimodalBehavioralModel: sequence of windows; straight concatenation
  per window, GRU over the temporal axis.

Encoders (all pretrained; downloaded on first use):
- Audio: facebook/wav2vec2-base   (expects 16 kHz mono float32 in [-1, 1])
- Video: torchvision resnet50 / densenet121  (expects 224x224 RGB, ImageNet-normalized)
- Text : distilbert-base-uncased  (expects DistilBERT token ids + attention mask)

Output: 7 logits (the caller applies softmax).

Replace this file with your trained implementation, or keep the class shapes
and load a state_dict on top via `torch.load(...)` + `model.load_state_dict(...)`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import DistilBertModel, Wav2Vec2Model
from torchvision.models import densenet121, resnet50

NUM_CLASSES = 7
HIDDEN = 256


def _video_backbone(name: str) -> tuple[nn.Module, int]:
    if name == "resnet50":
        m = resnet50(weights=None)
        dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, dim
    if name == "densenet121":
        m = densenet121(weights=None)
        dim = m.classifier.in_features
        m.classifier = nn.Identity()
        return m, dim
    raise ValueError(f"unknown video backbone: {name}")


class _Encoders(nn.Module):
    """Shared trio of pretrained encoders + per-modality projection."""

    def __init__(self, video_backbone: str = "resnet50", hidden: int = HIDDEN):
        super().__init__()
        self.audio = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.text = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.video, video_dim = _video_backbone(video_backbone)

        self.audio_proj = nn.Linear(self.audio.config.hidden_size, hidden)
        self.video_proj = nn.Linear(video_dim, hidden)
        self.text_proj = nn.Linear(self.text.config.hidden_size, hidden)

    def encode(
        self,
        audio_values: torch.Tensor,       # (B, T_a) 16 kHz mono in [-1, 1]
        video_pixels: torch.Tensor,       # (B, 3, 224, 224)
        text_input_ids: torch.Tensor,     # (B, T_t)
        text_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self.audio(audio_values).last_hidden_state.mean(dim=1)
        v = self.video(video_pixels)
        t = self.text(input_ids=text_input_ids, attention_mask=text_attention_mask).last_hidden_state[:, 0]
        return self.audio_proj(a), self.video_proj(v), self.text_proj(t)


class MultimodalBehavioralModel(nn.Module):
    """Base model: cross-modal attention fusion, single window."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden: int = HIDDEN,
        video_backbone: str = "resnet50",
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoders = _Encoders(video_backbone=video_backbone, hidden=hidden)

        # Cross-modal attention: each modality attends over the stack of the other two.
        self.attn_audio = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.attn_video = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.attn_text = nn.MultiheadAttention(hidden, num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(
        self,
        audio_values: torch.Tensor,
        video_pixels: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        a, v, t = self.encoders.encode(audio_values, video_pixels, text_input_ids, text_attention_mask)
        a1 = a.unsqueeze(1); v1 = v.unsqueeze(1); t1 = t.unsqueeze(1)  # (B, 1, H)

        vt = torch.cat([v1, t1], dim=1)  # (B, 2, H) — video+text KV for audio query
        at = torch.cat([a1, t1], dim=1)
        av = torch.cat([a1, v1], dim=1)

        a_out, _ = self.attn_audio(a1, vt, vt)
        v_out, _ = self.attn_video(v1, at, at)
        t_out, _ = self.attn_text(t1, av, av)

        fused = torch.cat([a_out.squeeze(1), v_out.squeeze(1), t_out.squeeze(1)], dim=-1)
        return self.classifier(fused)  # (B, num_classes)


class TemporalMultimodalBehavioralModel(nn.Module):
    """Temporal variant: T windows in, concat-fuse per window, GRU over T."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden: int = HIDDEN,
        video_backbone: str = "resnet50",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoders = _Encoders(video_backbone=video_backbone, hidden=hidden)
        self.temporal = nn.GRU(hidden * 3, hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(
        self,
        audio_windows: torch.Tensor,      # (B, T, T_a)
        video_windows: torch.Tensor,      # (B, T, 3, 224, 224)
        text_input_ids: torch.Tensor,     # (B, T, T_t) — same text repeated per window, or per-window transcript
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T = audio_windows.shape[:2]
        a_flat = audio_windows.reshape(B * T, -1)
        v_flat = video_windows.reshape(B * T, *video_windows.shape[-3:])
        ti_flat = text_input_ids.reshape(B * T, -1)
        tm_flat = text_attention_mask.reshape(B * T, -1)

        a, v, t = self.encoders.encode(a_flat, v_flat, ti_flat, tm_flat)
        fused = torch.cat([a, v, t], dim=-1).reshape(B, T, -1)  # concat fusion per window
        seq, _ = self.temporal(fused)                            # (B, T, H)
        return self.classifier(seq[:, -1])                       # (B, num_classes)


# 7-class label set — replace with your training labels.
DEFAULT_LABELS = [
    "neutral",
    "anxiety",
    "depression",
    "stress",
    "elevated_mood",
    "disorganized",
    "distress",
]
