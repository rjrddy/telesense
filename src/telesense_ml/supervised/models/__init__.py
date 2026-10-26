"""Supervised model components: fusion, temporal encoders and localization heads."""

from .fusion import CrossModalAttention
from .multimodal_model import MultimodalBehavioralModel
from .temporal_encoders import (
    ShortTermEncoder,
    TemporalConvolutionalNetwork,
    LongTermTransformerEncoder,
    MultiScaleTemporalEncoder
)
from .temporal_localization import (
    TemporalAttention,
    TemporalLocalizationHead,
    TemporalSmoothing,
    TemporalLocalizationModel
)
from .temporal_behavioral_model import TemporalMultimodalBehavioralModel

__all__ = [
    "CrossModalAttention",
    "MultimodalBehavioralModel",
    "ShortTermEncoder",
    "TemporalConvolutionalNetwork",
    "LongTermTransformerEncoder",
    "MultiScaleTemporalEncoder",
    "TemporalAttention",
    "TemporalLocalizationHead",
    "TemporalSmoothing",
    "TemporalLocalizationModel",
    "TemporalMultimodalBehavioralModel",
]

