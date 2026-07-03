"""Signal modifiers for audio processing.

This package provides components that modify audio signals through callable objects.
Modifiers can be chained together using the Chain composer to create complex
signal processing pipelines.
"""

from src.engine.dsp.modifiers.amplitude import (
    Clipper,
    ModulatedClipper,
    ModulatedVolume,
    Volume,
    apply_vectorized_clip,
    apply_vectorized_gain,
)
from src.engine.dsp.modifiers.base import Modifier
from src.engine.dsp.modifiers.frequency import Frequency
from src.engine.dsp.modifiers.panning import (
    ModulatedPanner,
    Panner,
    apply_vectorized_panning,
)

__all__ = [
    "Modifier",
    "Panner",
    "ModulatedPanner",
    "apply_vectorized_panning",
    "Volume",
    "ModulatedVolume",
    "apply_vectorized_gain",
    "Frequency",
    "Clipper",
    "ModulatedClipper",
    "apply_vectorized_clip",
]
