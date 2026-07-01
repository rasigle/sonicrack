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
)
from src.engine.dsp.modifiers.base import Modifier
from src.engine.dsp.modifiers.frequency import Frequency
from src.engine.dsp.modifiers.panning import ModulatedPanner, Panner

__all__ = [
    "Modifier",
    "Panner",
    "ModulatedPanner",
    "Volume",
    "ModulatedVolume",
    "Frequency",
    "Clipper",
    "ModulatedClipper",
]
