"""Widgets package for modular synth interface."""

from .knob_widget import Knob
from .slider_widget import VSlider, HSlider
from .waveform_display import WaveformDisplay
from .spectrum_analyzer import SpectrumAnalyzer

__all__ = [
    "Knob",
    "VSlider",
    "HSlider",
    "WaveformDisplay",
    "SpectrumAnalyzer",
]

