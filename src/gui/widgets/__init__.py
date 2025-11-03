"""Widgets package for modular synth interface."""

from src.gui.widgets.knob_widget import Knob
from src.gui.widgets.slider_widget import VSlider, HSlider
from src.gui.widgets.waveform_display import WaveformDisplay
from src.gui.widgets.spectrum_analyzer import SpectrumAnalyzer

__all__ = [
    "Knob",
    "VSlider",
    "HSlider",
    "WaveformDisplay",
    "SpectrumAnalyzer",
]
