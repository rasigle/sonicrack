"""Modular synthesizer GUI package.

This package provides a full-featured graphical interface for the AudioPlayground
synthesis engine, featuring:
- Modular patching system (drag-and-drop cable connections)
- Real-time audio playback
- Visual feedback (waveforms, spectrum analysis)
- Preset management
- MIDI support (future)
"""

from src.gui.main_window import ModularSynthWindow

__all__ = ["ModularSynthWindow"]

