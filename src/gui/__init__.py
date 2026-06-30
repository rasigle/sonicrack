"""Modular synthesizer GUI package.

This package provides a full-featured graphical interface for the AudioPlayground
synthesis engine, featuring:
- Modular patching system (drag-and-drop cable connections)
- Real-time audio playback
- Visual feedback (waveforms, spectrum analysis)
- Preset management
- MIDI support (future)
"""

__all__ = ["ModularSynthWindow"]


def __getattr__(name: str):
    """Lazily import GUI window classes so submodule imports stay lightweight."""
    if name == "ModularSynthWindow":
        from src.gui.main_window import ModularSynthWindow

        return ModularSynthWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
