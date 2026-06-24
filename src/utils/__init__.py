"""Audio utilities package.

This package provides utility functions for audio processing, file I/O,
and logging configuration.

Main modules:
    - utils: Audio file I/O, playback, and conversions
    - logging_config: Centralized logging configuration

Quick imports:
    >>> from src.utils import save_wave, load_wave, play_wave, note_to_frequency
"""

# Audio utilities
from src.utils.audio_utils import (
    load_wave,
    note_to_frequency,
    play_wave,
    save_wave,
    # Primary API (recommended)
    to_int16,
)

# Logging utilities
from src.utils.logging_config import DEFAULT_LOG_LEVEL

__all__ = [
    # Audio utilities - primary API
    "to_int16",
    "save_wave",
    "load_wave",
    "play_wave",
    "note_to_frequency",
    # Audio utilities
    "to_int16",
    "save_wave",
    "load_wave",
    "note_to_frequency",
    # Logging
    "DEFAULT_LOG_LEVEL",
]
