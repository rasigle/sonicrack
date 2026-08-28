"""Audio utilities package.

This package provides utility functions for audio processing, file I/O,
and logging configuration.

Main modules:
    - utils: Audio file I/O, playback, and conversions
    - logging_config: Centralized logging configuration

Quick imports:
    >>> from sonicrack.utils import save_wave, load_wave, play_wave, note_to_frequency
"""

# Audio utilities
from sonicrack.utils.audio_utils import (
    load_wave,
    note_to_frequency,
    play_wave,
    save_wave,
    # Primary API (recommended)
    to_int16,
)

# Logging utilities
from sonicrack.utils.logging_config import DEFAULT_LOG_LEVEL

__all__ = [
    "to_int16",
    "save_wave",
    "load_wave",
    "play_wave",
    "note_to_frequency",
    "DEFAULT_LOG_LEVEL",
]
