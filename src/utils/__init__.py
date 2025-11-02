"""Audio utilities package.

This package provides utility functions for audio processing, file I/O,
and logging configuration.

Main modules:
    - utils: Audio file I/O, playback, and conversions
    - logging_config: Centralized logging configuration

Quick imports:
    >>> from utils import save_wave, load_wave, play_wave, note_to_frequency
    >>> from utils import get_logger, setup_logging
"""

# Audio utilities
from .utils import (
    # Primary API (recommended)
    to_int16,
    save_wave,
    load_wave,
    play_wave,
    note_to_frequency,
    # Legacy API (for backward compatibility)
    to_int16,
    save_wave,
    load_wave,
    note_to_frequency,
)

# Logging utilities
from .logging_config import (
    setup_logging,
    get_logger,
    get_engine_logger,
    DEFAULT_LOG_LEVEL,
)

__all__ = [
    # Audio utilities - primary API
    "to_int16",
    "save_wave",
    "load_wave",
    "play_wave",
    "note_to_frequency",
    # Audio utilities - legacy API
    "to_int16",
    "save_wave",
    "load_wave",
    "note_to_frequency",
    # Logging
    "setup_logging",
    "get_logger",
    "get_engine_logger",
    "DEFAULT_LOG_LEVEL",
]

