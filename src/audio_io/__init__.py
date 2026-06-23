"""Optional audio-device output support for AudioPlayground.

Install with the `audio-io` extra to use the sounddevice-backed output stream.
"""

from src.audio_io.output import (
    DEFAULT_FADEIN_DURATION_MS,
    DEFAULT_FADEOUT_DURATION_MS,
    AudioOutput,
)
from src.audio_io.realtime import RealtimeAudioCallback

__all__ = [
    "AudioOutput",
    "DEFAULT_FADEIN_DURATION_MS",
    "DEFAULT_FADEOUT_DURATION_MS",
    "RealtimeAudioCallback",
]
