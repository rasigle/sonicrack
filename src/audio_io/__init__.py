"""Optional audio-device output support for AudioPlayground.

Install with the `audio-io` extra to use the sounddevice-backed output stream:

    uv sync --extra audio-io

The main entry point is `AudioOutput`. Provide a callback that receives the
number of frames requested by the audio backend and returns either mono samples
with shape `(frames,)` or stereo samples with shape `(frames, 2)`.

Example:
    >>> import numpy as np
    >>> from src.audio_io import AudioOutput
    >>>
    >>> phase = 0.0
    >>> sample_rate = 44100
    >>>
    >>> def sine_callback(frames: int) -> np.ndarray:
    ...     global phase
    ...     t = (np.arange(frames) + phase) / sample_rate
    ...     samples = 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    ...     phase += frames
    ...     return samples.astype(np.float32)
    >>>
    >>> output = AudioOutput(
    ...     sample_rate=sample_rate,
    ...     buffer_size=512,
    ...     audio_callback=sine_callback,
    ... )
    >>> output.start_playback()
    >>> # Later, stop and release the device:
    >>> output.cleanup()
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
