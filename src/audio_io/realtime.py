"""Realtime audio callback contracts for audio IO adapters."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class RealtimeAudioCallback(Protocol):
    """Generate audio frames for a realtime callback.

    Implementations are called from the audio backend callback thread. For stable
    low-latency playback, callbacks should avoid blocking IO, logging, acquiring
    contended locks, and avoidable allocations.
    """

    def __call__(self, num_frames: int) -> np.ndarray | None:
        """Return mono or stereo audio for exactly `num_frames` requested frames."""
