"""Shared MIDI note-on trigger pulse helper."""

from __future__ import annotations

from typing import Any

import numpy as np


class MIDITriggerOutput:
    """Emit a single-sample trigger pulse for each armed note-on.

    Eurorack-style MIDI-to-CV modules expose both Gate (held high while a note
    is active) and Trig (a short pulse on every note-on). This helper arms a
    pulse from the UI/MIDI thread and consumes it once during the next audio
    render cycle, including legato note changes while the gate stays high.
    """

    def __init__(self) -> None:
        self._pending = False

    def arm(self) -> None:
        """Schedule a trigger pulse for the next ``get_samples`` call."""
        self._pending = True

    def reset(self) -> None:
        """Clear any pending pulse without emitting it."""
        self._pending = False

    @property
    def pending(self) -> bool:
        return self._pending

    def get_samples(self, n: int, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return a buffer with a one-sample pulse at index 0 when armed."""
        del args, kwargs
        samples = np.zeros(n, dtype=np.float32)
        if self._pending and n > 0:
            samples[0] = 1.0
            self._pending = False
        return samples
