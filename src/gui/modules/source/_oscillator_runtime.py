"""Shared runtime helpers for GUI oscillator source modules."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class RuntimeOscillator(Protocol):
    """Minimal oscillator API needed for runtime frequency ramping."""

    frequency: float

    def get_samples(self, n: int) -> np.ndarray: ...

    def __next__(self) -> float: ...


def render_with_frequency_ramp(
    oscillator: RuntimeOscillator,
    previous_frequency: float,
    target_frequency: float,
    num_samples: int,
) -> np.ndarray:
    """Render one buffer while smoothing frequency changes across the buffer."""
    if previous_frequency == target_frequency:
        oscillator.frequency = target_frequency
        return oscillator.get_samples(num_samples)

    samples = []
    for frequency in np.linspace(
        previous_frequency, target_frequency, num_samples, dtype=np.float64
    ):
        oscillator.frequency = float(frequency)
        samples.append(next(oscillator))
    oscillator.frequency = target_frequency
    return np.asarray(samples, dtype=np.float32)
