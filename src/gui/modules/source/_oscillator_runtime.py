"""Shared runtime helpers for GUI oscillator source modules."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class RuntimeOscillator(Protocol):
    """Minimal oscillator API needed for runtime frequency ramping."""

    frequency: float

    def get_samples(self, n: int) -> np.ndarray: ...

    def __next__(self) -> float: ...


DEFAULT_FREQUENCY_SLEW_TIME_MS = 35.0


def _frequency_slew_values(
    previous_frequency: float,
    target_frequency: float,
    num_samples: int,
    sample_rate: float,
    slew_time_ms: float,
) -> np.ndarray:
    """Build a smooth per-sample frequency curve in pitch space."""
    if num_samples <= 0:
        return np.empty(0, dtype=np.float64)
    if previous_frequency == target_frequency or slew_time_ms <= 0.0:
        return np.full(num_samples, target_frequency, dtype=np.float64)

    slew_samples = max(sample_rate * slew_time_ms / 1000.0, 1.0)
    alpha = 1.0 - math.exp(-1.0 / slew_samples)
    frequencies = np.empty(num_samples, dtype=np.float64)

    if previous_frequency > 0.0 and target_frequency > 0.0:
        current = math.log(previous_frequency)
        target = math.log(target_frequency)
        for index in range(num_samples):
            current += (target - current) * alpha
            frequencies[index] = math.exp(current)
    else:
        current = previous_frequency
        for index in range(num_samples):
            current += (target_frequency - current) * alpha
            frequencies[index] = current

    if abs(frequencies[-1] - target_frequency) < max(
        1e-6, abs(target_frequency) * 1e-6
    ):
        frequencies[-1] = target_frequency

    return frequencies


def render_with_frequency_ramp(
    oscillator: RuntimeOscillator,
    previous_frequency: float,
    target_frequency: float,
    num_samples: int,
) -> tuple[np.ndarray, float]:
    """Render one buffer while slewing frequency smoothly across samples."""
    sample_rate = float(getattr(oscillator, "sample_rate", 44100.0))
    if previous_frequency == target_frequency:
        oscillator.frequency = target_frequency
        return oscillator.get_samples(num_samples), target_frequency

    frequencies = _frequency_slew_values(
        previous_frequency,
        target_frequency,
        num_samples,
        sample_rate,
        DEFAULT_FREQUENCY_SLEW_TIME_MS,
    )
    samples = []
    for frequency in frequencies:
        oscillator.frequency = float(frequency)
        samples.append(next(oscillator))
    final_frequency = float(frequencies[-1]) if len(frequencies) else target_frequency
    return np.asarray(samples, dtype=np.float32), final_frequency
