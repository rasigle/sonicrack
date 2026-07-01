"""Sample-accurate step clock utilities."""

from __future__ import annotations

import math

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.utils.validation import validate_sample_count, validate_sample_rate


class StepClock:
    """Generate pulse buffers for musical step timing.

    The clock emits a value of ``1.0`` on samples where a new step begins and
    ``0.0`` elsewhere. The first rendered buffer starts with a pulse at sample
    zero so downstream sequencers can initialize their first step immediately.
    """

    _DIVISION_BEATS = {
        "1/4": 1.0,
        "1/8": 0.5,
        "1/16": 0.25,
        "1/32": 0.125,
    }

    def __init__(
        self,
        bpm: float = 120.0,
        division: str = "1/16",
        swing: float = 0.0,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.sample_rate = validate_sample_rate(sample_rate)
        self.bpm = float(bpm)
        self.division = division
        self.swing = float(swing)
        self._samples_until_next_step = 0
        self._step_index = 0

    def reset(self) -> None:
        """Reset phase so the next rendered buffer starts on a step."""
        self._samples_until_next_step = 0
        self._step_index = 0

    @property
    def step_samples(self) -> int:
        """Base number of samples per step before swing alternation."""
        if self.bpm <= 0:
            raise ValueError(f"bpm must be positive, got {self.bpm}")
        beats = self._DIVISION_BEATS.get(self.division)
        if beats is None:
            raise ValueError(
                f"Unsupported division {self.division!r}. "
                f"Expected one of {tuple(self._DIVISION_BEATS)}"
            )
        samples = (60.0 / self.bpm) * beats * self.sample_rate
        return max(1, int(round(samples)))

    def _next_step_samples(self) -> int:
        base = self.step_samples
        swing = float(np.clip(self.swing, 0.0, 0.75))
        if swing == 0.0:
            return base

        # Delay every second 16th-style step and shorten the following step by
        # the same amount so the two-step cycle length stays stable.
        direction = 1.0 if self._step_index % 2 == 0 else -1.0
        return max(1, int(round(base * (1.0 + direction * swing))))

    def process(self, num_samples: int, *, running: bool = True) -> np.ndarray:
        """Render a clock pulse buffer."""
        num_samples = validate_sample_count(num_samples)
        pulses = np.zeros(num_samples, dtype=np.float32)
        if not running:
            return pulses

        index = 0
        while index < num_samples:
            if self._samples_until_next_step <= 0:
                pulses[index] = 1.0
                self._samples_until_next_step = self._next_step_samples()
                self._step_index += 1

            advance = min(self._samples_until_next_step, num_samples - index)
            self._samples_until_next_step -= advance
            index += advance

        return pulses


def division_to_beats(division: str) -> float:
    """Return beat duration for a supported textual division."""
    value = StepClock._DIVISION_BEATS.get(division)
    if value is None or not math.isfinite(value):
        raise ValueError(f"Unsupported division {division!r}")
    return value
