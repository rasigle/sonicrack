"""Pitch slide and portamento processors."""

from __future__ import annotations

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.utils.validation import validate_sample_rate


class SlideProcessor:
    """Smooth pitch changes when a slide signal is active."""

    def __init__(
        self,
        time: float = 0.08,
        *,
        always_on: bool = False,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.sample_rate = validate_sample_rate(sample_rate)
        self.time = float(time)
        self.always_on = bool(always_on)
        self._current_frequency: float | None = None

    def reset(self, frequency: float | None = None) -> None:
        self._current_frequency = None if frequency is None else float(frequency)

    def process(
        self, target_pitch_cv: np.ndarray, slide_signal: np.ndarray | None = None
    ) -> np.ndarray:
        """Render smoothed 1V/oct pitch CV for one buffer."""
        targets = np.asarray(target_pitch_cv, dtype=np.float32).reshape(-1)
        if slide_signal is None:
            slides = (
                np.ones(len(targets), dtype=np.float32)
                if self.always_on
                else np.zeros(len(targets), dtype=np.float32)
            )
        else:
            slides = self._fit_signal(slide_signal, len(targets))

        output = np.empty(len(targets), dtype=np.float32)
        if len(targets) == 0:
            return output

        if self._current_frequency is None:
            self._current_frequency = float(targets[0])

        slide_samples = max(1, int(round(max(0.0, self.time) * self.sample_rate)))
        alpha = 1.0 if slide_samples <= 1 else 1.0 / slide_samples

        for index, target in enumerate(targets):
            should_slide = self.always_on or slides[index] > 0.5
            if should_slide:
                self._current_frequency += (
                    float(target) - self._current_frequency
                ) * alpha
            else:
                self._current_frequency = float(target)
            output[index] = self._current_frequency

        return output

    @staticmethod
    def _fit_signal(values: np.ndarray, num_samples: int) -> np.ndarray:
        signal = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(signal) == num_samples:
            return signal
        if len(signal) > num_samples:
            return signal[:num_samples]
        padded = np.zeros(num_samples, dtype=np.float32)
        padded[: len(signal)] = signal
        return padded
