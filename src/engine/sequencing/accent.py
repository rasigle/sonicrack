"""Accent CV shaping utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.utils.validation import validate_sample_rate


@dataclass(slots=True)
class AccentFrame:
    """Rendered accent CV outputs for one audio buffer."""

    amp: np.ndarray
    cutoff: np.ndarray
    envelope: np.ndarray


class AccentProcessor:
    """Convert accent gates into smoothed control-voltage lanes."""

    def __init__(
        self,
        amount: float = 1.0,
        decay: float = 0.08,
        *,
        amp_depth: float = 0.35,
        cutoff_depth: float = 0.6,
        envelope_depth: float = 0.5,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.sample_rate = validate_sample_rate(sample_rate)
        self.amount = float(amount)
        self.decay = float(decay)
        self.amp_depth = float(amp_depth)
        self.cutoff_depth = float(cutoff_depth)
        self.envelope_depth = float(envelope_depth)
        self._level = 0.0
        self._previous_accent = 0.0

    def reset(self) -> None:
        """Clear accent state."""
        self._level = 0.0
        self._previous_accent = 0.0

    def process(self, accent_signal: np.ndarray) -> AccentFrame:
        """Render smoothed accent CV lanes for one buffer."""
        accents = np.asarray(accent_signal, dtype=np.float32).reshape(-1)
        base = np.empty(len(accents), dtype=np.float32)
        decay_samples = max(1, int(round(max(0.0, self.decay) * self.sample_rate)))
        release = 1.0 if decay_samples <= 1 else 1.0 / decay_samples
        amount = float(np.clip(self.amount, 0.0, 1.0))

        for index, accent in enumerate(accents):
            current = float(accent)
            if self._previous_accent <= 0.5 < current:
                self._level = amount
            elif current > 0.5:
                self._level = max(self._level, amount)
            else:
                self._level = max(0.0, self._level - release)

            base[index] = self._level
            self._previous_accent = current

        return AccentFrame(
            amp=np.clip(base * self.amp_depth, 0.0, 1.0).astype(np.float32),
            cutoff=np.clip(base * self.cutoff_depth, 0.0, 1.0).astype(np.float32),
            envelope=np.clip(base * self.envelope_depth, 0.0, 1.0).astype(np.float32),
        )
