"""Decay envelope generator for plucks and percussive sounds."""

from __future__ import annotations

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    ComponentCategory,
    ComponentDescriptor,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.core.registry import register_component
from src.engine.dsp.modulators.base import Modulator
from src.engine.utils.validation import validate_sample_count, validate_sample_rate


@register_component()
class DecayEnvelope(Modulator):
    """Triggered attack-decay envelope for plucks and filter modulation."""

    descriptor = ComponentDescriptor(
        name="DecayEnvelope",
        category=ComponentCategory.MODULATOR,
        description="Attack-decay envelope generator",
        tags=["envelope", "modulator", "decay", "pluck"],
        parameters=make_parameter_descriptors(
            "attack_duration",
            "decay_duration",
            "amount",
            "sample_rate",
            attack_duration=ParameterDescriptor(
                name="attack_duration",
                default=0.0,
                minimum=0.0,
                unit="s",
                description="Attack duration.",
            ),
            decay_duration=ParameterDescriptor(
                name="decay_duration",
                default=0.18,
                minimum=0.0,
                unit="s",
                description="Decay duration.",
            ),
            amount=ParameterDescriptor(
                name="amount",
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                description="Peak envelope amount.",
            ),
        ),
        fluent_api_name="decay_envelope",
    )

    def __init__(
        self,
        attack_duration: float = 0.0,
        decay_duration: float = 0.18,
        amount: float = 1.0,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self._attack_duration = float(attack_duration)
        self._decay_duration = float(decay_duration)
        super().__init__(sample_rate=sample_rate)
        self.amount = float(amount)
        self._phase = "idle"
        self._phase_position = 0
        self.val = 0.0
        self.ended = True
        self._update_phase_samples()

    @property
    def attack_duration(self) -> float:
        return self._attack_duration

    @attack_duration.setter
    def attack_duration(self, value: float) -> None:
        self._attack_duration = float(value)
        self._update_phase_samples()

    @property
    def decay_duration(self) -> float:
        return self._decay_duration

    @decay_duration.setter
    def decay_duration(self, value: float) -> None:
        self._decay_duration = float(value)
        self._update_phase_samples()

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        self._sample_rate = validate_sample_rate(value)
        self._update_phase_samples()

    def reset(self) -> None:
        """Return the envelope to idle."""
        self._phase = "idle"
        self._phase_position = 0
        self.val = 0.0
        self.ended = True

    def trigger_note_on(self) -> None:
        """Start the attack/decay shape from the beginning."""
        self._phase = "attack" if self._attack_samples > 0 else "decay"
        self._phase_position = 0
        self.val = 0.0 if self._phase == "attack" else self.amount
        self.ended = False

    def trigger_note_off(self) -> None:
        """Pluck envelopes ignore note-off; decay continues naturally."""

    def get_samples(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False, mode: str = "auto"
    ) -> np.ndarray:
        _ = mode
        n = validate_sample_count(n)
        if reset:
            self.reset()
        return np.asarray([next(self) for _ in range(n)], dtype=np.float32)

    def __next__(self) -> float:
        if self._phase == "idle" or self.ended:
            self.val = 0.0
            return 0.0

        if self._phase == "attack":
            progress = self._phase_position / max(1, self._attack_samples)
            self.val = self.amount * min(progress, 1.0)
            self._phase_position += 1
            if self._phase_position >= self._attack_samples:
                self._phase = "decay"
                self._phase_position = 0
            return float(self.val)

        if self._decay_samples <= 0:
            self.reset()
            return 0.0

        progress = self._phase_position / self._decay_samples
        self.val = self.amount * max(0.0, 1.0 - progress)
        self._phase_position += 1
        if self._phase_position > self._decay_samples:
            self.reset()
        return float(self.val)

    def _update_phase_samples(self) -> None:
        self._attack_samples = int(max(0.0, self._attack_duration) * self._sample_rate)
        self._decay_samples = int(max(0.0, self._decay_duration) * self._sample_rate)
