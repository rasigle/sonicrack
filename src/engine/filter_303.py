"""Acid-style resonant low-pass filter components."""

from __future__ import annotations

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.audio_component import (
    ComponentCategory,
    ComponentDescriptor,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.audio_component_registry import register_component
from src.engine.filter import BiquadResonantFilter
from src.engine.modifier import Modifier
from src.engine.validation import validate_sample_rate


@register_component()
class AcidResonantFilter(Modifier):
    """303-inspired resonant low-pass filter with envelope and accent CV.

    This is a pragmatic character filter for acid patches. It composes the
    existing stateful biquad filter with 303-oriented modulation lanes and drive,
    keeping the general-purpose filter unchanged.
    """

    descriptor = ComponentDescriptor(
        name="AcidResonantFilter",
        category=ComponentCategory.MODIFIER,
        description="Acid-style resonant low-pass filter with env/accent CV",
        tags=["filter", "acid", "303", "resonant", "drive"],
        parameters=make_parameter_descriptors(
            "cutoff",
            "resonance",
            "env_amount",
            "accent_amount",
            "drive_db",
            "output_gain_db",
            "sample_rate",
            cutoff=ParameterDescriptor(
                name="cutoff",
                default=700.0,
                minimum=20.0,
                unit="Hz",
                description="Base cutoff frequency.",
            ),
            resonance=ParameterDescriptor(
                name="resonance",
                default=8.0,
                minimum=0.1,
                maximum=30.0,
                description="Filter Q/resonance.",
            ),
            env_amount=ParameterDescriptor(
                name="env_amount",
                default=2.5,
                minimum=0.0,
                maximum=6.0,
                unit="oct",
                description="Envelope cutoff modulation depth in octaves.",
            ),
            accent_amount=ParameterDescriptor(
                name="accent_amount",
                default=1.0,
                minimum=0.0,
                maximum=4.0,
                unit="oct",
                description="Accent cutoff boost in octaves.",
            ),
            drive_db=ParameterDescriptor(
                name="drive_db",
                default=6.0,
                unit="dB",
                description="Pre-filter saturation drive.",
            ),
            output_gain_db=ParameterDescriptor(
                name="output_gain_db",
                default=-6.0,
                unit="dB",
                description="Post-filter output gain.",
            ),
        ),
        fluent_api_name="acid_filter",
    )

    def __init__(
        self,
        cutoff: float = 700.0,
        resonance: float = 8.0,
        env_amount: float = 2.5,
        accent_amount: float = 1.0,
        drive_db: float = 6.0,
        output_gain_db: float = -6.0,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ) -> None:
        super().__init__()
        self.sample_rate = validate_sample_rate(sample_rate)
        self.cutoff = float(cutoff)
        self.resonance = float(resonance)
        self.env_amount = float(env_amount)
        self.accent_amount = float(accent_amount)
        self.drive_db = float(drive_db)
        self.output_gain_db = float(output_gain_db)
        self._filter = BiquadResonantFilter(
            cutoff=self.cutoff,
            resonance=self.resonance,
            filter_type="low",
            drive_db=self.drive_db,
            output_gain_db=self.output_gain_db,
            sample_rate=self.sample_rate,
        )

    def reset_state(self) -> None:
        """Reset filter memory."""
        self._filter.reset_state()

    def __call__(self, val: float | tuple[float, ...] | np.ndarray):
        if isinstance(val, np.ndarray):
            return self.process_modulated(val)
        return self._filter(val)

    def process_modulated(
        self,
        samples: np.ndarray,
        cutoff_cv: np.ndarray | None = None,
        env_cv: np.ndarray | None = None,
        accent_cv: np.ndarray | None = None,
    ) -> np.ndarray:
        """Process a buffer with 303-oriented cutoff modulation lanes."""
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return samples

        num_samples = len(samples)
        cutoff_values = np.full(num_samples, self.cutoff, dtype=np.float32)
        if cutoff_cv is not None:
            cutoff_values *= np.power(2.0, self._fit_cv(cutoff_cv, num_samples))
        if env_cv is not None:
            cutoff_values *= np.power(
                2.0,
                self._fit_cv(env_cv, num_samples) * self.env_amount,
            )
        if accent_cv is not None:
            cutoff_values *= np.power(
                2.0,
                self._fit_cv(accent_cv, num_samples) * self.accent_amount,
            )

        cutoff_values = np.clip(cutoff_values, 20.0, self.sample_rate * 0.45)
        self._filter.configure(
            cutoff=self.cutoff,
            resonance=self.resonance,
            filter_type="low",
            drive_db=self.drive_db,
            output_gain_db=self.output_gain_db,
        )
        return self._filter.process_modulated(
            samples,
            cutoff_values=cutoff_values,
            resonance_values=None,
            drive_db_values=None,
            output_gain_db_values=None,
        )

    @staticmethod
    def _fit_cv(values: np.ndarray, num_samples: int) -> np.ndarray:
        cv = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(cv) == num_samples:
            return cv
        if len(cv) > num_samples:
            return cv[:num_samples]
        padded = np.zeros(num_samples, dtype=np.float32)
        padded[: len(cv)] = cv
        return padded
