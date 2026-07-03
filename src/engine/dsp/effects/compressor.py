"""Dynamic range compressor effect for audio processing.

A compressor reduces the dynamic range of a signal by turning down audio that
rises above a threshold. This makes peaks less dominant and can make quieter
details feel more consistent after makeup gain is applied. The implementation
uses a feed-forward level detector: each sample is measured, a target gain
reduction is calculated in decibels, and attack/release smoothing is applied so
gain changes do not jump abruptly between samples.

Parameters:
    threshold_db: The input level in dBFS where compression starts. Signals below
        this level pass through unchanged except for makeup gain and mix.
    ratio: How strongly signal above the threshold is reduced. A ratio of 1.0
        applies no compression; 4.0 means 4 dB above threshold becomes 1 dB above
        threshold; higher ratios approach limiting.
    attack_ms: How quickly gain reduction increases after the signal crosses the
        threshold. Short attacks catch transients; longer attacks preserve more
        initial punch.
    release_ms: How quickly gain reduction relaxes after the signal falls back
        below the threshold. Short releases recover quickly; longer releases sound
        smoother and avoid rapid gain movement.
    makeup_gain_db: Gain applied after compression to restore or reshape output
        level.
    mix: Dry/wet blend between uncompressed and compressed signal, useful for
        parallel compression.
    sample_rate: Processing sample rate used to convert attack/release times into
        smoothing coefficients.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    ComponentCategory,
    ComponentDescriptor,
    ParameterDescriptor,
)
from src.engine.core.registry import register_component
from src.engine.core.sample_mode import SampleMode, VALID_SAMPLE_MODES
from src.engine.dsp.modifiers.base import Modifier
from src.engine.utils.validation import (
    validate_numeric_range,
    validate_sample_count,
    validate_sample_rate,
)


@register_component()
class Compressor(Modifier):
    """Feed-forward dynamic range compressor.

    Args:
        source: Optional input audio source for iterator/vectorized source usage.
        threshold_db: Level above which compression begins, in dBFS.
        ratio: Compression ratio. 1.0 means no compression.
        attack_ms: Gain-reduction attack time in milliseconds.
        release_ms: Gain-reduction release time in milliseconds.
        makeup_gain_db: Output gain applied after compression.
        mix: Dry/wet mix.
        sample_rate: Processing sample rate in Hz.
    """

    descriptor = ComponentDescriptor(
        name="Compressor",
        category=ComponentCategory.EFFECT,
        description="Dynamic range compressor with attack and release smoothing",
        tags=["effect", "compressor", "dynamics"],
        parameters={
            "threshold_db": ParameterDescriptor(
                name="threshold_db",
                default=-18.0,
                minimum=-60.0,
                maximum=0.0,
                unit="dB",
                description="Compression threshold.",
            ),
            "ratio": ParameterDescriptor(
                name="ratio",
                default=4.0,
                minimum=1.0,
                maximum=20.0,
                description="Compression ratio.",
            ),
            "attack_ms": ParameterDescriptor(
                name="attack_ms",
                default=10.0,
                minimum=0.1,
                maximum=200.0,
                unit="ms",
                description="Gain-reduction attack time.",
            ),
            "release_ms": ParameterDescriptor(
                name="release_ms",
                default=100.0,
                minimum=1.0,
                maximum=1000.0,
                unit="ms",
                description="Gain-reduction release time.",
            ),
            "makeup_gain_db": ParameterDescriptor(
                name="makeup_gain_db",
                default=0.0,
                minimum=-24.0,
                maximum=24.0,
                unit="dB",
                description="Post-compression makeup gain.",
            ),
            "mix": ParameterDescriptor(
                name="mix",
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                description="Dry/wet mix.",
            ),
            "sample_rate": ParameterDescriptor(
                name="sample_rate",
                default=DEFAULT_SAMPLE_RATE,
                minimum=1.0,
                unit="Hz",
                description="Processing sample rate.",
            ),
        },
        fluent_api_name="compressor",
    )

    def __init__(
        self,
        source: Any | None = None,
        threshold_db: float = -18.0,
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        makeup_gain_db: float = 0.0,
        mix: float = 1.0,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source = source
        self._sample_rate = validate_sample_rate(sample_rate)
        self._threshold_db = validate_numeric_range(
            threshold_db, -60.0, 0.0, name="threshold_db"
        )
        self._ratio = validate_numeric_range(ratio, 1.0, 20.0, name="ratio")
        self._attack_ms = validate_numeric_range(
            attack_ms, 0.1, 200.0, name="attack_ms"
        )
        self._release_ms = validate_numeric_range(
            release_ms, 1.0, 1000.0, name="release_ms"
        )
        self._makeup_gain_db = validate_numeric_range(
            makeup_gain_db, -24.0, 24.0, name="makeup_gain_db"
        )
        self._mix = validate_numeric_range(mix, 0.0, 1.0, name="mix")
        self._gain_reduction_db = 0.0
        self._update_coefficients()

    @property
    def threshold_db(self) -> float:
        return self._threshold_db

    @threshold_db.setter
    def threshold_db(self, value: float) -> None:
        self._threshold_db = validate_numeric_range(
            value, -60.0, 0.0, name="threshold_db"
        )

    @property
    def ratio(self) -> float:
        return self._ratio

    @ratio.setter
    def ratio(self, value: float) -> None:
        self._ratio = validate_numeric_range(value, 1.0, 20.0, name="ratio")

    @property
    def attack_ms(self) -> float:
        return self._attack_ms

    @attack_ms.setter
    def attack_ms(self, value: float) -> None:
        self._attack_ms = validate_numeric_range(value, 0.1, 200.0, name="attack_ms")
        self._update_coefficients()

    @property
    def release_ms(self) -> float:
        return self._release_ms

    @release_ms.setter
    def release_ms(self, value: float) -> None:
        self._release_ms = validate_numeric_range(value, 1.0, 1000.0, name="release_ms")
        self._update_coefficients()

    @property
    def makeup_gain_db(self) -> float:
        return self._makeup_gain_db

    @makeup_gain_db.setter
    def makeup_gain_db(self, value: float) -> None:
        self._makeup_gain_db = validate_numeric_range(
            value, -24.0, 24.0, name="makeup_gain_db"
        )

    @property
    def mix(self) -> float:
        return self._mix

    @mix.setter
    def mix(self, value: float) -> None:
        self._mix = validate_numeric_range(value, 0.0, 1.0, name="mix")

    @property
    def sample_rate(self) -> float:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        self._sample_rate = validate_sample_rate(value)
        self._update_coefficients()

    def reset(self) -> None:
        """Clear compressor envelope state."""
        self._gain_reduction_db = 0.0

    def _update_coefficients(self) -> None:
        attack_seconds = self._attack_ms / 1000.0
        release_seconds = self._release_ms / 1000.0
        self._attack_coeff = float(np.exp(-1.0 / (attack_seconds * self._sample_rate)))
        self._release_coeff = float(
            np.exp(-1.0 / (release_seconds * self._sample_rate))
        )

    def _target_gain_reduction_db(self, sample: float) -> float:
        level_db = 20.0 * np.log10(max(abs(sample), 1.0e-12))
        if level_db <= self._threshold_db or self._ratio <= 1.0:
            return 0.0

        compressed_level = self._threshold_db + (
            (level_db - self._threshold_db) / self._ratio
        )
        return float(compressed_level - level_db)

    def _process_sample(self, sample: float) -> float:
        target_reduction = self._target_gain_reduction_db(sample)
        coeff = (
            self._attack_coeff
            if target_reduction < self._gain_reduction_db
            else self._release_coeff
        )
        self._gain_reduction_db = (
            coeff * self._gain_reduction_db + (1.0 - coeff) * target_reduction
        )
        gain = 10.0 ** ((self._gain_reduction_db + self._makeup_gain_db) / 20.0)
        wet = sample * gain
        return sample * (1.0 - self._mix) + wet * self._mix

    def __call__(self, val: float | np.ndarray) -> float | np.ndarray:
        """Apply compression to a scalar or sample buffer."""
        if isinstance(val, (float, int, np.number)):
            return self._process_sample(float(val))

        samples = np.asarray(val, dtype=np.float32)
        output = np.empty(len(samples), dtype=np.float32)
        for index, sample in enumerate(samples):
            output[index] = self._process_sample(float(sample))
        return output

    def __iter__(self):
        if self.source is not None:
            source = cast(Any, self.source)
            iter(source)
        return self

    def __next__(self) -> float:
        if self.source is None:
            raise ValueError("source is required for iterator usage")

        source = cast(Any, self.source)
        return self._process_sample(float(next(source)))

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        n = validate_sample_count(n)
        if self.source is None:
            raise ValueError("source is required for get_samples_vectorized()")

        source = cast(Any, self.source)
        return np.asarray(self(source.get_samples_vectorized(n)), dtype=np.float32)

    def get_samples(
        self, n: int, mode: SampleMode = "vectorized", **kwargs
    ) -> np.ndarray:
        """Get n samples using specified mode."""
        if mode not in VALID_SAMPLE_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'auto', 'iterator', or 'vectorized'."
            )
        if mode == "auto":
            mode = "vectorized" if n >= 512 else "iterator"
        if mode == "vectorized":
            return self.get_samples_vectorized(n)

        n = validate_sample_count(n)
        return np.array([next(self) for _ in range(n)], dtype=np.float32)
