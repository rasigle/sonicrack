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
from src.engine.core.parameter import RuntimeParameter, SmoothingPolicy
from src.engine.core.registry import register_component
from src.engine.core.sample_mode import VALID_SAMPLE_MODES, SampleMode
from src.engine.dsp.modifiers.base import Modifier
from src.engine.utils.validation import (
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

        # Create RuntimeParameters with validation and smoothing
        threshold_descriptor = ParameterDescriptor(
            name="threshold_db",
            default=threshold_db,
            minimum=-60.0,
            maximum=0.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=10.0,
        )
        self._threshold_param = RuntimeParameter(
            threshold_descriptor, self._sample_rate
        )

        ratio_descriptor = ParameterDescriptor(
            name="ratio",
            default=ratio,
            minimum=1.0,
            maximum=20.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=10.0,
        )
        self._ratio_param = RuntimeParameter(ratio_descriptor, self._sample_rate)

        attack_descriptor = ParameterDescriptor(
            name="attack_ms",
            default=attack_ms,
            minimum=0.1,
            maximum=200.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=10.0,
        )
        self._attack_ms_param = RuntimeParameter(attack_descriptor, self._sample_rate)

        release_descriptor = ParameterDescriptor(
            name="release_ms",
            default=release_ms,
            minimum=1.0,
            maximum=1000.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=10.0,
        )
        self._release_ms_param = RuntimeParameter(release_descriptor, self._sample_rate)

        makeup_gain_descriptor = ParameterDescriptor(
            name="makeup_gain_db",
            default=makeup_gain_db,
            minimum=-24.0,
            maximum=24.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=10.0,
        )
        self._makeup_gain_param = RuntimeParameter(
            makeup_gain_descriptor, self._sample_rate
        )

        mix_descriptor = ParameterDescriptor(
            name="mix",
            default=mix,
            minimum=0.0,
            maximum=1.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=10.0,
        )
        self._mix_param = RuntimeParameter(mix_descriptor, self._sample_rate)

        self._gain_reduction_db: float | np.ndarray = 0.0
        self._sample_shape: tuple[int, ...] = ()
        self._update_coefficients()

    @property
    def threshold_db(self) -> float:
        return self._threshold_param._target_value

    @threshold_db.setter
    def threshold_db(self, value: float) -> None:
        self._threshold_param.value = value

    @property
    def ratio(self) -> float:
        return self._ratio_param._target_value

    @ratio.setter
    def ratio(self, value: float) -> None:
        self._ratio_param.value = value

    @property
    def attack_ms(self) -> float:
        return self._attack_ms_param._target_value

    @attack_ms.setter
    def attack_ms(self, value: float) -> None:
        self._attack_ms_param.value = value
        self._update_coefficients()

    @property
    def release_ms(self) -> float:
        return self._release_ms_param._target_value

    @release_ms.setter
    def release_ms(self, value: float) -> None:
        self._release_ms_param.value = value
        self._update_coefficients()

    @property
    def makeup_gain_db(self) -> float:
        return self._makeup_gain_param._target_value

    @makeup_gain_db.setter
    def makeup_gain_db(self, value: float) -> None:
        self._makeup_gain_param.value = value

    @property
    def mix(self) -> float:
        return self._mix_param._target_value

    @mix.setter
    def mix(self, value: float) -> None:
        self._mix_param.value = value

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
        self._sample_shape = ()

    def _update_coefficients(self) -> None:
        attack_seconds = self._attack_ms_param.value / 1000.0
        release_seconds = self._release_ms_param.value / 1000.0
        self._attack_coeff = float(np.exp(-1.0 / (attack_seconds * self._sample_rate)))
        self._release_coeff = float(
            np.exp(-1.0 / (release_seconds * self._sample_rate))
        )

    def _ensure_state_shape(self, sample_shape: tuple[int, ...]) -> None:
        if sample_shape == self._sample_shape:
            return
        self._sample_shape = sample_shape
        self._gain_reduction_db = (
            0.0 if sample_shape == () else np.zeros(sample_shape, dtype=np.float32)
        )

    def _target_gain_reduction_db(self, sample: np.ndarray) -> np.ndarray:
        level_db = 20.0 * np.log10(np.maximum(np.abs(sample), 1.0e-12))
        ratio = self._ratio_param.value
        threshold = self._threshold_param.value

        if ratio <= 1.0:
            return np.zeros_like(level_db, dtype=np.float32)

        compressed_level = threshold + ((level_db - threshold) / ratio)
        target = compressed_level - level_db

        return np.where(level_db <= threshold, 0.0, target).astype(
            np.float32,
            copy=False,
        )

    def _process_sample(self, sample: float) -> float:
        return float(self._process_frame(np.asarray(sample, dtype=np.float32)))

    def _process_frame(self, sample: np.ndarray) -> np.ndarray:
        self._ensure_state_shape(sample.shape)
        target_reduction = self._target_gain_reduction_db(sample)
        coeff = np.where(
            target_reduction < self._gain_reduction_db,
            self._attack_coeff,
            self._release_coeff,
        )
        self._gain_reduction_db = (
            coeff * self._gain_reduction_db + (1.0 - coeff) * target_reduction
        )
        makeup_gain = self._makeup_gain_param.value
        mix = self._mix_param.value
        gain = 10.0 ** ((self._gain_reduction_db + makeup_gain) / 20.0)
        wet = sample * gain
        return (sample * (1.0 - mix) + wet * mix).astype(
            np.float32,
            copy=False,
        )

    def _process_buffer(self, samples: np.ndarray) -> np.ndarray:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return samples.copy()

        self._ensure_state_shape(samples.shape[1:])
        output = np.empty_like(samples, dtype=np.float32)
        for index, sample in enumerate(samples):
            output[index] = self._process_frame(sample)
        return output

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Apply compression to a scalar or sample buffer."""
        if isinstance(val, (float, int, np.number)):
            return self._process_sample(float(val))

        if isinstance(val, tuple):
            result = self._process_frame(np.asarray(val, dtype=np.float32))
            return tuple(float(sample) for sample in result)

        return self._process_buffer(np.asarray(val, dtype=np.float32))

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
