"""Distortion effect for audio processing."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from src.constants import AUTO_MODE_VECTORIZE_THRESHOLD
from src.engine.core.component import (
    ComponentCategory,
    ComponentDescriptor,
    ParameterDescriptor,
)
from src.engine.core.registry import register_component
from src.engine.core.sample_mode import VALID_SAMPLE_MODES, SampleMode
from src.engine.dsp.modifiers.base import Modifier
from src.engine.utils.validation import (
    validate_numeric_range,
    validate_sample_count,
)


@register_component()
class Distortion(Modifier):
    """Distortion effect with multiple distortion types.

    Applies non-linear waveshaping to the input signal, creating harmonic distortion.
    Supports various distortion algorithms including soft clipping, hard clipping,
    and waveshaping.

    Can be used in two ways:
    1. In Chain as Modifier: Chain(osc, Distortion(drive=2.0))
    2. Wrapping a source: Distortion(source=osc, drive=2.0)

    Args:
        source: Optional input audio source (for wrapping usage). Default: None
        drive: Amount of distortion (0.0 to 10.0). Default: 1.0
        mix: Dry/wet mix (0.0 = dry, 1.0 = wet). Default: 1.0
            output_gain: Output level compensation (0.0 to 2.0). Default: 0.5
        distortion_type: Type of distortion ("soft", "hard", "fuzz", "tube").
            Default: "soft"
    """

    descriptor = ComponentDescriptor(
        name="Distortion",
        category=ComponentCategory.EFFECT,
        description="Distortion effect with multiple distortion types",
        tags=["effect", "distortion", "overdrive", "waveshaper"],
        parameters={
            "drive": ParameterDescriptor(
                name="drive",
                default=1.0,
                minimum=0.0,
                maximum=10.0,
                description="Distortion pre-gain amount.",
            ),
            "mix": ParameterDescriptor(
                name="mix",
                default=1.0,
                minimum=0.0,
                maximum=1.0,
                description="Dry/wet mix.",
            ),
            "output_gain": ParameterDescriptor(
                name="output_gain",
                default=0.5,
                minimum=0.0,
                maximum=2.0,
                description="Post-distortion output gain.",
            ),
            "distortion_type": ParameterDescriptor(
                name="distortion_type",
                default="soft",
                choices=("soft", "hard", "fuzz", "tube"),
                description="Waveshaping algorithm.",
            ),
        },
        fluent_api_name="distortion",
    )

    def __init__(
        self,
        source: Any | None = None,
        drive: float = 1.0,
        mix: float = 1.0,
        output_gain: float = 0.5,
        distortion_type: str = "soft",
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize distortion effect.

        Args:
            source: Optional input audio source (for wrapping usage)
            drive: Distortion amount (0.0-10.0)
            mix: Dry/wet mix (0.0-1.0)
            output_gain: Output level (0.0-2.0)
            distortion_type: Distortion algorithm type

        Raises:
            ValueError: If parameters are out of valid range
        """
        super().__init__(*args, **kwargs)
        self.source = source  # Optional!
        self._drive = validate_numeric_range(drive, 0.0, 10.0, name="drive")
        self._mix = validate_numeric_range(mix, 0.0, 1.0, name="mix")
        self._output_gain = validate_numeric_range(
            output_gain, 0.0, 2.0, name="output_gain"
        )

        valid_types = ["soft", "hard", "fuzz", "tube"]
        if distortion_type not in valid_types:
            raise ValueError(
                f"distortion_type must be one of {valid_types}, got '{distortion_type}'"
            )
        self._distortion_type = distortion_type

    @property
    def drive(self) -> float:
        """float: Distortion drive amount (0.0-10.0)."""
        return self._drive

    @drive.setter
    def drive(self, value: float):
        """Set drive amount."""
        self._drive = validate_numeric_range(value, 0.0, 10.0, name="drive")

    @property
    def mix(self) -> float:
        """float: Dry/wet mix (0.0-1.0)."""
        return self._mix

    @mix.setter
    def mix(self, value: float):
        """Set mix amount."""
        self._mix = validate_numeric_range(value, 0.0, 1.0, name="mix")

    @property
    def output_gain(self) -> float:
        """float: Output gain (0.0-2.0)."""
        return self._output_gain

    @output_gain.setter
    def output_gain(self, value: float):
        """Set output gain."""
        self._output_gain = validate_numeric_range(value, 0.0, 2.0, name="output_gain")

    @property
    def distortion_type(self) -> str:
        """str: Distortion type."""
        return self._distortion_type

    @distortion_type.setter
    def distortion_type(self, value: str):
        """Set distortion type."""
        valid_types = ["soft", "hard", "fuzz", "tube"]
        if value not in valid_types:
            raise ValueError(
                f"distortion_type must be one of {valid_types}, got '{value}'"
            )
        self._distortion_type = value

    def _apply_distortion(self, samples: np.ndarray) -> np.ndarray:
        """Apply selected distortion algorithm to samples.

        Args:
            samples: Input samples

        Returns:
            Distorted samples
        """
        # Pre-amplify based on drive
        amplified = samples * (1.0 + self._drive)

        if self._distortion_type == "soft":
            # Soft clipping using tanh
            distorted = np.tanh(amplified)

        elif self._distortion_type == "hard":
            # Hard clipping
            distorted = np.clip(amplified, -1.0, 1.0)

        elif self._distortion_type == "fuzz":
            # Fuzz using asymmetric clipping
            distorted = np.where(
                amplified > 0,
                np.tanh(amplified * 2.0) * 0.7,
                np.clip(amplified * 1.5, -1.0, 0) * 0.8,
            )

        elif self._distortion_type == "tube":
            # Tube-style distortion (soft knee)
            threshold = 0.3
            distorted = np.where(
                np.abs(amplified) < threshold,
                amplified,
                np.sign(amplified)
                * (
                    threshold
                    + (1.0 - threshold)
                    * np.tanh((np.abs(amplified) - threshold) / (1.0 - threshold))
                ),
            )
        else:
            # Fallback to soft clipping if unknown type
            distorted = np.tanh(amplified)

        return np.asarray(distorted, dtype=np.float32)

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Apply distortion to value(s) - Modifier interface.

        This allows Distortion to be used directly in Chain.

        Args:
            val: Input value (scalar or array)

        Returns:
            Distorted value (same type as input)
        """
        # Handle scalar
        if isinstance(val, (float, int, np.number)):
            dry_sample = float(val)
            wet_sample = self._apply_distortion(
                np.array([dry_sample], dtype=np.float32)
            )[0]
            mixed = dry_sample * (1.0 - self._mix) + wet_sample * self._mix
            return mixed * self._output_gain

        if isinstance(val, tuple):
            result = self(np.asarray(val, dtype=np.float32))
            return tuple(float(sample) for sample in np.asarray(result))

        # Handle array
        dry_samples = np.asarray(val, dtype=np.float32)
        wet_samples = self._apply_distortion(dry_samples)
        mixed = dry_samples * (1.0 - self._mix) + wet_samples * self._mix
        return (mixed * self._output_gain).astype(np.float32)

    def __iter__(self):
        """Initialize iterator."""
        if self.source is not None:
            source = cast(Any, self.source)
            iter(source)
        return self

    def __next__(self) -> float:
        """Get next sample with distortion applied."""
        if self.source is None:
            raise ValueError("source is required for iterator usage")

        source = cast(Any, self.source)
        dry = next(source)

        # Apply distortion
        wet = self._apply_distortion(np.array([dry]))[0]

        # Mix dry and wet
        mixed = dry * (1.0 - self._mix) + wet * self._mix

        # Apply output gain
        return mixed * self._output_gain

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n distorted samples using vectorized processing.

        Args:
            n: Number of samples to generate

        Returns:
            Array of distorted samples

        Raises:
            ValueError: If source is None (required for this method)
        """
        n = validate_sample_count(n)
        if self.source is None:
            raise ValueError("source is required for get_samples_vectorized()")

        # Get input samples
        source = cast(Any, self.source)
        dry = source.get_samples_vectorized(n)

        # Apply distortion
        wet = self._apply_distortion(dry)

        # Mix dry and wet
        mixed = dry * (1.0 - self._mix) + wet * self._mix

        # Apply output gain
        return (mixed * self._output_gain).astype(np.float32)

    def get_samples(
        self, n: int, mode: SampleMode = "vectorized", **kwargs
    ) -> np.ndarray:
        """Get n samples using specified mode."""
        if mode not in VALID_SAMPLE_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'auto', 'iterator', or 'vectorized'."
            )
        if mode == "auto":
            mode = "vectorized" if n >= AUTO_MODE_VECTORIZE_THRESHOLD else "iterator"
        if mode == "vectorized":
            return self.get_samples_vectorized(n)

        n = validate_sample_count(n)
        return np.array([next(self) for _ in range(n)], dtype=np.float32)
