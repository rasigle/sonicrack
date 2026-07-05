"""Panning modifiers for stereo positioning."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.component import ComponentDescriptor, ParameterDescriptor
from src.engine.core.parameter import RuntimeParameter, SmoothingPolicy
from src.engine.core.registry import ComponentCategory, register_component
from src.engine.dsp.modifiers.base import (
    Modifier,
    _get_modulation_values,
    _get_next_modulation_value,
    _validate_modulator,
)
from src.engine.utils.validation import validate_numeric, validate_sample_rate

logger = logging.getLogger(__name__)


def _validate_position(value: float) -> float:
    """Validate and clamp pan position to the supported range [-1.0, 1.0].

    Args:
        value: Pan position value.

    Returns:
        Validated and clamped pan position.

    Raises:
        TypeError: If value is not numeric.
    """
    value = validate_numeric(value, "position")
    return float(np.clip(value, -1.0, 1.0))


def _validate_smoothing_time_ms(value: float) -> float:
    """Validate smoothing duration in milliseconds.

    Args:
        value: Smoothing duration in milliseconds.

    Returns:
        Validated smoothing duration.

    Raises:
        TypeError: If value is not numeric.
        ValueError: If value is negative.
    """
    value = validate_numeric(value, "smoothing_time_ms")
    if value < 0:
        raise ValueError(f"smoothing_time_ms must be non-negative, got {value}")
    return float(value)


def _validate_mono_array(samples: np.ndarray, name: str = "samples") -> None:
    """Validate that samples are mono 1D audio data.

    Panner converts mono input to stereo output. Passing an already-stereo or
    multichannel array would make the output shape ambiguous, so it is rejected.

    Args:
        samples: Input samples.
        name: Argument name used in error messages.

    Raises:
        ValueError: If samples is not a 1D array.
    """
    if not isinstance(samples, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array, got {type(samples)}")

    if samples.ndim != 1:
        raise ValueError(
            f"{name} must be a mono 1D NumPy array for panning; "
            f"got shape {samples.shape}"
        )


def _calculate_constant_power_gains(
    positions: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Calculate constant-power panning gains for pan position(s).

    Args:
        positions: Pan position or positions in range [-1.0, 1.0].

    Returns:
        Tuple of left and right gain value(s).
    """
    clipped_positions = np.clip(positions, -1.0, 1.0)
    angles = (clipped_positions + 1.0) * np.pi / 4.0
    return np.cos(angles), np.sin(angles)


@register_component()
class Panner(Modifier):
    """Converts mono input into stereo output with configurable pan position.

    Uses a constant-power panning law for perceptually uniform panning.
    Position range: -1.0 (hard left) to 1.0 (hard right), 0.0 (center).

    Args:
        position: Pan position. -1.0=left, 0.0=center, 1.0=right. Defaults to 0.0.
        sample_rate: Processing sample rate used for smoothing duration.
        smoothing_time_ms: Pan transition duration in milliseconds.
    """

    descriptor = ComponentDescriptor(
        name="Panner",
        category=ComponentCategory.MODIFIER,
        parameters={
            "position": ParameterDescriptor(
                name="position",
                default=0.0,
                minimum=-1.0,
                maximum=1.0,
                description="Pan position from hard left to hard right.",
            ),
            "sample_rate": ParameterDescriptor(
                name="sample_rate",
                default=DEFAULT_SAMPLE_RATE,
                minimum=1.0,
                unit="Hz",
                description="Processing sample rate.",
            ),
            "smoothing_time_ms": ParameterDescriptor(
                name="smoothing_time_ms",
                default=10.0,
                minimum=0.0,
                unit="ms",
                description="Pan smoothing duration.",
            ),
        },
        description="Stereo panner with constant-power panning law",
        fluent_api_name="panner",
        tags=["modifier", "panner", "stereo"],
    )

    def __init__(
        self,
        position: float = 0.0,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        smoothing_time_ms: float = 10.0,
    ) -> None:
        """Initialize panner with pan position.

        Args:
            position: Pan value. -1.0 is hard left, 0.0 is center,
                and 1.0 is hard right. Values outside the range are clamped.
            sample_rate: Processing sample rate used for smoothing duration.
            smoothing_time_ms: Pan transition duration in milliseconds.

        Raises:
            TypeError: If position or smoothing_time_ms is not numeric.
            ValueError: If sample_rate is invalid or smoothing_time_ms is negative.
        """
        super().__init__()
        self.sample_rate = validate_sample_rate(sample_rate)
        self.smoothing_time_ms = _validate_smoothing_time_ms(smoothing_time_ms)

        position = _validate_position(position)
        initial_left, initial_right = _calculate_constant_power_gains(position)

        left_gain_descriptor = ParameterDescriptor(
            name="left_gain",
            default=float(initial_left),
            minimum=0.0,
            maximum=1.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=self.smoothing_time_ms,
        )
        self._left_gain_param = RuntimeParameter(
            left_gain_descriptor, sample_rate=self.sample_rate
        )

        right_gain_descriptor = ParameterDescriptor(
            name="right_gain",
            default=float(initial_right),
            minimum=0.0,
            maximum=1.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=self.smoothing_time_ms,
        )
        self._right_gain_param = RuntimeParameter(
            right_gain_descriptor, sample_rate=self.sample_rate
        )

        self._position = position
        logger.debug("Panner initialized with position: %s", self._position)

    @property
    def position(self) -> float:
        """Current pan position (-1.0 to 1.0)."""
        return self._position

    @position.setter
    def position(self, value: float) -> None:
        """Set pan position and update left/right gains with smoothing.

        Values outside [-1.0, 1.0] are clamped.
        """
        value = _validate_position(value)
        self._position = value

        new_left, new_right = _calculate_constant_power_gains(value)
        self._left_gain_param.value = float(new_left)
        self._right_gain_param.value = float(new_right)

    @property
    def _smoothing_samples_remaining(self) -> int:
        """Backward compatibility: return max smoothing samples from parameters."""
        return max(
            self._left_gain_param._smoothing_samples_remaining,
            self._right_gain_param._smoothing_samples_remaining,
        )

    @property
    def _smoothing_duration_samples(self) -> int:
        """Backward compatibility: return smoothing duration in samples."""
        return self._left_gain_param._smoothing_duration_samples

    def __call__(
        self, val: float | np.ndarray
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """Convert mono signal to stereo with panning.

        Scalar input uses the current target gains for immediate response.
        NumPy array input uses ``pan_vectorized()`` and therefore applies
        smoothing envelopes when a pan change is in progress.

        Args:
            val: Mono scalar input value or mono 1D NumPy array.

        Returns:
            Tuple of (left, right) stereo values or arrays.

        Raises:
            TypeError: If val is not numeric or a NumPy array.
            ValueError: If val is a non-1D NumPy array.
        """
        if isinstance(val, np.ndarray):
            return self.pan_vectorized(val)

        if isinstance(val, (float, int, np.number)):
            left_gain = self._left_gain_param.target
            right_gain = self._right_gain_param.target
            return float(left_gain * val), float(right_gain * val)

        logger.error("Invalid input type for Panner: %s", type(val))
        raise TypeError(
            "Input value must be an int, float, numpy number, or mono 1D "
            f"NumPy array. Got {type(val)}"
        )

    def pan_vectorized(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply panning to a mono sample array with smoothing.

        Args:
            samples: Mono 1D input array.

        Returns:
            Tuple of (left, right) stereo arrays as float32.

        Raises:
            ValueError: If samples is not a 1D array.
        """
        _validate_mono_array(samples)
        n = len(samples)

        left_envelope = self._left_gain_param.get_interpolated_buffer(n)
        right_envelope = self._right_gain_param.get_interpolated_buffer(n)

        return (
            (samples * left_envelope).astype(np.float32),
            (samples * right_envelope).astype(np.float32),
        )


@register_component()
class ModulatedPanner(Panner):
    """Panner with modulated pan position.

    Same as Panner but takes a modulator to dynamically set the pan value.

    **CV Range:** The modulator should output values in range [-1, 1]:
    - -1.0 = hard left
    -  0.0 = center
    -  1.0 = hard right

    Values outside this range are clamped internally.

    Args:
        modulator: Generator that returns pan position values.
        sample_rate: Processing sample rate used for smoothing duration.
        smoothing_time_ms: Pan transition duration in milliseconds. This affects
            scalar/modifier state transitions. Vectorized modulation applies the
            modulator values directly per sample.
    """

    descriptor = ComponentDescriptor(
        name="Panner (Mod)",
        category=ComponentCategory.MODIFIER,
        description="Stereo panner with modulated position",
        fluent_api_name="panner (mod)",
        parameters={
            "modulator": ParameterDescriptor(
                name="modulator",
                default=None,
                description=(
                    "Generator providing pan position values; values are clamped "
                    "to [-1, 1]."
                ),
            ),
            "sample_rate": ParameterDescriptor(
                name="sample_rate",
                default=DEFAULT_SAMPLE_RATE,
                minimum=1.0,
                unit="Hz",
                description="Processing sample rate.",
            ),
            "smoothing_time_ms": ParameterDescriptor(
                name="smoothing_time_ms",
                default=10.0,
                minimum=0.0,
                unit="ms",
                description="Pan smoothing duration.",
            ),
        },
        tags=["modifier", "panner", "modulated", "stereo"],
    )

    def __init__(
        self,
        modulator: Any,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        smoothing_time_ms: float = 10.0,
    ) -> None:
        """Initialize modulated panner.

        Args:
            modulator: Any generator/iterator that returns values in range [-1, 1].
                Values outside the range are clamped.
            sample_rate: Processing sample rate used for smoothing duration.
            smoothing_time_ms: Pan transition duration in milliseconds.

        Raises:
            TypeError: If modulator is None or not iterable.
            ValueError: If sample_rate or smoothing_time_ms is invalid.
        """
        _validate_modulator(modulator)

        super().__init__(
            position=0.0,
            sample_rate=sample_rate,
            smoothing_time_ms=smoothing_time_ms,
        )

        self._modulator_source = modulator
        self.modulator = iter(self._modulator_source)

        logger.debug("ModulatedPanner initialized and modulator started")

    def __iter__(self) -> ModulatedPanner:
        """Re-initialize modulator for iteration."""
        self.modulator = iter(self._modulator_source)
        return self

    def __next__(self) -> float:
        """Get next modulated pan value and update gains.

        Returns:
            Current pan position.
        """
        self.position = next(self.modulator)
        return self.position

    def __call__(
        self, val: float | np.ndarray
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """Apply modulated panning to input value(s).

        For scalar inputs, advances the modulator once and applies panning.
        For array inputs, uses vectorized processing and applies one modulation
        value per sample.

        Args:
            val: Mono scalar input value or mono 1D NumPy array.

        Returns:
            Tuple of (left, right) stereo values or arrays.

        Raises:
            TypeError: If val is not numeric or a NumPy array.
            ValueError: If val is a non-1D NumPy array.
        """
        if isinstance(val, np.ndarray):
            _validate_mono_array(val, name="val")
            mod_values = self._get_modulation_values(len(val))
            return _apply_vectorized_panning(val, mod_values)

        if isinstance(val, (float, int, np.number)):
            mod_value = self._get_next_modulation_value()
            self.position = mod_value
            return super().__call__(val)

        logger.error("Invalid input type for ModulatedPanner: %s", type(val))
        raise TypeError(
            "Input value must be an int, float, numpy number, or mono 1D "
            f"NumPy array. Got {type(val)}"
        )

    def _get_next_modulation_value(self) -> float:
        """Get the next modulation value for scalar processing.

        Returns:
            Next modulation value. The ``position`` setter clamps it to [-1, 1].
        """
        value, self.modulator = _get_next_modulation_value(
            self._modulator_source, self.modulator
        )
        return value

    def _get_modulation_values(self, num_samples: int) -> np.ndarray:
        """Get modulation values for vectorized processing.

        Args:
            num_samples: Number of modulation values to retrieve.

        Returns:
            Array of modulation values clipped to [-1, 1].
        """
        mod_values = _get_modulation_values(
            self._modulator_source, self.modulator, num_samples
        )
        return np.clip(mod_values, -1.0, 1.0)

    def trigger_release(self) -> None:
        """Trigger release on the modulator if supported."""
        if hasattr(self._modulator_source, "trigger_release"):
            self._modulator_source.trigger_release()

    @property
    def ended(self) -> bool:
        """Check whether the modulator has ended, if it exposes that state."""
        if hasattr(self._modulator_source, "ended"):
            return bool(self._modulator_source.ended)
        return False


def apply_vectorized_panning(
    samples: np.ndarray, positions: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply constant-power panning using vectorized operations.

    This public utility can be used by GUI modules to apply panning without
    creating a Panner component instance.

    Args:
        samples: Mono 1D input samples.
        positions: Pan positions in range [-1, 1]. Can be a single float or an
            array matching the length of samples.

    Returns:
        Tuple of (left, right) stereo arrays as float32.

    Raises:
        TypeError: If positions is not numeric or an array.
        ValueError: If samples is not 1D, or if position array length does not
            match the number of samples.
    """
    _validate_mono_array(samples)

    if isinstance(positions, (int, float, np.number)):
        positions_array = np.full(len(samples), float(positions), dtype=np.float32)
    elif isinstance(positions, np.ndarray):
        if positions.ndim != 1:
            raise ValueError(
                "positions must be a scalar or 1D NumPy array; "
                f"got shape {positions.shape}"
            )
        if len(positions) != len(samples):
            raise ValueError(
                "positions length must match samples length; "
                f"got {len(positions)} positions for {len(samples)} samples"
            )
        positions_array = positions.astype(np.float32, copy=False)
    else:
        raise TypeError(
            "positions must be an int, float, numpy number, or 1D NumPy array. "
            f"Got {type(positions)}"
        )

    left_gains, right_gains = _calculate_constant_power_gains(positions_array)

    left = left_gains * samples
    right = right_gains * samples

    return left.astype(np.float32), right.astype(np.float32)


def _apply_vectorized_panning(
    samples: np.ndarray, mod_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy wrapper for apply_vectorized_panning.

    Args:
        samples: Mono 1D input samples.
        mod_values: Pan positions in range [-1, 1].

    Returns:
        Tuple of (left, right) stereo arrays as float32.
    """
    return apply_vectorized_panning(samples, mod_values)
