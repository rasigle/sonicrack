"""Panning modifiers for stereo positioning."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    ComponentDescriptor,
    ParameterDescriptor,
)
from src.engine.core.parameter import RuntimeParameter, SmoothingPolicy
from src.engine.core.registry import ComponentCategory, register_component
from src.engine.dsp.modifiers.base import (
    Modifier,
    _get_modulation_values,
    _get_next_modulation_value,
    _validate_modulator,
)
from src.engine.utils.validation import validate_sample_rate

logger = logging.getLogger(__name__)


@register_component()
class Panner(Modifier):
    """Converts mono input into stereo output with configurable pan position.

    Uses constant-power panning law for perceptually uniform panning.
    Position range: -1.0 (hard left) to 1.0 (hard right), 0.0 (center).

    Args:
        position: Pan position. -1.0=left, 0.0=center, 1.0=right. Defaults to 0.0.

    Attributes:
        _left_gain: Precomputed left channel gain.
        _right_gain: Precomputed right channel gain.
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
            position: Pan value, -1.0 means 100% left panned, 1.0 means 100% right
                panned, 0.0 is center panned.
            sample_rate: Processing sample rate used for smoothing duration.
            smoothing_time_ms: Pan transition duration in milliseconds.

        Raises:
            TypeError: If position is not a number.
        """
        super().__init__()
        self.sample_rate = validate_sample_rate(sample_rate)
        self.smoothing_time_ms = smoothing_time_ms

        # Calculate initial gains from position
        position = float(np.clip(position, -1.0, 1.0))
        angle = (position + 1.0) * np.pi / 4.0
        initial_left = float(np.cos(angle))
        initial_right = float(np.sin(angle))

        # Create RuntimeParameters for left/right gains with smoothing
        left_gain_descriptor = ParameterDescriptor(
            name="left_gain",
            default=initial_left,
            minimum=0.0,
            maximum=1.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=smoothing_time_ms,
        )
        self._left_gain_param = RuntimeParameter(left_gain_descriptor, sample_rate)

        right_gain_descriptor = ParameterDescriptor(
            name="right_gain",
            default=initial_right,
            minimum=0.0,
            maximum=1.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=smoothing_time_ms,
        )
        self._right_gain_param = RuntimeParameter(right_gain_descriptor, sample_rate)

        # Store the position value (for property getter)
        self._position = position

    @property
    def position(self) -> float:
        """float: Current pan position (-1.0 to 1.0)."""
        return self._position

    @position.setter
    def position(self, value: float):
        """Set pan position and update gains with smoothing."""
        # Validate and clip position
        value = float(np.clip(value, -1.0, 1.0))
        self._position = value

        # Calculate new gains using constant-power law
        angle = (value + 1.0) * np.pi / 4.0
        new_left = float(np.cos(angle))
        new_right = float(np.sin(angle))

        # Update RuntimeParameters - they will handle smoothing
        self._left_gain_param.value = new_left
        self._right_gain_param.value = new_right

    def _update_gains(self) -> None:
        """Update left/right gains based on position using constant-power law."""
        # This method is now replaced by the position setter
        # Kept for backward compatibility but not used internally
        angle = (self._position + 1.0) * np.pi / 4.0
        new_left = float(np.cos(angle))
        new_right = float(np.sin(angle))
        self._left_gain_param.value = new_left
        self._right_gain_param.value = new_right

    def __call__(
        self, val: float | np.ndarray
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """Convert mono signal to stereo with panning.

        Args:
            val: Mono input value or array.

        Returns:
            Tuple of (left, right) stereo values or arrays.
        """
        # For scalar calls, use target value for instant response (no smoothing)
        # This is important for modulated panning where rapid changes are expected
        left_gain = self._left_gain_param._target_value
        right_gain = self._right_gain_param._target_value

        if isinstance(val, np.ndarray):
            return left_gain * val, right_gain * val

        return left_gain * val, right_gain * val

    def pan_vectorized(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply panning to an array of samples (vectorized with smoothing).

        Args:
            samples: Mono input array.

        Returns:
            Tuple of (left, right) stereo arrays (float32).
        """
        n = len(samples)

        # Get smoothed gain envelopes from RuntimeParameters
        left_envelope = self._left_gain_param.get_interpolated_buffer(n)
        right_envelope = self._right_gain_param.get_interpolated_buffer(n)

        return (
            (samples * left_envelope).astype(np.float32),
            (samples * right_envelope).astype(np.float32),
        )

    # Backward-compatible properties for tests and legacy code
    @property
    def _left_gain(self) -> float:
        """Backward compatibility: current left gain."""
        return self._left_gain_param.value

    @property
    def _right_gain(self) -> float:
        """Backward compatibility: current right gain."""
        return self._right_gain_param.value

    @property
    def _target_left_gain(self) -> float:
        """Backward compatibility: delegate to RuntimeParameter."""
        return self._left_gain_param._target_value

    @property
    def _target_right_gain(self) -> float:
        """Backward compatibility: delegate to RuntimeParameter."""
        return self._right_gain_param._target_value

    @property
    def _current_left_gain(self) -> float:
        """Backward compatibility: delegate to RuntimeParameter."""
        return self._left_gain_param._current_value

    @property
    def _current_right_gain(self) -> float:
        """Backward compatibility: delegate to RuntimeParameter."""
        return self._right_gain_param._current_value

    @property
    def _smoothing_samples_remaining(self) -> int:
        """Backward compatibility: return max of left/right smoothing remaining."""
        return max(
            self._left_gain_param._smoothing_samples_remaining,
            self._right_gain_param._smoothing_samples_remaining,
        )

    @property
    def _smoothing_duration_samples(self) -> int:
        """Backward compatibility: delegate to RuntimeParameter."""
        return self._left_gain_param._smoothing_duration_samples


@register_component()
class ModulatedPanner(Panner):
    """Panner with modulated pan position.

    Same as Panner but takes a modulator to dynamically set the pan value.

    **CV Range:** The modulator should output values in range [-1, 1]:
    - -1.0 = hard left
    -  0.0 = center
    -  1.0 = hard right

    This matches the natural output range of oscillators, so LFOs can be
    used directly without scaling.

    **Note:** If your CV source outputs a different range (e.g., envelope [0, 1]),
    use CVScaler to convert it:
        >>> from src.engine import unipolar_to_bipolar, ADSREnvelope
        >>> env = ADSREnvelope(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
        >>> scaled_env = unipolar_to_bipolar(env)  # Convert [0,1] to [-1,1]
        >>> panner = ModulatedPanner(scaled_env)

    Args:
        modulator: Generator that returns values in range [-1, 1].
                  Values outside this range are clamped internally.

    Attributes:
        modulator: The modulator instance.

    Example:
        >>> from src.engine import SineOscillator, ModulatedPanner, Chain
        >>> # LFO oscillates between -1 and 1, directly controlling pan
        >>> lfo = SineOscillator(4)  # 4 Hz auto-pan
        >>> panner = ModulatedPanner(lfo)
        >>> chain = Chain(SineOscillator(440), panner)
        >>> samples = chain.get_samples(1000)
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
            modulator: Any kind of generator that returns a value within the range
                of [-1, 1]. This value directly sets the pan position:
                -1 = hard left, 0 = center, 1 = hard right.
            sample_rate: Processing sample rate used for smoothing duration.
            smoothing_time_ms: Pan transition duration in milliseconds.

        Raises:
            TypeError: If modulator is None or not iterable.
        """
        _validate_modulator(modulator)

        super().__init__(
            position=0.0,
            sample_rate=sample_rate,
            smoothing_time_ms=smoothing_time_ms,
        )

        # Keep the original modulator object (source) so we can re-create
        # fresh iterators when needed (some modulators are iterable but not
        # iterator objects themselves). Also keep an iterator instance that
        # we advance during normal operation.
        self._modulator_source = modulator
        self.modulator = iter(self._modulator_source)

        logger.debug("ModulatedPanner initialized and modulator started")

    def __iter__(self) -> ModulatedPanner:
        """Re-initialize modulator for iteration."""
        # Re-create the iterator from the original source so iteration
        # always starts fresh. This handles both iterator and iterable
        # modulator implementations.
        self.modulator = iter(self._modulator_source)
        return self

    def __next__(self) -> float:
        """Get next modulated pan value and update gains.

        Returns:
            Current pan position.
        """
        self.position = float(np.clip(next(self.modulator), -1.0, 1.0))
        return self.position

    def __call__(
        self, val: float | np.ndarray
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """Apply modulated panning to input value(s).

        For scalar inputs, advances the modulator once and applies panning.
        For array inputs, uses vectorized processing for optimal performance.

        Args:
            val: Mono input value or array.

        Returns:
            Tuple of (left, right) stereo values or arrays.
        """
        if isinstance(val, np.ndarray):
            # Vectorized path: process entire array at once
            mod_values = self._get_modulation_values(len(val))
            return _apply_vectorized_panning(val, mod_values)

        # Scalar path: advance modulator once and update position
        mod_value = self._get_next_modulation_value()
        self.position = float(np.clip(mod_value, -1.0, 1.0))
        return super().__call__(val)

    def _get_next_modulation_value(self) -> float:
        """Get the next modulation value for scalar processing.

        Returns:
            Next modulation value.
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
            Array of modulation values, clipped to [-1, 1].
        """
        mod_values = _get_modulation_values(
            self._modulator_source, self.modulator, num_samples
        )
        # Clip to valid range for panning
        return np.clip(mod_values, -1.0, 1.0)


def apply_vectorized_panning(
    samples: np.ndarray, positions: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply constant-power panning using vectorized operations.

    This is a public utility function that can be used by GUI modules
    to apply panning without creating a Panner component instance.

    Args:
        samples: Mono input samples.
        positions: Pan positions in range [-1, 1]. Can be a single float
            or an array matching the length of samples.

    Returns:
        Tuple of (left, right) stereo arrays.
    """
    # Ensure positions is an array
    if isinstance(positions, (int, float)):
        positions = np.full(len(samples), positions, dtype=np.float32)

    # Clip to valid range
    positions = np.clip(positions, -1.0, 1.0)

    # Convert to angles [0, π/2] - fully vectorized
    angles = (positions + 1.0) * np.pi / 4.0

    # Calculate gains using constant-power panning law
    left_gains = np.cos(angles)
    right_gains = np.sin(angles)

    # Apply gains
    left = left_gains * samples
    right = right_gains * samples

    return left.astype(np.float32), right.astype(np.float32)


def _apply_vectorized_panning(
    samples: np.ndarray, mod_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy wrapper for apply_vectorized_panning (deprecated).

    Args:
        samples: Mono input samples.
        mod_values: Pan positions in range [-1, 1].

    Returns:
        Tuple of (left, right) stereo arrays.
    """
    return apply_vectorized_panning(samples, mod_values)
