"""Signal modifiers for audio processing.

This module provides components that modify audio signals through callable objects.
Modifiers can be chained together using the Chain composer to create complex
signal processing pipelines.

Classes:
    Modifier: Abstract base class for all modifiers.
    Panner: Converts mono signals to stereo with configurable pan position.
    ModulatedPanner: Panner with modulated pan position.
    Volume: Scales signal amplitude.
    ModulatedVolume: Volume with modulated amplitude.
    Frequency: Scales frequency-related values.
    ModulatedFrequency: Frequency modifier with modulation.
    Clipper: Clips signals to specified range.

Example:
    >>> from src.engine import SineOscillator, Chain
    >>>
    >>> osc = SineOscillator(440)
    >>> volume = Volume(0.5)
    >>> panner = Panner(0.7)  # Pan right
    >>> chain = Chain(osc, volume, panner)
    >>> samples = chain.get_samples(1000)

Note:
    Modifiers are designed to be used with the Chain composer but can also
    be used standalone by calling them directly with signal values.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Iterable
from numbers import Real
from typing import Any

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.audio_component import (
    AudioComponent,
    ComponentDescriptor,
    ParameterDescriptor,
)
from src.engine.audio_component_registry import ComponentCategory, register_component
from src.engine.oscillator import _derive_amplitude_from_init
from src.engine.validation import validate_sample_rate
from src.utils.logging_config import get_engine_logger
from src.utils.math import db_to_linear, linear_to_db
from src.utils.utils import track_provided_args

logger = get_engine_logger("modifier")


def _smoothing_sample_count(
    sample_rate: float, smoothing_time_ms: float, *, name: str = "smoothing_time_ms"
) -> int:
    """Return a finite non-negative smoothing duration in samples."""
    if isinstance(smoothing_time_ms, bool) or not isinstance(smoothing_time_ms, Real):
        raise TypeError(
            f"{name} must be a real number, got {type(smoothing_time_ms).__name__}"
        )

    smoothing_time_ms = float(smoothing_time_ms)
    if not math.isfinite(smoothing_time_ms) or smoothing_time_ms < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")

    return int(smoothing_time_ms * sample_rate / 1000)


def _validate_modulator(modulator: Any) -> None:
    """Validate that a modulator is usable.

    Args:
        modulator: The modulator to validate.

    Raises:
        TypeError: If modulator is None or not iterable/iterator.
    """
    if modulator is None:
        raise TypeError("modulator cannot be None")
    if not hasattr(modulator, "__iter__") and not hasattr(modulator, "__next__"):
        raise TypeError(
            f"modulator must be iterable or have __next__, got "
            f"{type(modulator).__name__}"
        )


def _get_modulation_values(
    modulator_source: Any, modulator_iter: Any, num_samples: int
) -> np.ndarray:
    """Get modulation values for vectorized processing.

    Tries to get samples from the modulator in the most efficient way:
    1. Call get_samples() on the source (preferred - usually vectorized)
    2. Call get_samples() on the iterator (if available)
    3. Fall back to iterating manually

    Args:
        modulator_source: The original modulator source object.
        modulator_iter: The iterator instance.
        num_samples: Number of modulation values to retrieve.

    Returns:
        Array of modulation values.
    """
    # Prefer calling `get_samples` on the original source if available
    if hasattr(modulator_source, "get_samples"):
        return modulator_source.get_samples(num_samples, mode="vectorized")

    if hasattr(modulator_iter, "get_samples"):
        # Iterator might itself expose get_samples
        return modulator_iter.get_samples(num_samples, mode="vectorized")

    # Fallback to iterator if vectorization not available
    return np.array(
        [next(modulator_iter) for _ in range(num_samples)], dtype=np.float32
    )


def _get_next_modulation_value(
    modulator_source: Any, modulator_iter: Any
) -> tuple[float, Any]:
    """Get the next modulation value for scalar processing.

    Handles iterator exhaustion by recreating the iterator from the source.

    Args:
        modulator_source: The original modulator source object.
        modulator_iter: The current iterator instance.

    Returns:
        Tuple of (next_value, potentially_new_iterator).
    """
    try:
        return next(modulator_iter), modulator_iter
    except StopIteration:
        # Re-create iterator and advance
        new_iter = iter(modulator_source)
        return next(new_iter), new_iter


class Modifier(AudioComponent):
    """Base for components that modify signals (effects, filters)."""

    @abstractmethod
    def __call__(self, val: Any) -> Any:
        """Apply modification to a value or a bunch of values.

        Args:
            val: Input value (mono float or stereo tuple).

        Returns:
            Modified value (same type as input).
        """
        pass


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
                clamp=True,
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

        # Input validation
        if not isinstance(position, (int, float, np.number)):
            raise TypeError(f"position must be a number, got {type(position).__name__}")

        self._position: float = np.clip(position, -1.0, 1.0)

        # Precompute gains
        self._left_gain: float = 0.0
        self._right_gain: float = 0.0
        self._update_gains()

        # Pan smoothing to prevent clicks when changing pan position
        self._target_left_gain = self._left_gain
        self._target_right_gain = self._right_gain
        self._current_left_gain = self._left_gain  # Start at current position
        self._current_right_gain = self._right_gain  # Start at current position
        self._smoothing_samples_remaining = 0
        self._smoothing_duration_samples = _smoothing_sample_count(
            self.sample_rate, self.smoothing_time_ms
        )

    @property
    def position(self) -> float:
        """float: Current pan position (-1.0 to 1.0)."""
        return self._position

    @position.setter
    def position(self, value: float):
        """Set pan position and update gains with smoothing."""
        self._position = np.clip(value, -1.0, 1.0)
        self._update_gains()

        # Initiate smooth transition (prevents clicks)
        self._target_left_gain = self._left_gain
        self._target_right_gain = self._right_gain
        self._smoothing_samples_remaining = self._smoothing_duration_samples

    def _update_gains(self) -> None:
        """Update left/right gains based on position using constant-power law."""
        # Convert position from [-1, 1] to angle [0, π/2]
        # -1.0 -> 0 (all left), 0.0 -> π/4 (center), 1.0 -> π/2 (all right)
        angle = (self._position + 1.0) * np.pi / 4.0
        self._left_gain = np.cos(angle)
        self._right_gain = np.sin(angle)

    def __call__(
        self, val: float | np.ndarray
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """Convert mono signal to stereo with panning.

        Args:
            val: Mono input value or array.

        Returns:
            Tuple of (left, right) stereo values or arrays.
        """
        if isinstance(val, np.ndarray):
            return self._left_gain * val, self._right_gain * val

        return self._left_gain * val, self._right_gain * val

    def pan_vectorized(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply panning to an array of samples (vectorized with smoothing).

        Args:
            samples: Mono input array.

        Returns:
            Tuple of (left, right) stereo arrays (float32).
        """
        n = len(samples)

        # Apply pan gains with smoothing if transitioning (prevents clicks!)
        if self._smoothing_samples_remaining > 0:
            # Calculate how many samples to smooth in this buffer
            smooth_count = min(n, self._smoothing_samples_remaining)

            # Create smooth gain envelopes (linear ramp)
            left_envelope = np.linspace(
                self._current_left_gain, self._target_left_gain, smooth_count
            )
            right_envelope = np.linspace(
                self._current_right_gain, self._target_right_gain, smooth_count
            )

            # Apply smoothed gains to first part
            left = np.zeros(n, dtype=np.float32)
            right = np.zeros(n, dtype=np.float32)
            left[:smooth_count] = samples[:smooth_count] * left_envelope
            right[:smooth_count] = samples[:smooth_count] * right_envelope

            # Apply target gains to rest (if any)
            if smooth_count < n:
                left[smooth_count:] = samples[smooth_count:] * self._target_left_gain
                right[smooth_count:] = samples[smooth_count:] * self._target_right_gain

            # Update state
            self._smoothing_samples_remaining -= smooth_count
            if self._smoothing_samples_remaining <= 0:
                self._current_left_gain = self._target_left_gain
                self._current_right_gain = self._target_right_gain
            else:
                # Update current gain to end of ramp for next buffer
                self._current_left_gain = left_envelope[-1]
                self._current_right_gain = right_envelope[-1]

            return left.astype(np.float32), right.astype(np.float32)
        else:
            # No smoothing needed - use target gains (which match
            # _left_gain/_right_gain)
            left = (self._target_left_gain * samples).astype(np.float32)
            right = (self._target_right_gain * samples).astype(np.float32)
            return left, right


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
        >>> from engine import unipolar_to_bipolar, ADSREnvelope
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
        # Property setter handles clipping, no need to clip here
        self.position = next(self.modulator)  # Setter clips and updates gains
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
        self.position = mod_value
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


@register_component()
class Volume(Modifier):
    """Volume control modifier with support for linear and dB gain.

    Scales the input signal by an amplitude multiplier. Supports both
    linear amplitude control and professional decibel (dB) gain control.

    **Amplitude vs. Gain (dB):**

    - Use **gain_db** for audio work (professional standard)
      * 0 dB = unity gain (no change)
      * -6 dB = half amplitude
      * -20 dB = 1/10 amplitude
      * -∞ dB = silence

    - Use **amplitude** for direct linear control
      * 1.0 = unity gain (no change)
      * 0.5 = half amplitude
      * 0.0 = silence

    **If both gain_db and amplitude are specified:**
    gain_db takes priority. A warning is logged if they don't match.

    Args:
        amplitude: Linear amplitude multiplier. Default: 1.0
            Note: Ignored if gain_db is specified.
        gain_db: Gain in decibels. Default: None (uses amplitude)
            Overrides amplitude if provided.

    Example:
        >>> # Using dB control (recommended for audio)
        >>> vol = Volume(gain_db=-6)  # -6 dB reduction
        >>> vol.gain_db = 0  # Unity gain
        >>>
        >>> # Using linear amplitude
        >>> vol2 = Volume(amplitude=0.5)  # Half amplitude
        >>> vol2.amplitude = 1.0  # Full amplitude
    """

    descriptor = ComponentDescriptor(
        name="Volume",
        category=ComponentCategory.MODIFIER,
        description="Volume control modifier with dB support",
        fluent_api_name="volume",
        parameters={
            "amplitude": ParameterDescriptor(
                name="amplitude",
                default=1.0,
                minimum=0.0,
                description="Linear gain multiplier.",
            ),
            "gain_db": ParameterDescriptor(
                name="gain_db",
                default=DEFAULT_GAIN_DB,
                unit="dB",
                description="Gain in decibels.",
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
                description="Gain smoothing duration.",
            ),
        },
        tags=["modifier", "gain_db", "volume", "amplitude", "gain", "db"],
    )

    @track_provided_args
    def __init__(
        self,
        amplitude: float = 1.0,
        gain_db: float | None = DEFAULT_GAIN_DB,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        smoothing_time_ms: float = 10.0,
    ) -> None:
        """Initialize volume modifier.

        Args:
            amplitude: Amplitude multiplier (1.0 = no change, 0.0 = silence).
                Ignored if gain_db is specified.
            gain_db: Gain in decibels (0 dB = no change, -∞ dB = silence).
                Overrides amplitude if provided.

        Raises:
            TypeError: If amplitude is not a number.
            ValueError: If amplitude is negative.
        """
        super().__init__()
        self.sample_rate = validate_sample_rate(sample_rate)
        self.smoothing_time_ms = smoothing_time_ms

        # Input validation
        if not isinstance(amplitude, (int, float, np.number)):
            raise TypeError(
                f"amplitude must be a number, got {type(amplitude).__name__}"
            )
        if amplitude < 0:
            raise ValueError(f"amplitude must be non-negative, got {amplitude}")

        self._amplitude = _derive_amplitude_from_init(
            self._provided_args, amplitude, gain_db  # noqa
        )

        # Amplitude smoothing to prevent clicks when changing gain
        self._target_amplitude = self._amplitude
        self._current_amplitude = self._amplitude
        self._smoothing_samples_remaining = 0
        self._smoothing_duration_samples = _smoothing_sample_count(
            self.sample_rate, self.smoothing_time_ms
        )

        logger.debug(f"Volume initialized with amplitude: {self._amplitude}")

    @property
    def amplitude(self) -> float:
        """float: Current amplitude multiplier (linear scale).

        For audio work, consider using the gain_db property instead.

        Returns the target amplitude (the value you set), not the smoothed value.
        """
        return self._target_amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        if value < 0:
            raise ValueError(f"amplitude must be non-negative, got {value}")
        # Initiate smooth transition (prevents clicks)
        self._target_amplitude = float(value)
        self._smoothing_samples_remaining = self._smoothing_duration_samples
        # Don't update _amplitude immediately - let smoothing handle it
        # self._amplitude = float(value)  # REMOVED - causes hot-swap to fail!

    @property
    def gain_db(self) -> float:
        """float: Current gain in decibels (professional audio standard).

        Common dB values:
            0 dB = unity gain (no change)
            -6 dB = half amplitude
            -20 dB = 1/10 amplitude
            -∞ dB = silence

        Returns the target gain (the value you set), not the smoothed value.
        """
        return linear_to_db(self._target_amplitude)

    @gain_db.setter
    def gain_db(self, value: float):
        new_amplitude = float(db_to_linear(value))
        # Initiate smooth transition (prevents clicks)
        self._target_amplitude = new_amplitude
        self._smoothing_samples_remaining = self._smoothing_duration_samples
        # Don't update _amplitude immediately - let smoothing handle it
        # self._amplitude = new_amplitude  # REMOVED - causes hot-swap to fail!

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Apply volume scaling to input.

        Args:
            val: Input value (mono float, stereo tuple, or array).

        Returns:
            Scaled value (same type as input).

        Raises:
            TypeError: If input is not int, float, numpy array, or Iterable.
        """
        # Scalar input
        if isinstance(val, (float, int, np.number)):
            # Apply amplitude with smoothing if transitioning (prevents clicks!)
            if self._smoothing_samples_remaining > 0:
                # Create smooth amplitude envelope (linear ramp)
                amp_envelope = np.linspace(
                    self._current_amplitude,
                    self._target_amplitude,
                    self._smoothing_samples_remaining,
                )

                # Apply smoothed amplitude to single sample
                result = val * amp_envelope[0]

                # Update state
                self._smoothing_samples_remaining -= 1
                if self._smoothing_samples_remaining <= 0:
                    self._current_amplitude = self._target_amplitude
                    self._amplitude = self._target_amplitude  # Update _amplitude too!

                return float(result)
            else:
                # No smoothing needed - direct multiplication
                return float(val * self._amplitude)

        # Vectorized input
        if isinstance(val, (tuple, np.ndarray, Iterable)):
            return self._scale_vectorized(np.asarray(val))

        logger.error(f"Invalid input type for Volume: {type(val)}")
        raise TypeError(
            f"Input value must be an int, float, numpy number, array, or Iterable. "
            f"Got {type(val)}"
        )

    def _scale_vectorized(self, samples: np.ndarray) -> np.ndarray:
        """Apply volume scaling to array of samples (vectorized).

        Args:
            samples: Input array.

        Returns:
            Scaled array (float32).
        """
        n = len(samples)

        # Apply amplitude with smoothing if transitioning (prevents clicks!)
        if self._smoothing_samples_remaining > 0:
            # Calculate how many samples to smooth in this buffer
            smooth_count = min(n, self._smoothing_samples_remaining)

            # Create smooth amplitude envelope (linear ramp)
            amp_envelope = np.linspace(
                self._current_amplitude, self._target_amplitude, smooth_count
            )

            # Apply smoothed amplitude to first part
            result = np.zeros(n, dtype=np.float32)
            result[:smooth_count] = samples[:smooth_count] * amp_envelope

            # Apply target amplitude to rest (if any)
            if smooth_count < n:
                result[smooth_count:] = samples[smooth_count:] * self._target_amplitude

            # Update state
            self._smoothing_samples_remaining -= smooth_count
            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude
                self._amplitude = self._target_amplitude  # Update _amplitude too!

            return result.astype(np.float32)
        else:
            # No smoothing needed - direct multiplication
            return (samples * self._amplitude).astype(np.float32)


@register_component()
class ModulatedVolume(Volume):
    """Volume control with time-varying modulation (tremolo, auto-gain, etc.).

    This component inherits all dB and amplitude control from Volume,
    but the amplitude is dynamically controlled by a modulator component
    (e.g., LFO, envelope) for time-varying effects.

    **Inherited features from Volume:**
    - gain_db property (professional dB control)
    - amplitude property (linear control)
    - db_to_linear() / linear_to_db() utilities

    **Modulation & CV Range:**
    - **For amplitude modulation:** CV should be in range [0, max_amplitude]
      (e.g., [0, 1] for normal volume control)
    - **For gain_db modulation:** CV should be in dB range (e.g., [-60, 12])

    **Note:** If your CV source outputs a different range:
        >>> from src.engine import bipolar_to_unipolar
        >>> lfo = SineOscillator(2)  # Output: [-1, 1]
        >>> scaled_lfo = bipolar_to_unipolar(lfo)  # Output: [0, 1]
        >>> volume = ModulatedVolume(scaled_lfo)

    Args:
        modulator: Generator that produces amplitude/gain values.
            - For amplitude (default): [0.0, max] range
            - For gain_db: dB range (e.g., [-60, 12])
            Examples: LFO for tremolo, ADSR for envelope shaping.

    Example:
        >>> from engine import SineOscillator, ADSREnvelope
        >>> # Tremolo effect with LFO
        >>> lfo = SineOscillator(frequency=5, amplitude=0.5, gain_db=None)
        >>> tremolo = ModulatedVolume(lfo)
        >>>
        >>> # Envelope shaping
        >>> env = ADSREnvelope(attack=0.1, decay=0.2, sustain=0.7, release=0.3)
        >>> shaped = ModulatedVolume(env)
    """

    descriptor = ComponentDescriptor(
        name="Volume (Mod)",
        category=ComponentCategory.MODIFIER,
        description="Time-varying volume control with modulation support",
        fluent_api_name="volume (mod)",
        tags=["modifier", "volume", "amplitude", "modulation", "tremolo", "envelope"],
    )

    def __init__(
        self,
        modulator,
        modulation_target: str = "amplitude",
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        smoothing_time_ms: float = 10.0,
    ):
        """Initialize modulated volume.

        Args:
            modulator: Any kind of generator that returns a value within the range
                of [0, max_amp] for amplitude or dB range for gain_db.
                If max_amp is > 1 then the amplitude of the input will increase.
            modulation_target: What to modulate - either "amplitude" or "gain_db".
                - "amplitude": Modulator output directly sets linear amplitude (default)
                - "gain_db": Modulator output sets gain in decibels
                Defaults to "amplitude".

        Raises:
            TypeError: If modulator is None or not iterable.
            ValueError: If modulation_target is not "amplitude" or "gain_db".
        """
        _validate_modulator(modulator)

        # Validate modulation_target
        if modulation_target not in ("amplitude", "gain_db"):
            raise ValueError(
                f"modulation_target must be 'amplitude' or 'gain_db', "
                f"got '{modulation_target}'"
            )

        super().__init__(
            0.0,
            sample_rate=sample_rate,
            smoothing_time_ms=smoothing_time_ms,
        )

        # Keep the original modulator object (source) so we can re-create
        # fresh iterators when needed. Also keep an iterator instance.
        self._modulator_source = modulator
        self.modulator = iter(self._modulator_source)
        self._modulation_target = modulation_target

        logger.debug(
            f"ModulatedVolume initialized with modulation_target={modulation_target}"
        )

    def __iter__(self):
        """Re-initialize modulator for iteration."""
        # Re-create the iterator from the original source
        self.modulator = iter(self._modulator_source)
        return self

    def __next__(self):
        """Get next modulated value and update volume.

        Returns:
            Current amplitude (always returns linear amplitude regardless of target).
        """
        mod_value = next(self.modulator)
        if self._modulation_target == "gain_db":
            self.gain_db = mod_value
        else:
            self.amplitude = mod_value
        return self.amplitude

    def trigger_release(self):
        """Trigger release on modulator if supported."""
        if hasattr(self.modulator, "trigger_release"):
            self.modulator.trigger_release()

    @property
    def ended(self):
        """Check if modulator has ended."""
        if hasattr(self._modulator_source, "ended"):
            return self._modulator_source.ended
        return False

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Apply modulated volume to input value(s).

        For scalar inputs, advances the modulator once and applies volume.
        For array inputs, uses vectorized processing for optimal performance.

        Args:
            val: Input value (mono float, stereo tuple, or array).

        Returns:
            Volume-scaled value (same type as input).
        """
        if isinstance(val, np.ndarray):
            # Vectorized path: process entire array at once (50-100x faster)
            mod_values = self._get_modulation_values(len(val))
            return self._apply_vectorized_volume(val, mod_values)

        # Scalar path: advance modulator once and update amplitude or gain_db
        mod_value = self._get_next_modulation_value()
        if self._modulation_target == "gain_db":
            self.gain_db = mod_value
        else:
            self.amplitude = mod_value
        return super().__call__(val)

    def _get_modulation_values(self, num_samples: int) -> np.ndarray:
        """Get modulation values for vectorized processing.

        Args:
            num_samples: Number of modulation values to retrieve.

        Returns:
            Array of modulation (amplitude) values.
        """
        return _get_modulation_values(
            self._modulator_source, self.modulator, num_samples
        )

    def _apply_vectorized_volume(
        self, samples: np.ndarray, mod_values: np.ndarray
    ) -> np.ndarray:
        """Apply time-varying volume using vectorized operations.

        Args:
            samples: Input samples.
            mod_values: Modulation values (amplitude or gain_db depending on target).

        Returns:
            Volume-modulated samples.
        """
        if self._modulation_target == "gain_db":
            # Convert dB values to linear amplitude
            amplitude_values = np.asarray(db_to_linear(mod_values), dtype=np.float32)
            return (samples * amplitude_values).astype(np.float32)
        else:
            # Direct amplitude modulation
            return (samples * mod_values).astype(np.float32)

    def _get_next_modulation_value(self) -> float:
        """Get the next modulation value for scalar processing.

        Returns:
            Next amplitude value.
        """
        value, self.modulator = _get_next_modulation_value(
            self._modulator_source, self.modulator
        )
        return value


@register_component()
class Frequency(Modifier):
    """Scales the input values by frequency multiplier.

    Can be used to increase or decrease frequency-related values.

    Args:
        frequency: Frequency multiplier. 1.0=no change. Defaults to 1.0.

    Attributes:
        frequency: Current frequency multiplier.
    """

    descriptor = ComponentDescriptor(
        name="Frequency",
        category=ComponentCategory.MODIFIER,
        description="Frequency scaling modifier",
        fluent_api_name="frequency",
        tags=["modifier", "frequency"],
    )

    def __init__(self, frequency: float = 1.0):
        """Initialize frequency modifier.

        Args:
            frequency: Sets the frequency multiplier for the
                input signal (1 : no change, 0 : no output).

        Raises:
            TypeError: If frequency is not a number.
            ValueError: If frequency is negative.
        """
        super().__init__()

        # Input validation
        if not isinstance(frequency, (int, float, np.number)):
            raise TypeError(
                f"frequency must be a number, got {type(frequency).__name__}"
            )
        if frequency < 0:
            raise ValueError(f"frequency must be non-negative, got {frequency}")

        self.frequency = frequency
        logger.debug(f"Frequency initialized with multiplier: {frequency}")

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Apply frequency scaling to input.

        Args:
            val: Input value (float, tuple, or array).

        Returns:
            Scaled value (same type as input).

        Raises:
            TypeError: If input is not int, float, array, or Iterable.
        """
        # Optimize: check array first
        if isinstance(val, np.ndarray):
            return val * self.frequency

        if isinstance(val, Iterable):
            return tuple(v * self.frequency for v in val)

        if isinstance(val, (int, float, np.number)):
            return val * self.frequency

        raise TypeError("Input value must be an int, float, numpy array, or Iterable.")

    def scale_vectorized(self, samples: np.ndarray) -> np.ndarray:
        """Apply frequency scaling to array of samples (vectorized).

        Args:
            samples: Input array.

        Returns:
            Scaled array (float32).
        """
        return (samples * self.frequency).astype(np.float32)


@register_component()
class Clipper(Modifier):
    """Component that clips the input signal to the given wave range.

    Uses NumPy's optimized clip function for fast, vectorized clipping.
    """

    descriptor = ComponentDescriptor(
        name="Clipper",
        category=ComponentCategory.MODIFIER,
        description="Audio clipper/limiter",
        parameters={
            "wave_range": ParameterDescriptor(
                name="wave_range",
                default=(-1.0, 1.0),
                description="Clipping range as minimum and maximum values.",
            )
        },
        fluent_api_name="clipper",
        tags=["modifier", "clipper", "limiter"],
    )

    def __init__(self, wave_range: tuple[float, float] = (-1.0, 1.0)):
        """Initialize clipper with wave range.

        Args:
            wave_range: tuple of (min, max) values which are used to clip the input
                signal.

        Raises:
            TypeError: If wave_range is not a tuple or doesn't have 2 elements.
            ValueError: If min >= max.
        """
        # Input validation
        if not isinstance(wave_range, (tuple, list)):
            raise TypeError(
                f"wave_range must be a tuple or list, got {type(wave_range).__name__}"
            )
        if len(wave_range) != 2:
            raise ValueError(
                f"wave_range must have exactly 2 elements, got {len(wave_range)}"
            )

        min_val, max_val = wave_range

        if not isinstance(min_val, (int, float, np.number)):
            raise TypeError(
                f"wave_range min must be a number, got {type(min_val).__name__}"
            )
        if not isinstance(max_val, (int, float, np.number)):
            raise TypeError(
                f"wave_range max must be a number, got {type(max_val).__name__}"
            )
        if min_val >= max_val:
            raise ValueError(
                f"wave_range min ({min_val}) must be less than max ({max_val})"
            )

        self._min, self._max = float(min_val), float(max_val)
        logger.debug(f"Clipper initialized with range: ({self._min}, {self._max})")

    @property
    def wave_range(self) -> tuple[float, float]:
        """tuple[float, float]: Current clipping range (min, max)."""
        return self._min, self._max

    @wave_range.setter
    def wave_range(self, value: tuple[float, float]):
        """Set clipping range and update min/max values."""
        self._min, self._max = value

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Clip input value(s) to range.

        Args:
            val: Input value (float, tuple, or array).

        Returns:
            Clipped value (same type as input).
        """
        if isinstance(val, np.ndarray):
            # Vectorized clipping (fastest)
            return np.clip(val, self._min, self._max)

        if isinstance(val, Iterable):
            # Tuple clipping
            return tuple(np.clip(v, self._min, self._max) for v in val)

        # Scalar clipping
        return np.clip(val, self._min, self._max)

    def clip_vectorized(self, samples: np.ndarray) -> np.ndarray:
        """Clip array of samples (vectorized).

        Args:
            samples: Input array.

        Returns:
            Clipped array (float32).
        """
        return np.clip(samples, self._min, self._max).astype(np.float32)


@register_component()
class ModulatedClipper(Modifier):
    """Clipper with modulated threshold.

    The modulator controls the clipping threshold symmetrically.

    **CV Range:** The modulator should output values in range [0, 1]:
    - 0.0 = tight clipping (maximum distortion)
    - 1.0 = no clipping (clean signal)

    **Note:** If your CV source outputs a different range (e.g., oscillator [-1, 1]),
    use CVScaler to convert it:
        >>> from engine import bipolar_to_unipolar, SineOscillator
        >>> lfo = SineOscillator(2)  # Output: [-1, 1]
        >>> scaled_lfo = bipolar_to_unipolar(lfo)  # Output: [0, 1]
        >>> clipper = ModulatedClipper(scaled_lfo)

    Example:
        >>> lfo = SineOscillator(2, amplitude=0.5)  # Output: [-0.5, 0.5]
        >>> # Need to scale to [0, 1]
        >>> from src.engine import scale_cv
        >>> scaled = scale_cv(lfo, from_range=(-0.5, 0.5), to_range=(0, 1))
        >>> clipper = ModulatedClipper(scaled)
    """

    descriptor = ComponentDescriptor(
        name="ModulatedClipper",
        category=ComponentCategory.MODIFIER,
        description="Clipper with CV threshold control (CV range: [0, 1])",
        fluent_api_name="modulated_clipper",
        tags=["modifier", "clipper", "modulation"],
    )

    def __init__(self, modulator):
        """Initialize modulated clipper.

        Args:
            modulator: Component that provides threshold modulation values.
                **Expected range: [0, 1]**
                - 0.0 = maximum clipping
                - 1.0 = no clipping

        Raises:
            TypeError: If modulator doesn't implement iterator protocol

        Note:
            The modulator output is clamped to [0, 1] internally, so values
            outside this range will be clipped. For best results, use CVScaler
            to properly scale your CV source.
        """
        if not (hasattr(modulator, "__iter__") and hasattr(modulator, "__next__")):
            raise TypeError(
                f"modulator must be iterable or have __next__, "
                f"got {type(modulator).__name__}"
            )

        self._modulator_source = modulator
        self.modulator = iter(modulator)

        logger.debug("ModulatedClipper initialized (CV range: [0, 1])")

    def __iter__(self):
        """Reset iterator."""
        self.modulator = iter(self._modulator_source)
        return self

    def __next__(self):
        """Get next modulation value and advance modulator."""
        return next(self.modulator)

    def trigger_release(self):
        """Trigger release on modulator if supported."""
        if hasattr(self._modulator_source, "trigger_release"):
            self._modulator_source.trigger_release()

    @property
    def ended(self):
        """Check if modulator has ended."""
        if hasattr(self._modulator_source, "ended"):
            return self._modulator_source.ended
        return False

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Clip input using modulated threshold.

        Args:
            val: Input value (mono float, stereo tuple, or array)

        Returns:
            Clipped value (same type as input)
        """
        if isinstance(val, np.ndarray):
            # Vectorized path
            mod_values = _get_modulation_values(
                self._modulator_source, self.modulator, len(val)
            )
            # Clamp modulation to [0, 1] and use as threshold
            thresholds = np.clip(mod_values, 0.0, 1.0)

            # Clip sample by sample (since threshold varies)
            result = np.empty_like(val)
            for i in range(len(val)):
                thresh = thresholds[i]
                result[i] = np.clip(val[i], -thresh, thresh)

            return result.astype(np.float32)

        # Scalar path
        mod_value = next(self.modulator)
        threshold = np.clip(mod_value, 0.0, 1.0)

        if isinstance(val, Iterable):
            # Stereo tuple
            return tuple(float(np.clip(v, -threshold, threshold)) for v in val)

        # Mono scalar
        return float(np.clip(val, -threshold, threshold))


def _apply_vectorized_panning(
    samples: np.ndarray, mod_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply constant-power panning using vectorized operations.

    Args:
        samples: Mono input samples.
        mod_values: Pan positions in range [-1, 1].

    Returns:
        Tuple of (left, right) stereo arrays.
    """
    # Convert to angles [0, π/2] - fully vectorized
    angles = (mod_values + 1.0) * np.pi / 4.0

    # Calculate gains using constant-power panning law
    left_gains = np.cos(angles)
    right_gains = np.sin(angles)

    # Apply gains
    left = left_gains * samples
    right = right_gains * samples

    return left.astype(np.float32), right.astype(np.float32)
