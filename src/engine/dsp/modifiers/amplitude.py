"""Amplitude modifiers (volume, clipping)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    ComponentDescriptor,
    ParameterDescriptor,
)
from src.engine.core.parameter import AutomationMode, RuntimeParameter, SmoothingPolicy
from src.engine.core.registry import ComponentCategory, register_component
from src.engine.dsp.modifiers.base import (
    Modifier,
    _get_modulation_values,
    _get_next_modulation_value,
    _validate_modulator,
)
from src.engine.generators.oscillators.oscillator import _derive_amplitude_from_init
from src.engine.utils.decorators import track_provided_args
from src.engine.utils.math import db_to_linear, linear_to_db
from src.engine.utils.validation import validate_sample_rate

logger = logging.getLogger(__name__)


def apply_vectorized_gain(
    samples: np.ndarray, gain_values: np.ndarray | float, is_db: bool = False
) -> np.ndarray:
    """Apply gain to samples using vectorized operations.

    This is a public utility function that can be used by GUI modules
    to apply gain without creating a Volume component instance.

    Args:
        samples: Input samples (mono or stereo).
        gain_values: Gain values to apply. Can be a single float or an array
            matching the length of samples.
        is_db: If True, gain_values are in dB and will be converted to linear.
            If False, gain_values are linear multipliers.

    Returns:
        Samples with gain applied (float32).
    """
    # Convert dB to linear if needed
    if is_db:
        if isinstance(gain_values, (int, float)):
            gain_values = db_to_linear(gain_values)
        else:
            gain_values = db_to_linear(gain_values)

    # Ensure gain_values is an array
    if isinstance(gain_values, (int, float)):
        result = samples * gain_values
    else:
        # Handle multi-dimensional samples (e.g., stereo)
        if samples.ndim > 1:
            gain_values = gain_values.reshape(-1, *([1] * (samples.ndim - 1)))
        result = samples * gain_values

    return result.astype(np.float32)


def apply_vectorized_clip(
    samples: np.ndarray, threshold_values: np.ndarray | float
) -> np.ndarray:
    """Apply clipping to samples using vectorized operations.

    This is a public utility function that can be used by GUI modules
    to apply clipping without creating a Clipper component instance.

    Args:
        samples: Input samples (mono or stereo).
        threshold_values: Threshold values (positive). Can be a single float
            or an array matching the length of samples. Samples will be
            clipped to [-threshold, +threshold].

    Returns:
        Clipped samples (float32).
    """
    # Ensure threshold is positive
    if isinstance(threshold_values, (int, float)):
        threshold_values = abs(threshold_values)
        return np.clip(samples, -threshold_values, threshold_values).astype(np.float32)
    else:
        threshold_values = np.abs(threshold_values)
        # For array thresholds, clip each sample independently
        return np.clip(samples, -threshold_values, threshold_values).astype(np.float32)


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
                smoothing_policy=SmoothingPolicy.LINEAR,
                smoothing_duration_ms=10.0,
                automation_mode=AutomationMode.AUDIO_RATE,
            ),
            "gain_db": ParameterDescriptor(
                name="gain_db",
                default=DEFAULT_GAIN_DB,
                unit="dB",
                description="Gain in decibels.",
                smoothing_policy=SmoothingPolicy.LOGARITHMIC,
                smoothing_duration_ms=10.0,
                automation_mode=AutomationMode.CONTROL_RATE,
            ),
            "sample_rate": ParameterDescriptor(
                name="sample_rate",
                default=DEFAULT_SAMPLE_RATE,
                minimum=1.0,
                unit="Hz",
                description="Processing sample rate.",
                smoothing_policy=SmoothingPolicy.NONE,
                automation_mode=AutomationMode.NONE,
            ),
            "smoothing_time_ms": ParameterDescriptor(
                name="smoothing_time_ms",
                default=10.0,
                minimum=0.0,
                unit="ms",
                description="Gain smoothing duration.",
                smoothing_policy=SmoothingPolicy.NONE,
                automation_mode=AutomationMode.CONTROL_RATE,
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

        # Determine initial amplitude
        if not isinstance(amplitude, (int, float, np.number)):
            raise TypeError(
                f"Amplitude must be a number, got {type(amplitude).__name__}"
            )
        if amplitude < 0:
            raise ValueError(f"Amplitude must be non-negative, got {amplitude}")

        initial_amplitude = _derive_amplitude_from_init(
            self._provided_args,
            amplitude,
            gain_db,
        )

        # Create RuntimeParameters with appropriate smoothing policies
        # amplitude: LINEAR smoothing in amplitude space
        amplitude_descriptor = ParameterDescriptor(
            name="amplitude",
            default=initial_amplitude,
            minimum=0.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=smoothing_time_ms,
            automation_mode=AutomationMode.AUDIO_RATE,
        )
        self._amplitude_param = RuntimeParameter(amplitude_descriptor, sample_rate)

        # gain_db: LOGARITHMIC smoothing in dB space
        gain_db_descriptor = ParameterDescriptor(
            name="gain_db",
            default=linear_to_db(initial_amplitude),
            smoothing_policy=SmoothingPolicy.LOGARITHMIC,
            smoothing_duration_ms=smoothing_time_ms,
            automation_mode=AutomationMode.CONTROL_RATE,
        )
        self._gain_db_param = RuntimeParameter(gain_db_descriptor, sample_rate)

        # Track which parameter is active for smoothing
        self._active_param = "amplitude"  # or 'gain_db'

        # Store smoothing_time_ms for backward compatibility
        self.smoothing_time_ms = smoothing_time_ms

        logger.debug(f"Volume initialized with amplitude: {initial_amplitude}")

    @property
    def _smoothing_samples_remaining(self) -> int:
        """Backward compatibility: return smoothing samples from active parameter."""
        if self._active_param == "amplitude":
            return self._amplitude_param._smoothing_samples_remaining
        else:
            return self._gain_db_param._smoothing_samples_remaining

    @property
    def _smoothing_duration_samples(self) -> int:
        """Backward compatibility: return smoothing duration in samples."""
        if self._active_param == "amplitude":
            return self._amplitude_param._smoothing_duration_samples
        else:
            return self._gain_db_param._smoothing_duration_samples

    @property
    def _target_amplitude(self) -> float:
        """Backward compatibility: return target amplitude."""
        if self._active_param == "amplitude":
            return self._amplitude_param.target
        else:
            return db_to_linear(self._gain_db_param.target)

    @property
    def _current_amplitude(self) -> float:
        """Backward compatibility: return current amplitude."""
        if self._active_param == "amplitude":
            return self._amplitude_param.value
        else:
            return db_to_linear(self._gain_db_param.value)

    @property
    def amplitude(self) -> float:
        """float: Current amplitude multiplier (linear scale).

        For audio work, consider using the gain_db property instead.

        Returns the target amplitude (the value you set), not the smoothed value.
        """
        return self._amplitude_param.target

    @amplitude.setter
    def amplitude(self, value: float):
        if value < 0:
            raise ValueError(f"amplitude must be non-negative, got {value}")

        # Set amplitude parameter (triggers LINEAR smoothing)
        self._amplitude_param.value = value
        # Update gain_db to stay in sync (no smoothing trigger)
        self._gain_db_param._target_value = linear_to_db(value)
        self._gain_db_param._current_value = linear_to_db(self._amplitude_param.value)
        self._active_param = "amplitude"

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
        return self._gain_db_param.target

    @gain_db.setter
    def gain_db(self, value: float):
        # Set gain_db parameter (triggers LOGARITHMIC smoothing in dB space)
        self._gain_db_param.value = value
        # Update amplitude to stay in sync (no smoothing trigger)
        new_amplitude = db_to_linear(value)
        self._amplitude_param._target_value = new_amplitude
        self._amplitude_param._current_value = db_to_linear(self._gain_db_param.value)
        self._active_param = "gain_db"

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
            # Get current amplitude (with smoothing if active)
            amp = self._get_current_amplitude()
            return float(val * amp)

        # Vectorized input
        if isinstance(val, (tuple, np.ndarray, Iterable)):
            return self._scale_vectorized(np.asarray(val))

        logger.error(f"Invalid input type for Volume: {type(val)}")
        raise TypeError(
            f"Input value must be an int, float, numpy number, array, or Iterable. "
            f"Got {type(val)}"
        )

    def _get_current_amplitude(self) -> float:
        """Get current amplitude value, handling smoothing."""
        if self._active_param == "amplitude":
            return self._amplitude_param.value
        else:  # gain_db
            return db_to_linear(self._gain_db_param.value)

    def _scale_vectorized(self, samples: np.ndarray) -> np.ndarray:
        """Apply volume scaling to array of samples (vectorized).

        Args:
            samples: Input array.

        Returns:
            Scaled array (float32).
        """
        n = len(samples)

        # Check if smoothing is active
        if self._active_param == "amplitude" and self._amplitude_param.is_smoothing:
            # Get amplitude envelope (LINEAR smoothing)
            amp_envelope = self._amplitude_param.get_interpolated_buffer(n)
        elif self._active_param == "gain_db" and self._gain_db_param.is_smoothing:
            # Get gain_db envelope (LOGARITHMIC smoothing) and convert to amplitude
            gain_db_envelope = self._gain_db_param.get_interpolated_buffer(n)
            amp_envelope = db_to_linear(gain_db_envelope)
        else:
            # No smoothing - use constant value
            amp = self._get_current_amplitude()
            return (samples * amp).astype(np.float32)

        # Apply envelope
        if samples.ndim > 1:
            amp_envelope = amp_envelope.reshape(-1, *([1] * (samples.ndim - 1)))

        result = samples * amp_envelope
        return result.astype(np.float32)


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
        >>> from src.engine import SineOscillator, ADSREnvelope
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
        parameters={
            "modulator": ParameterDescriptor(
                name="modulator",
                default=None,
                description="Generator providing amplitude or gain values.",
            ),
            "modulation_target": ParameterDescriptor(
                name="modulation_target",
                default="amplitude",
                choices=("amplitude", "gain_db"),
                description="Volume parameter controlled by the modulator.",
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
            if samples.ndim > 1:
                amplitude_values = amplitude_values.reshape(
                    -1, *([1] * (samples.ndim - 1))
                )
            return (samples * amplitude_values).astype(np.float32)

        # Direct amplitude modulation
        if samples.ndim > 1:
            mod_values = mod_values.reshape(-1, *([1] * (samples.ndim - 1)))
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

    def __init__(
        self, wave_range: tuple[float, float] = (-1.0, 1.0), *args: Any, **kwargs: Any
    ):
        """Initialize clipper with wave range.

        Args:
            wave_range: tuple of (min, max) values which are used to clip the input
                signal.

        Raises:
            TypeError: If wave_range is not a tuple or doesn't have 2 elements.
            ValueError: If min >= max.
        """
        # Input validation
        super().__init__(*args, **kwargs)
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
        >>> from src.engine import bipolar_to_unipolar, SineOscillator
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
        parameters={
            "modulator": ParameterDescriptor(
                name="modulator",
                default=None,
                minimum=0.0,
                maximum=1.0,
                clamp=True,
                description=(
                    "Generator providing clipping threshold values; values are "
                    "clamped to [0, 1]."
                ),
            )
        },
        tags=["modifier", "clipper", "modulation"],
    )

    def __init__(self, modulator, *args: Any, **kwargs: Any):
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
        super().__init__(*args, **kwargs)
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
            val: Input value: mono float, stereo tuple, or NumPy array.

        Returns:
            Clipped value with the same shape/type category as input.
            Arrays are returned as np.float32.
        """
        if isinstance(val, np.ndarray):
            mod_values = _get_modulation_values(
                self._modulator_source,
                self.modulator,
                len(val),
            )

            thresholds = np.clip(mod_values, 0.0, 1.0)

            # Fully vectorized clipping.
            # For 1D audio: val.shape == (n,)
            # For multi-channel audio: val.shape == (n, channels), so expand threshold
            # dims.
            if val.ndim > 1:
                thresholds = thresholds.reshape(-1, *([1] * (val.ndim - 1)))

            return np.clip(val, -thresholds, thresholds).astype(np.float32, copy=False)

        mod_value = next(self.modulator)
        threshold = float(np.clip(mod_value, 0.0, 1.0))

        if isinstance(val, tuple):
            # Stereo tuple
            return tuple(float(np.clip(v, -threshold, threshold)) for v in val)

        # Mono scalar
        return float(np.clip(val, -threshold, threshold))
