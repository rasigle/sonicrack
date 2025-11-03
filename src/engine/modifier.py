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

from abc import abstractmethod, ABC
from collections.abc import Iterable
from typing import Any

import numpy as np

from src.utils.logging_config import get_engine_logger

logger = get_engine_logger("modifier")


class Modifier(ABC):
    """Base class for all modifiers."""

    @abstractmethod
    def __call__(self, val: float | tuple[float, ...]) -> float | tuple[float, ...]:
        """Apply modification to a value.

        Args:
            val: Input value (mono float or stereo tuple).

        Returns:
            Modified value (same type as input).
        """
        pass


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

    def __init__(self, position: float = 0.0) -> None:
        """Initialize panner with pan position.

        Args:
            position: Pan value, -1.0 means 100% left panned, 1.0 means 100% right
                panned, 0.0 is center panned.

        Raises:
            TypeError: If position is not a number.
        """
        # Input validation
        if not isinstance(position, (int, float, np.number)):
            raise TypeError(f"position must be a number, got {type(position).__name__}")

        self._position: float = np.clip(position, -1.0, 1.0)

        # Precompute gains
        self._left_gain: float = 0.0
        self._right_gain: float = 0.0
        self._update_gains()

    @property
    def position(self) -> float:
        """float: Current pan position (-1.0 to 1.0)."""
        return self._position

    @position.setter
    def position(self, value: float):
        """Set pan position and update gains."""
        self._position = np.clip(value, -1.0, 1.0)
        self._update_gains()

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
        """Apply panning to an array of samples (vectorized).

        Args:
            samples: Mono input array.

        Returns:
            Tuple of (left, right) stereo arrays (float32).
        """
        left = (self._left_gain * samples).astype(np.float32)
        right = (self._right_gain * samples).astype(np.float32)
        return left, right


class ModulatedPanner(Panner):
    """Panner with modulated pan position.

    Same as Panner but takes a modulator to dynamically set the pan value.
    The modulator should output values in range [-1, 1] for pan position.
    This matches the natural output range of oscillators.

    Args:
        modulator: Generator that returns values in range [-1, 1].
                  -1 = hard left, 0 = center, 1 = hard right.

    Attributes:
        modulator: The modulator instance.

    Example:
        >>> from src.engine import SineOscillator, ModulatedPanner, Chain
        >>> # LFO oscillates between -1 and 1, directly controlling pan
        >>> lfo = SineOscillator(4)  # 4 Hz auto-pan, no wave_range needed!
        >>> panner = ModulatedPanner(lfo)
        >>> chain = Chain(SineOscillator(440), panner)
        >>> samples = chain.get_samples(1000)
    """

    def __init__(self, modulator: Any) -> None:
        """Initialize modulated panner.

        Args:
            modulator: Any kind of generator that returns a value within the range
                of [-1, 1]. This value directly sets the pan position:
                -1 = hard left, 0 = center, 1 = hard right.

        Raises:
            TypeError: If modulator is None or not iterable.
        """
        # Input validation
        if modulator is None:
            raise TypeError("modulator cannot be None")
        if not hasattr(modulator, "__iter__") and not hasattr(modulator, "__next__"):
            raise TypeError(
                f"modulator must be iterable or have __next__, got "
                f"{type(modulator).__name__}"
            )

        super().__init__(position=0.0)
        self.modulator = modulator

        # Auto-initialize the modulator to avoid common errors
        iter(self.modulator)
        logger.debug("ModulatedPanner initialized and modulator started")

    def __iter__(self) -> "ModulatedPanner":
        """Re-initialize modulator for iteration."""
        iter(self.modulator)
        return self

    def __next__(self) -> float:
        """Get next modulated pan value and update gains.

        Returns:
            Current pan position.
        """
        # Property setter handles clipping, no need to clip here
        mod_value = next(self.modulator)
        self.position = mod_value  # Setter clips and updates gains
        return self.position

    def pan_vectorized(
        self, samples: np.ndarray, num_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply modulated panning to an array of samples (fully vectorized).

        Args:
            samples: Mono input array.
            num_samples: Number of samples to process.

        Returns:
            Tuple of (left, right) stereo arrays.
        """
        # Get modulation values vectorized (50-100x faster than loop!)
        if hasattr(self.modulator, "get_samples"):
            mod_values = self.modulator.get_samples(
                num_samples, reset=False, mode="vectorized"
            )
        else:
            # Fallback to iterator if vectorization not available
            mod_values = np.array(
                [next(self.modulator) for _ in range(num_samples)], dtype=np.float32
            )

        # Clip to valid range
        mod_values = np.clip(mod_values, -1.0, 1.0)

        # Convert to angles [0, π/2] - fully vectorized
        angles = (mod_values + 1.0) * np.pi / 4.0

        # Calculate gains - vectorized
        left_gains = np.cos(angles)
        right_gains = np.sin(angles)

        # Apply gains - vectorized
        left = left_gains * samples
        right = right_gains * samples

        return left.astype(np.float32), right.astype(np.float32)


class Volume(Modifier):
    """Scales the input values by amplitude multiplier.

    Can be used to increase or decrease the amplitude of signals.

    Args:
        amplitude: Amplitude multiplier. 1.0=no change, 0.0=silence. Defaults to 1.0.

    Attributes:
        amplitude: Current amplitude multiplier.
    """

    def __init__(self, amplitude: float = 1.0) -> None:
        """Initialize volume modifier.

        Args:
            amplitude: Sets the amplitude multiplier for the
                input signal (1 : no change, 0 : no output).

        Raises:
            TypeError: If amplitude is not a number.
            ValueError: If amplitude is negative.
        """
        # Input validation
        if not isinstance(amplitude, (int, float, np.number)):
            raise TypeError(
                f"amplitude must be a number, got {type(amplitude).__name__}"
            )
        if amplitude < 0:
            raise ValueError(f"amplitude must be non-negative, got {amplitude}")

        self.amplitude: float = amplitude
        logger.debug(f"Volume initialized with amplitude: {amplitude}")

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
        # Optimize: check array first (most common in vectorized code)
        if isinstance(val, np.ndarray):
            return val * self.amplitude

        if isinstance(val, Iterable):
            return tuple(v * self.amplitude for v in val)

        # Accept int, float, and numpy number types
        if isinstance(val, (int, float, np.number)):
            return val * self.amplitude

        logger.error(f"Invalid input type for Volume: {type(val)}")
        raise TypeError(
            "Input value must be an int, float, numpy number, array, or Iterable."
        )

    def scale_vectorized(self, samples: np.ndarray) -> np.ndarray:
        """Apply volume scaling to array of samples (vectorized).

        Args:
            samples: Input array.

        Returns:
            Scaled array (float32).
        """
        return (samples * self.amplitude).astype(np.float32)


class ModulatedVolume(Volume):
    """Same as the volume component but the internal `amp` is set by a modulator."""

    def __init__(self, modulator):
        """Initialize modulated volume.

        Args:
            modulator: Any kind of generator that returns a value within the range
                of [0, max_amp] this is used to set the `amp` value directly.
                If max_amp is > 1 then the amplitude of the input will increase.

        Raises:
            TypeError: If modulator is None or not iterable.
        """
        if modulator is None:
            raise TypeError("modulator cannot be None")
        if not hasattr(modulator, "__iter__") and not hasattr(modulator, "__next__"):
            raise TypeError(
                f"modulator must be iterable or have __next__, "
                f"got {type(modulator).__name__}"
            )

        super().__init__(0.0)
        self.modulator = modulator
        # Auto-initialize the modulator to avoid common errors
        iter(self.modulator)
        logger.debug("ModulatedVolume initialized and modulator started")

    def __iter__(self):
        """Re-initialize modulator for iteration."""
        iter(self.modulator)
        return self

    def __next__(self):
        self.amplitude = next(self.modulator)
        return self.amplitude

    def trigger_release(self):
        if hasattr(self.modulator, "trigger_release"):
            self.modulator.trigger_release()

    @property
    def ended(self):
        if hasattr(self.modulator, "ended"):
            return self.modulator.ended
        return False

    def scale_vectorized(self, samples: np.ndarray) -> np.ndarray:
        """Apply modulated volume scaling to array of samples (vectorized).

        Args:
            samples: Input array.

        Returns:
            Scaled array (float32) with time-varying amplitude.

        Note:
            This enables true vectorization in Chain, avoiding the iterator fallback.
        """
        n = len(samples)

        # Get modulation values for all samples (vectorized)
        if hasattr(self.modulator, "get_samples"):
            mod_values = self.modulator.get_samples(n, reset=False, mode="vectorized")
        else:
            # Fallback to iterator if modulator doesn't have get_samples
            mod_values = np.array(
                [next(self.modulator) for _ in range(n)], dtype=np.float32
            )

        # Apply time-varying amplitude
        return (samples * mod_values).astype(np.float32)


class Frequency(Modifier):
    """Scales the input values by frequency multiplier.

    Can be used to increase or decrease frequency-related values.

    Args:
        frequency: Frequency multiplier. 1.0=no change. Defaults to 1.0.

    Attributes:
        frequency: Current frequency multiplier.
    """

    def __init__(self, frequency: float = 1.0):
        """Initialize frequency modifier.

        Args:
            frequency: Sets the frequency multiplier for the
                input signal (1 : no change, 0 : no output).

        Raises:
            TypeError: If frequency is not a number.
            ValueError: If frequency is negative.
        """
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


class ModulatedFrequency(Frequency):
    """Same as the frequency component but the internal `freq` is set by a modulator."""

    def __init__(self, modulator):
        """Initialize modulated frequency.

        Args:
            modulator: Any kind of generator that returns a value within the range
                of [0, max_freq] this is used to set the frequency value directly.
                If max_freq is > 1 then the frequency of the input will increase.

        Raises:
            TypeError: If modulator is None or not iterable.
        """
        # Input validation
        if modulator is None:
            raise TypeError("modulator cannot be None")
        if not hasattr(modulator, "__iter__") and not hasattr(modulator, "__next__"):
            raise TypeError(
                f"modulator must be iterable or have __next__, got "
                f"{type(modulator).__name__}"
            )

        super().__init__(1.0)
        self.modulator = modulator
        # Auto-initialize the modulator to avoid common errors
        iter(self.modulator)
        logger.debug("ModulatedFrequency initialized and modulator started")

    def __iter__(self):
        """Re-initialize modulator for iteration."""
        iter(self.modulator)
        return self

    def __next__(self):
        self.frequency = next(self.modulator)
        return self.frequency

    def trigger_release(self):
        if hasattr(self.modulator, "trigger_release"):
            self.modulator.trigger_release()

    @property
    def ended(self):
        if hasattr(self.modulator, "ended"):
            return self.modulator.ended
        return False

    def scale_vectorized(self, samples: np.ndarray) -> np.ndarray:
        """Apply modulated frequency scaling to array of samples (vectorized).

        Args:
            samples: Input array.

        Returns:
            Scaled array (float32) with time-varying frequency multiplier.

        Note:
            This enables true vectorization in Chain, avoiding the iterator fallback.
        """
        n = len(samples)

        # Get modulation values for all samples (vectorized)
        if hasattr(self.modulator, "get_samples"):
            mod_values = self.modulator.get_samples(n, reset=False, mode="vectorized")
        else:
            # Fallback to iterator if modulator doesn't have get_samples
            mod_values = np.array(
                [next(self.modulator) for _ in range(n)], dtype=np.float32
            )

        # Apply time-varying frequency multiplier
        return (samples * mod_values).astype(np.float32)


class Clipper(Modifier):
    """Component that clips the input signal to the given wave range.

    Uses NumPy's optimized clip function for fast, vectorized clipping.
    """

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
