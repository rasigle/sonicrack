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
    >>> from engine import SineOscillator, Chain
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
from typing import Union, Tuple, Any

import numpy as np
from librosa import frequency_weighting

from src.utils.logging_config import get_engine_logger

logger = get_engine_logger("modifier")


class Modifier(ABC):
    """Base class for all modifiers."""

    @abstractmethod
    def __call__(self, val: Union[float, Tuple[float, ...]]) -> Union[float, Tuple[float, ...]]:
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
        position: Current pan position (-1.0 to 1.0).
        _left_gain: Precomputed left channel gain.
        _right_gain: Precomputed right channel gain.
    """

    def __init__(self, position: float = 0.0) -> None:
        """Initialize panner with pan position.

        Args:
            position: Pan value, -1.0 means 100% left panned, 1.0 means 100% right
                panned, 0.0 is center panned.
        """
        self.position: float = np.clip(position, -1.0, 1.0)
        self._update_gains()
        logger.debug(f"Panner initialized with pan position: {position}")

    def _update_gains(self) -> None:
        """Update left/right gains based on position using constant-power law."""

        # Convert position from [-1, 1] to angle [0, π/2]
        # -1.0 -> 0 (all left), 0.0 -> π/4 (center), 1.0 -> π/2 (all right)
        angle = (self.position + 1.0) * np.pi / 4.0
        self._left_gain = np.cos(angle)
        self._right_gain = np.sin(angle)

    def __call__(self, val: Union[float, np.ndarray]) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """Convert mono signal to stereo with panning.

        Args:
            val: Mono input value or array.

        Returns:
            Tuple of (left, right) stereo values or arrays.
        """
        if isinstance(val, np.ndarray):
            return self._left_gain * val, self._right_gain * val

        return self._left_gain * val, self._right_gain * val

    def pan_vectorized(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply panning to an array of samples (vectorized).

        Args:
            samples: Mono input array.

        Returns:
            Tuple of (left, right) stereo arrays.
        """
        return self._left_gain * samples, self._right_gain * samples


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
        >>> from engine import SineOscillator, ModulatedPanner, Chain
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
        """
        super().__init__(position=0.0)
        self.modulator = modulator

        # Auto-initialize the modulator to avoid common errors
        iter(self.modulator)
        logger.debug("ModulatedPanner initialized and modulator started")

    def __iter__(self) -> 'ModulatedPanner':
        """Re-initialize modulator for iteration."""
        iter(self.modulator)
        return self

    def __next__(self) -> float:
        """Get next modulated pan value and update gains.

        Returns:
            Current pan position.
        """
        # Use modulator output directly as pan position [-1, 1]
        mod_value = next(self.modulator)
        self.position = np.clip(mod_value, -1.0, 1.0)
        self._update_gains()
        return self.position

    def pan_vectorized(self, samples: np.ndarray, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply modulated panning to an array of samples (vectorized).

        Args:
            samples: Mono input array.
            num_samples: Number of samples to process.

        Returns:
            Tuple of (left, right) stereo arrays.
        """
        # Get modulation values for all samples (already in [-1, 1] range)
        mod_values = np.array([next(self) for _ in range(num_samples)])

        # Convert to angles [0, π/2]
        angles = (mod_values + 1.0) * np.pi / 4.0

        # Calculate gains
        left_gains = np.cos(angles)
        right_gains = np.sin(angles)

        # Apply gains
        left = left_gains * samples
        right = right_gains * samples

        return left, right


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
        """
        self.amplitude: float = amplitude
        logger.debug(f"Volume initialized with amplitude: {amplitude}")

    def __call__(self, val: Union[float, Tuple[float, ...]]) -> Union[float, Tuple[float, ...]]:
        """Apply volume scaling to input.

        Args:
            val: Input value (mono float or stereo tuple).

        Returns:
            Scaled value (same type as input).

        Raises:
            TypeError: If input is not int, float, or Iterable.
        """
        if isinstance(val, Iterable):
            return tuple(v * self.amplitude for v in val)

        # Accept int, float, and numpy number types
        if isinstance(val, (int, float, np.number)):
            return val * self.amplitude

        logger.error(f"Invalid input type for Volume: {type(val)}")
        raise TypeError("Input value must be an int, float, numpy number, or Iterable.")


class ModulatedVolume(Volume):
    """Same as the volume component but the internal `amp` is set by a modulator."""

    def __init__(self, modulator):
        """
        Args:
            modulator: Any kind of generator that returns a value within the range
                of [0, max_amp] this is used to set the `amp` value directly.
                If max_amp is > 1 then the amplitude of the input will increase.
        """
        super().__init__(0.0)
        self.modulator = modulator
        # Auto-initialize the modulator to avoid common errors
        iter(self.modulator)

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


class Frequency(Modifier):
    """Scales the input values by `amp`, can be used to increase or decrease the
    amplitude."""

    def __init__(self, frequency: float = 1.0):
        """Initializes the Frequency modifier.

        Args:
            frequency : sets the amplitude multiplier for the
                input signal (1 : no change, 0 : no output).
        """
        self.frequency = frequency

    def __call__(self, val):
        if isinstance(val, Iterable):
            return tuple(v * self.frequency for v in val)

        if isinstance(val, (int, float)):
            return val * self.frequency
        raise TypeError("Input value must be an int, float, or Iterable.")


class ModulatedFrequency(Frequency):
    """Same as the frequency component but the internal `freq` is set by a modulator."""

    def __init__(self, modulator):
        """
        Args:
            modulator: Any kind of generator that returns a value within the range
                of [0, max_amp] this is used to set the `amp` value directly.
                If max_amp is > 1 then the amplitude of the input will increase.
        """
        super().__init__(1.0)
        self.modulator = modulator
        # Auto-initialize the modulator to avoid common errors
        iter(self.modulator)

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


class Clipper(Modifier):
    """Component that clips the input signal to the given wave range."""

    def __init__(self, wave_range: tuple[float, float] = (-1.0, 1.0)):
        """

        Args:
            wave_range: tuple of (min, max) values which are used to clip the input
                signal.
        """
        self.range = wave_range
        mi, ma = wave_range
        self.mm = lambda v: max(mi, min(ma, v))

    def __call__(self, val):
        if isinstance(val, Iterable):
            return tuple(self.mm(v) for v in val)

        return self.mm(val)
