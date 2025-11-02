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

    Args:
        position: Right pan value. 0=fully left, 1=fully right, 0.5=center. Defaults to 0.5.

    Attributes:
        position: Current right pan value.
    """

    def __init__(self, position: float = 0.5) -> None:
        """Initialize panner with pan position.

        Args:
            position: Pan value, 0 means 100% left panned and 1 means 100% right
                panned, 0.5 is center panned.
        """
        self.position: float = np.clip(position, 0.0, 1.0)
        logger.debug(f"Panner initialized with pan position: {position}")

    def __call__(self, val: float) -> Tuple[float, float]:
        """Convert mono signal to stereo with panning.

        Args:
            val: Mono input value.

        Returns:
            Tuple of (left, right) stereo values.
        """
        right: float = self.position * 2
        left: float = 2 - right
        return left * val, right * val


class ModulatedPanner(Panner):
    """Panner with modulated pan position.

    Same as Panner but takes a modulator to dynamically set the pan value.

    Args:
        modulator: Generator that returns values in range [-1, 1].

    Attributes:
        modulator: The modulator instance.
        r: Current pan position (computed from modulator).
    """

    def __init__(self, modulator: Any) -> None:
        """Initialize modulated panner.

        Args:
            modulator: Any kind of generator that returns a value within the range
                of [-1, 1] this is used to set the `r` value that has a range of [0, 1].
        """
        super().__init__(position=0)
        self.modulator = modulator

        # Auto-initialize the modulator to avoid common errors
        iter(self.modulator)
        logger.debug("ModulatedPanner initialized and modulator started")

    def __iter__(self) -> 'ModulatedPanner':
        """Re-initialize modulator for iteration."""
        iter(self.modulator)
        return self

    def __next__(self) -> float:
        """Get next modulated pan value.

        Returns:
            Current pan position.
        """
        self.position = (next(self.modulator) + 1) / 2
        return self.position


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
