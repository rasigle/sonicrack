"""
Components that have __call__ implemented and whose instances can be used as functions
to alter the output of any kind of generator. Generally used in a Chain component.
"""

from abc import abstractmethod, ABC
from collections.abc import Iterable


class Modifier(ABC):
    """Base class for all modifiers."""

    @abstractmethod
    def __call__(self, val):
        pass


class Panner(Modifier):
    """Will convert a mono input into stereo."""

    def __init__(self, r: float = 0.5):
        """

        Args:
            r : is the right pan value, 0 means 100% left panned and 1 means 100% right
                panned, 0.5 is center panned.
        """
        self.right = r

    def __call__(self, val):
        right = self.right * 2
        left = 2 - right
        return left * val, right * val


class ModulatedPanner(Panner):
    """Same as the Panner but takes in a modulator to set the internal `r` value."""

    def __init__(self, modulator):
        """

        Args:
            modulator : any kind of generator that returns a value within the range
                of [-1, 1] this is used to set the `r` value that has a range of [0, 1].
        """
        super().__init__(r=0)
        self.modulator = modulator

    def __iter__(self):
        iter(self.modulator)
        return self

    def __next__(self):
        self.r = (next(self.modulator) + 1) / 2
        return self.r


class Volume(Modifier):
    """Scales the input values by `amp`, can be used to increase or decrease the
    amplitude."""

    def __init__(self, amp: float = 1.0):
        """
        amp : sets the amplitude multiplier for the
            input signal (1 : no change, 0 : no output).
        """
        self.amp = amp

    def __call__(self, val):
        if isinstance(val, Iterable):
            return tuple(v * self.amp for v in val)

        if isinstance(val, (int, float)):
            return val * self.amp

        raise TypeError("Input value must be an int, float, or Iterable.")


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

    def __iter__(self):
        iter(self.modulator)
        return self

    def __next__(self):
        self.amp = next(self.modulator)
        return self.amp

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

    def __init__(self, freq: float = 1.0):
        """Initializes the Frequency modifier.

        Args:
            freq : sets the amplitude multiplier for the
                input signal (1 : no change, 0 : no output).
        """
        self.freq = freq

    def __call__(self, val):
        if isinstance(val, Iterable):
            return tuple(v * self.freq for v in val)

        if isinstance(val, (int, float)):
            return val * self.freq
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

    def __iter__(self):
        iter(self.modulator)
        return self

    def __next__(self):
        self.freq = next(self.modulator)
        return self.freq

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
        mi, ma = wave_range
        self.mm = lambda v: max(mi, min(ma, v))

    def __call__(self, val):
        if isinstance(val, Iterable):
            return tuple(self.mm(v / 2) * 2 for v in val)

        return self.mm(val)
