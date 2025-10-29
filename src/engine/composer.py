"""Composers combine oscillators and modifiers to generate waves of different kinds."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from constants import DEFAULT_SAMPLE_RATE
from engine.modifier import Modifier
from engine.modulated_oscillator import ModulatedOscillator
from engine.oscillator import Oscillator


class Composer(ABC):

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def get_samples(self, n: int = DEFAULT_SAMPLE_RATE, it: bool = False):
        """Return the next n samples from this composer component."""
        osc = iter(self) if it else self
        return [next(osc) for _ in range(n)]


class Chain(Composer):
    """A component that allows for chaining a single generator with multiple modifiers
    after it.

    For sequential composition of waves.
    """

    def __init__(self, oscillator, *modifiers):
        """Initialize the Chain.

        Args:
            oscillator: instance of an Oscillator or anything else that can generate a
                sequence of numbers by using __iter__ and __next__.
            modifiers: Any function that takes in a value modifies it and returns
                another value. Example: instances of Panner.
        """
        if not (hasattr(oscillator, "__iter__") and hasattr(oscillator, "__next__")):
            raise TypeError(
                f"The given oscillator must implement the iterator protocol "
                f"(`__iter__` and `__next__`). Given: {type(oscillator)}"
            )
        if not all([isinstance(m, Modifier) for m in modifiers]):
            raise TypeError(
                f"All given modifiers should be instances of Modifier. "
                f"Given: {[type(mod) for mod in modifiers]}"
            )
        self.oscillator: Oscillator | ModulatedOscillator = oscillator
        self.modifiers = modifiers

        iter(self)

    def __getattr__(self, attr):
        if hasattr(self.oscillator, attr):
            return getattr(self.oscillator, attr)

        for modifier in self.modifiers:
            if hasattr(modifier, attr):
                return getattr(modifier, attr)

        raise AttributeError(f"attribute '{attr}' does not exist")

    def trigger_release(self):
        tr = "trigger_release"
        if hasattr(self.oscillator, tr):
            self.oscillator.trigger_release()

        for modifier in self.modifiers:
            if hasattr(modifier, tr):
                modifier.trigger_release()

    @property
    def ended(self):
        ended = []
        e = "ended"
        if hasattr(self.oscillator, e):
            ended.append(self.oscillator.ended)
        ended.extend([m.ended for m in self.modifiers if hasattr(m, e)])
        return all(ended)

    def __iter__(self):
        iter(self.oscillator)
        [iter(mod) for mod in self.modifiers if hasattr(mod, "__iter__")]
        return self

    def __next__(self):
        val = next(self.oscillator)
        [next(mod) for mod in self.modifiers if hasattr(mod, "__iter__")]
        for modifier in self.modifiers:
            val = modifier(val)
        return val


class WaveAdder(Composer):
    """
    Component that returns the mean of the output of multiple generators.

    For parallel composition of waves.
    """

    def __init__(self, *generators, stereo=False):
        """
        Args:
            generator : instance of an Oscillator or anything else that can generate a
                sequence of numbers by using __iter__ and __next__.
            stereo : if True the output will have a tuple of two numbers for the left
                and the right channel each, else only one number.
        """
        self.generators = generators
        self.stereo = stereo

    def _mod_channels(self, _val):
        if isinstance(_val, (int, float)) and self.stereo:
            return _val, _val

        if isinstance(_val, Sequence) and not self.stereo:
            return sum(_val) / len(_val)
        return _val

    def trigger_release(self):
        for gen in self.generators:
            if hasattr(gen, "trigger_release"):
                gen.trigger_release()

    @property
    def ended(self):
        ended = [gen.ended for gen in self.generators if hasattr(gen, "ended")]
        return all(ended)

    def __iter__(self):
        [iter(gen) for gen in self.generators]
        return self

    def __next__(self):
        vals = [self._mod_channels(next(gen)) for gen in self.generators]
        if self.stereo:
            l, r = zip(*vals)
            return sum(l) / len(l), sum(r) / len(r)

        return sum(vals) / len(vals)
