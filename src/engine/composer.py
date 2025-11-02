"""Signal routing and composition for audio synthesis.

This module provides components for combining multiple audio generators and
effects into complex synthesis patches. Composers enable both serial (Chain)
and parallel (WaveAdder) signal routing patterns.

Classes:
    Composer: Abstract base class for all composers.
    Chain: Serial signal chain for applying multiple modifiers sequentially.
    WaveAdder: Parallel mixer for combining multiple signal generators.

Example:
    >>> from engine import SineOscillator, Volume, Panner
    >>>
    >>> # Serial processing with Chain
    >>> osc = SineOscillator(440)
    >>> chain = Chain(osc, Volume(0.5), Panner(0.7))
    >>> samples = chain.get_samples(1000)
    >>>
    >>> # Parallel mixing with WaveAdder
    >>> osc1 = SineOscillator(440)
    >>> osc2 = SineOscillator(880)
    >>> adder = WaveAdder(osc1, osc2)
    >>> mixed = adder.get_samples(1000)

Signal Flow:
    - Chain: oscillator → modifier1 → modifier2 → ... → output
    - WaveAdder: (osc1 + osc2 + ... + oscN) / N → output

Note:
    Composers support both mono and stereo signal routing, with automatic
    handling of stereo/mono conversion where needed. They also propagate
    trigger_release() and ended properties to all child components.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.modifier import Modifier
from src.engine.modulated_oscillator import ModulatedOscillator
from src.engine.oscillator import Oscillator


class Composer(ABC):

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def get_samples_iterator(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False
    ) -> np.ndarray:
        """Generate n samples using Python iterator.

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            reset: If True, reset the composer to initial state before generating.

        Returns:
            list: List of `n` consecutive samples produced by calling `next(self)`.
        """
        osc = iter(self) if reset else self
        return np.array([next(osc) for _ in range(n)], np.float32)

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using iterator and convert to NumPy array.

        For composers, true vectorization depends on the underlying oscillators.
        This method generates samples via iterator and converts to array.

        Args:
            n: Number of samples to produce.

        Returns:
            np.ndarray: Array of `n` consecutive samples.

        Note:
            To get true vectorized performance, ensure underlying oscillators
            use their vectorized methods.
        """
        return self.get_samples_iterator(n, reset=True)

    def get_samples(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False, mode: str = "auto"
    ) -> np.ndarray:
        """Generate n samples using the specified method.

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            reset: If True, reset the composer to initial state before generating.
            mode: Generation mode. Options:
                - "auto": Automatically choose the best method (vectorized for n >= 512)
                - "iterator": Use Python iterator (returns list)
                - "vectorized": Convert to NumPy array (returns ndarray)

        Returns:
            np.ndarray or list: Generated samples. Type depends on mode.

        Raises:
            ValueError: If mode is not one of "auto", "iterator", or "vectorized".

        Examples:
            >>> from engine import SineOscillator, Volume
            >>>
            >>> chain = Chain(SineOscillator(), Volume(0.5))
            >>> samples1 = chain.get_samples(1000)  # Auto mode
            >>> samples2 = chain.get_samples(100, mode="iterator", reset=True)
        """
        if mode not in ("auto", "iterator", "vectorized"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'auto', 'iterator', or 'vectorized'."
            )

        if mode == "auto":
            mode = "vectorized" if n >= 512 else "iterator"

        if mode == "iterator":
            return self.get_samples_iterator(n, reset=reset)
        else:  # mode == "vectorized"
            if reset:
                iter(self)
            return self.get_samples_vectorized(n)


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
