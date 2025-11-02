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

    def get_samples_vectorized(self, n: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
        """Generate n samples using vectorized processing.

        Args:
            n: Number of samples to produce.

        Returns:
            np.ndarray: Array of n consecutive samples.
        """
        raise NotImplementedError(
            "Vectorized sample generation not implemented for this Composer."
        )

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

        # mode == "vectorized"
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

        Raises:
            TypeError: If oscillator doesn't implement iterator protocol.
            TypeError: If any modifier is not a Modifier instance.
            ValueError: If oscillator is None.
        """
        # Input validation
        if oscillator is None:
            raise ValueError("oscillator cannot be None")

        if not (hasattr(oscillator, "__iter__") and hasattr(oscillator, "__next__")):
            raise TypeError(
                f"The given oscillator must implement the iterator protocol "
                f"(`__iter__` and `__next__`). Given: {type(oscillator).__name__}"
            )
        if not all([isinstance(m, Modifier) for m in modifiers]):
            raise TypeError(
                f"All given modifiers should be instances of Modifier. "
                f"Given: {[type(mod).__name__ for mod in modifiers]}"
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

    def get_samples_vectorized(self, n: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
        """Generate n samples using fully vectorized processing.

        Applies each modifier using vectorized methods where available,
        providing speedup over iterator approach.

        Args:
            n: Number of samples to produce.

        Returns:
            np.ndarray: Array of n consecutive samples (float32).
                - Mono: shape (n, )
                - Stereo: shape (n, 2)
        """
        # Generate samples from oscillator (vectorized)
        if hasattr(self.oscillator, "get_samples"):
            samples = self.oscillator.get_samples(n, reset=False, mode="vectorized")
        else:
            # Fallback to iterator
            samples = np.array(
                [next(self.oscillator) for _ in range(n)], dtype=np.float32
            )

        # Apply each modifier in sequence using vectorized methods
        for modifier in self.modifiers:
            # Check for vectorized methods first (priority order for performance)

            if hasattr(modifier, "pan_vectorized"):
                # Panner/ModulatedPanner - optimized stereo panning
                if hasattr(modifier, "__next__"):
                    # ModulatedPanner - fully vectorized with modulation
                    left, right = modifier.pan_vectorized(samples, n)
                else:
                    # Static Panner - simple vectorized
                    left, right = modifier.pan_vectorized(samples)
                # Convert to stereo array: shape (n, 2)
                samples = np.column_stack((left, right))

            elif hasattr(modifier, "scale_vectorized"):
                # Volume/Frequency - vectorized scaling
                samples = modifier.scale_vectorized(samples)

            elif hasattr(modifier, "clip_vectorized"):
                # Clipper - vectorized clipping
                samples = modifier.clip_vectorized(samples)

            elif hasattr(modifier, "__call__") and not hasattr(modifier, "__next__"):
                # Static modifier without state - can apply directly to array
                # This handles Volume, Frequency, Clipper if they don't have vectorized
                # methods
                samples = modifier(samples)

            else:
                # Modifier with state (like ModulatedVolume) - need to iterate
                # This is rare and slower, but maintains correctness
                result = []
                for sample in samples:
                    if hasattr(modifier, "__next__"):
                        next(modifier)  # Advance modifier state
                    result.append(modifier(sample))

                # Convert result to appropriate format
                if result and isinstance(result[0], tuple):
                    # Stereo output
                    samples = np.array(result, dtype=np.float32)
                else:
                    # Mono output
                    samples = np.array(result, dtype=np.float32)

        return samples.astype(np.float32)


class WaveAdder(Composer):
    """Component that returns the mean of the output of multiple generators.

    For parallel composition of waves.
    """

    def __init__(self, *generators, stereo=False):
        """Initialize WaveAdder.

        Args:
            *generators: Instances of generators/oscillators that can generate a
                sequence of numbers by using __iter__ and __next__.
            stereo: if True the output will have a tuple of two numbers for the left
                and the right channel each, else only one number.

        Raises:
            ValueError: If no generators provided.
            TypeError: If stereo is not a boolean.
        """
        # Input validation
        if len(generators) == 0:
            raise ValueError("WaveAdder requires at least one generator")

        if not isinstance(stereo, bool):
            raise TypeError(f"stereo must be a boolean, got {type(stereo).__name__}")

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

    def get_samples_vectorized(self, n: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
        """Generate n samples using fully vectorized operations.

        Optimized implementation that avoids unnecessary copies and uses
        efficient NumPy operations for combining signals.

        Args:
            n: Number of samples to produce.

        Returns:
            np.ndarray: Array of n consecutive samples (float32).
                - Mono mode: shape (n,)
                - Stereo mode: shape (n, 2) or (n,) depending on inputs

        Note:
            Achieves ~50-100x speedup over iterator approach through
            vectorized generation and combination.
        """
        # Fast path for single generator (no mixing needed)
        if len(self.generators) == 1:
            gen = self.generators[0]
            if hasattr(gen, "get_samples"):
                samples = gen.get_samples(n, reset=False, mode="vectorized")
            else:
                samples = np.array([next(gen) for _ in range(n)], dtype=np.float32)

            # Handle stereo conversion if needed
            if self.stereo and samples.ndim == 1:
                samples = np.column_stack((samples, samples))
            elif not self.stereo and samples.ndim == 2:
                samples = samples.mean(axis=1)

            return samples.astype(np.float32)

        # Generate samples from all generators (vectorized)
        all_samples = []
        for gen in self.generators:
            if hasattr(gen, "get_samples"):
                samples = gen.get_samples(n, reset=False, mode="vectorized")
            else:
                # Fallback to iterator
                samples = np.array([next(gen) for _ in range(n)], dtype=np.float32)
            all_samples.append(samples)

        # Optimize for mono mode (most common case)
        if not self.stereo:
            # Check if all inputs are mono
            all_mono = all(s.ndim == 1 for s in all_samples)

            if all_mono:
                # Pure mono: direct mean (fastest path)
                # Stack all: (n_generators, n) -> mean -> (n,)
                stacked = np.stack(all_samples, axis=0)
                return stacked.mean(axis=0, dtype=np.float32)
            else:
                # Mixed mono/stereo: convert stereo to mono, then mean
                mono_samples = []
                for samples in all_samples:
                    if samples.ndim == 2:
                        # Stereo to mono: average channels
                        mono_samples.append(samples.mean(axis=1))
                    else:
                        # Already mono
                        mono_samples.append(samples)

                stacked = np.stack(mono_samples, axis=0)
                return stacked.mean(axis=0, dtype=np.float32)

        # Stereo mode
        # Check if we have mixed mono/stereo inputs
        has_mono = any(s.ndim == 1 for s in all_samples)

        if has_mono:
            # Convert mono to stereo only where needed
            stereo_samples = []
            for samples in all_samples:
                if samples.ndim == 1:
                    # Mono to stereo: duplicate channel
                    samples = np.column_stack((samples, samples))
                # Stereo samples pass through
                stereo_samples.append(samples)
            all_samples = stereo_samples

        # All samples now stereo: (n, 2) each
        # Stack: (n_generators, n, 2) -> mean -> (n, 2)
        stacked = np.stack(all_samples, axis=0)
        return stacked.mean(axis=0, dtype=np.float32)
