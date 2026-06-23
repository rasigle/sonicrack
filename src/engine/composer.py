"""Signal routing and composition for audio synthesis.

This module provides components for combining multiple audio generators and
effects into complex synthesis patches. Composers enable both serial (Chain)
and parallel (WaveAdder) signal routing patterns.

Classes:
    Composer: Abstract base class for all composers.
    Chain: Serial signal chain for applying multiple modifiers sequentially.
    WaveAdder: Parallel mixer for combining multiple signal generators.

Example:
    >>> from src.engine import SineOscillator, Volume, Panner
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
from typing import Any

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.audio_component import AudioComponent, ComponentDescriptor
from src.engine.audio_component_registry import ComponentCategory, register_component
from src.engine.oscillator import Oscillator
from src.engine.oscillator_modulated import ModulatedOscillator
from src.engine.validation import validate_sample_count
from src.utils.logging_config import get_engine_logger

logger = get_engine_logger("composer")


def _get_vectorized_samples(component: Any, n: int) -> np.ndarray:
    """Render a child component through its most direct vectorized entry point."""
    if hasattr(component, "get_samples_vectorized"):
        return component.get_samples_vectorized(n)

    if hasattr(component, "get_samples"):
        return component.get_samples(n, reset=False, mode="vectorized")

    return np.array([next(component) for _ in range(n)], dtype=np.float32)


class Composer(AudioComponent, ABC):
    """Base for components that combine signals (chain, mixer)."""

    def __init__(self, *components: AudioComponent, **kwargs: Any):
        super().__init__(*components, **kwargs)
        self.components = components

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
        n = validate_sample_count(n)
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
            >>> from src.engine import SineOscillator, Volume
            >>>
            >>> chain = Chain(SineOscillator(), Volume(0.5))
            >>> samples1 = chain.get_samples(1000)  # Auto mode
            >>> samples2 = chain.get_samples(100,mode="iterator")
        """
        n = validate_sample_count(n)
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


@register_component()
class Chain(Composer):
    """A component that allows for chaining a single generator with multiple modifiers
    after it.

    For sequential composition of waves.
    """

    descriptor = ComponentDescriptor(
        name="Chain",
        category=ComponentCategory.COMPOSER,
        description="Chains a generator with multiple modifiers in sequence.",
        tags=["composer", "chain"],
    )

    def __init__(self, oscillator, *modifiers):
        """Initialize the Chain.

        Args:
            oscillator: instance of an Oscillator or anything else that can generate a
                sequence of numbers by using __iter__ and __next__.
            modifiers: Modifiers or Effects (both implement __call__).
                Examples: Volume(0.5), Panner(0.7), Distortion(drive=2.0)

        Raises:
            TypeError: If oscillator doesn't implement iterator protocol.
            ValueError: If oscillator is None.

        Example:
            >>> from engine import SineOscillator, Chain, Volume, Distortion, Panner
            >>> # Mix Modifiers seamlessly - all derive from Modifier!
            >>> chain = Chain(
            ...     SineOscillator(440),
            ...     Volume(0.5),
            ...     Distortion(drive=2.0, mix=0.8),
            ...     Panner(0.7)
            ... )
        """
        super().__init__()

        # Input validation
        if oscillator is None:
            raise ValueError("oscillator cannot be None")

        if not (hasattr(oscillator, "__iter__") and hasattr(oscillator, "__next__")):
            raise TypeError(
                f"The given oscillator must implement the iterator protocol "
                f"(`__iter__` and `__next__`). Given: {type(oscillator).__name__}"
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
    def ended(self) -> bool:
        ended = []
        e = "ended"
        if hasattr(self.oscillator, e):
            ended.append(self.oscillator.ended)
        ended.extend([m.ended for m in self.modifiers if hasattr(m, e)])
        return all(ended)

    @ended.setter
    def ended(self, value: bool) -> None:
        self._ended = value

    def __iter__(self):
        iter(self.oscillator)
        for modifier in self.modifiers:
            if type(modifier).__next__ is not AudioComponent.__next__:
                iter(modifier)
        return self

    def __next__(self):
        val = next(self.oscillator)
        for modifier in self.modifiers:
            if type(modifier).__next__ is not AudioComponent.__next__:
                next(modifier)
        for modifier in self.modifiers:
            val = modifier(val)
        return val

    def get_samples_vectorized(self, n: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
        """Generate n samples using fully vectorized processing.

        Applies each modifier using their __call__ method, which handles
        vectorization internally. Modifiers that return tuples (left, right)
        are automatically detected as panners and converted to stereo arrays.

        Args:
            n: Number of samples to produce.

        Returns:
            np.ndarray: Array of n consecutive samples (float32).
                - Mono: shape (n, )
                - Stereo: shape (n, 2)
        """
        n = validate_sample_count(n)
        # Generate samples from oscillator (vectorized)
        samples = _get_vectorized_samples(self.oscillator, n)

        # Apply each modifier in sequence using vectorized methods
        for modifier in self.modifiers:
            # Use the modifier's __call__ method directly
            # This works for all modifiers: Panner, ModulatedPanner, Volume, etc.

            if samples.ndim == 1:
                # Mono input - call modifier
                result = modifier(samples)

                # Check if result is stereo (tuple) - indicates panning
                if isinstance(result, tuple) and len(result) == 2:
                    # Panner/ModulatedPanner returned (left, right)
                    left, right = result
                    samples = np.column_stack((left, right))
                else:
                    # Regular modifier (Volume, etc.) returned modified samples
                    samples = result
            else:
                # Stereo input - apply modifier to left channel
                # (Panners shouldn't receive stereo input, but handle gracefully)
                if modifier.__class__.__name__ == "ButterworthFilter":
                    result = modifier(samples)
                    if isinstance(result, np.ndarray) and result.shape == samples.shape:
                        samples = result
                        continue

                result = modifier(samples[:, 0])
                if isinstance(result, tuple) and len(result) == 2:
                    left, right = result
                    samples = np.column_stack((left, right))
                else:
                    # Apply same modification to both channels
                    samples = np.column_stack((result, modifier(samples[:, 1])))

        return samples.astype(np.float32)


@register_component()
class WaveAdder(Composer):
    """Component that combines the output of multiple generators.

    Supports two mixing modes:
    - 'average': Returns the mean (prevents clipping, default for backward
      compatibility)
    - 'sum': Returns the sum (standard mixer behavior, maintains levels)

    For parallel composition of waves.
    """

    descriptor = ComponentDescriptor(
        name="WaveAdder",
        category=ComponentCategory.COMPOSER,
        description="Adds the output of multiple generators together.",
        tags=["composer", "wave_adder"],
    )

    def __init__(self, *generators, stereo: bool = False, mix_mode: str = "average"):
        """Initialize WaveAdder.

        Args:
            *generators: Instances of generators/oscillators that can generate a
                sequence of numbers by using __iter__ and __next__.
            stereo: if True the output will have a tuple of two numbers for the left
                and the right channel each, else only one number.
            mix_mode: 'average' (default) or 'sum'.
                - 'average': divides by number of generators (prevents clipping)
                - 'sum': direct sum (standard mixer behavior)

        Raises:
            ValueError: If no generators provided or invalid mix_mode.
            TypeError: If stereo is not a boolean.
        """
        super().__init__(*generators)

        # Input validation
        if len(generators) == 0:
            raise ValueError("WaveAdder requires at least one generator")

        if not isinstance(stereo, bool):
            raise TypeError(f"stereo must be a boolean, got {type(stereo).__name__}")

        if mix_mode not in ("average", "sum"):
            raise ValueError(f"mix_mode must be 'average' or 'sum', got {mix_mode!r}")

        self.generators = generators
        self.stereo = stereo
        self.mix_mode = mix_mode

        # Debug logging
        logger.debug(
            f"WaveAdder initialized: {len(generators)} generators, "
            f"stereo={stereo}, mix_mode={mix_mode!r}"
        )

    def _mod_channels(self, _val):
        if isinstance(_val, (int, float)) and self.stereo:
            return _val, _val

        if isinstance(_val, Sequence) and not self.stereo:
            if self.mix_mode == "sum":
                return sum(_val)

            # average
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

    @ended.setter
    def ended(self, value: bool) -> None:
        self._ended = value

    def __iter__(self):
        [iter(gen) for gen in self.generators]
        return self

    def __next__(self):
        vals = [self._mod_channels(next(gen)) for gen in self.generators]
        if self.stereo:
            left_values, right_values = zip(*vals, strict=False)
            if self.mix_mode == "sum":
                return sum(left_values), sum(right_values)
            # average
            return (
                sum(left_values) / len(left_values),
                sum(right_values) / len(right_values),
            )

        if self.mix_mode == "sum":
            return sum(vals)
        # average
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
        n = validate_sample_count(n)
        # Fast path for single generator (no mixing needed)
        if len(self.generators) == 1:
            samples = self.generators[0].get_samples(n, mode="vectorized")

            # Handle stereo conversion if needed
            if self.stereo and samples.ndim == 1:
                # Convert mono to stereo
                samples = np.column_stack((samples, samples))
            elif not self.stereo and samples.ndim == 2:
                # Convert stereo to mono
                samples = samples.mean(axis=1)
            # else: already in correct format (mono->mono or stereo->stereo)

            return samples.astype(np.float32)

        # Generate samples from all generators (vectorized)
        all_samples = []
        for gen in self.generators:
            samples = _get_vectorized_samples(gen, n)
            all_samples.append(samples)

        # Optimize for mono mode (most common case)
        if not self.stereo:
            # Check if all inputs are mono
            all_mono = all(s.ndim == 1 for s in all_samples)

            if all_mono:
                # Pure mono: direct sum or mean (fastest path)
                # Stack all: (n_generators, n) -> sum/mean -> (n,)
                stacked = np.stack(all_samples, axis=0)
                if self.mix_mode == "sum":
                    return stacked.sum(axis=0, dtype=np.float32)
                # average
                return stacked.mean(axis=0, dtype=np.float32)

            else:
                # Mixed mono/stereo: convert stereo to mono, then sum/mean
                mono_samples = []
                for samples in all_samples:
                    if samples.ndim == 2:
                        # Stereo to mono: average channels
                        mono_samples.append(samples.mean(axis=1))
                    else:
                        # Already mono
                        mono_samples.append(samples)

                stacked = np.stack(mono_samples, axis=0)
                if self.mix_mode == "sum":
                    return stacked.sum(axis=0, dtype=np.float32)
                # average
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
        # Stack: (n_generators, n, 2) -> sum/mean -> (n, 2)
        stacked = np.stack(all_samples, axis=0)
        if self.mix_mode == "sum":
            return stacked.sum(axis=0, dtype=np.float32)
        # average
        return stacked.mean(axis=0, dtype=np.float32)
