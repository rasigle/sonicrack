"""Signal routing and composition for audio synthesis.

This module provides components for combining multiple audio generators and
effects into complex synthesis patches. Composers enable both serial (Chain)
and parallel (WaveAdder) signal routing patterns.

Classes:
    Composer: Abstract base class for all composers.
    Chain: Serial signal chain for applying multiple modifiers sequentially.
    WaveAdder: Parallel mixer for combining multiple signal sources.

Example:
    >>> from src.engine import SineOscillator, SquareOscillator, Volume, Panner
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
    >>>
    >>> # Multi-channel mixer: WaveAdder accepts any signal source that
    >>> # implements the iterator protocol (__iter__/__next__), not just raw
    >>> # oscillators. Since Chain also implements that protocol, each Chain
    >>> # below acts as one fully-processed channel (its own volume/pan),
    >>> # and WaveAdder sums them into a final mix:
    >>> mixer = WaveAdder(
    ...     Chain(SineOscillator(440), Volume(0.8), Panner(0.2)),
    ...     Chain(SquareOscillator(220), Volume(1.0), Panner(0.8)),
    ...     stereo=True,
    ...     mix_mode="sum",
    ... )
    >>> mix = mixer.get_samples(44100)

Signal Flow:
    - Chain: source → modifier1 → modifier2 → ... → output
    - WaveAdder: combines source outputs according to mix_mode:
        - 'average' (default): (source1 + source2 + ... + sourceN) / N
        - 'sum': source1 + source2 + ... + sourceN
      A "source" here is anything implementing the iterator protocol
      (__iter__/__next__) — a bare oscillator, or a full Chain with its own
      modifiers, or even another WaveAdder (mixers can be nested). A lone
      Modifier (e.g. Volume(0.5) on its own) does NOT qualify, since
      modifiers implement __call__ to transform an incoming value rather
      than __next__ to produce one on their own; wrap it in a Chain first
      if you want per-channel processing before mixing.

Note:
    Composers support both mono and stereo signal routing, with automatic
    handling of stereo/mono conversion where needed. They also propagate
    trigger_release() and ended properties to all child components.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from src.constants import AUTO_MODE_VECTORIZE_THRESHOLD, DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    AudioComponent,
    ComponentDescriptor,
    ParameterDescriptor,
)
from src.engine.core.registry import ComponentCategory, register_component
from src.engine.core.sample_mode import VALID_SAMPLE_MODES, SampleMode
from src.engine.generators.oscillators.oscillator import Oscillator
from src.engine.generators.oscillators.oscillator_modulated import ModulatedOscillator
from src.engine.utils.validation import validate_sample_count

if TYPE_CHECKING:
    from src.engine.dsp.modifiers import Modifier

logger = logging.getLogger(__name__)


def _get_vectorized_samples(component: AudioComponent, n: int) -> np.ndarray:
    """Render a child component through its most direct vectorized entry point."""
    if hasattr(component, "get_samples_vectorized"):
        return component.get_samples_vectorized(n)

    if hasattr(component, "get_samples"):
        return component.get_samples(n, reset=False, mode="vectorized")

    return np.array([next(component) for _ in range(n)], dtype=np.float32)


def _process_modifier_block(modifier: Modifier, samples: np.ndarray) -> Any:
    """Apply a modifier through the standardized block API when available."""
    if hasattr(modifier, "process_block"):
        return modifier.process_block(samples)
    return modifier(samples)


def _stereo_tuple_to_array(result: tuple) -> np.ndarray | None:
    """Convert ``(left, right)`` block results to an ``(n, 2)`` array.

    Contract: stereo-producing modifiers/generators must return a 2-tuple
    of (left, right) arrays/sequences, not a list or other 2-length
    sequence. This is the same convention used by ``_mix_stereo`` below.
    """
    if isinstance(result, tuple) and len(result) == 2:
        left, right = result
        return np.column_stack((left, right))
    return None


def _apply_modifier_to_buffer(modifier: Modifier, samples: np.ndarray) -> np.ndarray:
    """Apply one modifier to mono or stereo sample buffers.

    Note: this uses a try/except on process_block as a form of capability
    detection ("does this modifier accept a full stereo (n, 2) buffer?").
    That means a genuine bug inside a modifier's own process_block that
    happens to raise TypeError/ValueError will be silently reinterpreted
    as "doesn't support stereo blocks" rather than surfaced. If modifiers
    grow more complex, prefer an explicit capability flag (e.g.
    `modifier.supports_stereo_block`) over inferring it from exceptions.
    """
    if samples.ndim == 1:
        result = _process_modifier_block(modifier, samples)
        stereo_result = _stereo_tuple_to_array(result)
        if stereo_result is not None:
            return stereo_result
        return np.asarray(result)

    try:
        result = _process_modifier_block(modifier, samples)
    except (TypeError, ValueError):
        result = None

    if isinstance(result, np.ndarray) and result.shape == samples.shape:
        return result

    left_result = _process_modifier_block(modifier, samples[:, 0])
    stereo_left = _stereo_tuple_to_array(left_result)
    if stereo_left is not None:
        return stereo_left

    right_result = _process_modifier_block(modifier, samples[:, 1])
    return np.column_stack((left_result, right_result))


def _validate_iterable_generator(source: Any, label: str = "signal source") -> None:
    """Validate that a component implements the iterator protocol.

    Shared by Chain and WaveAdder so both fail fast with a clear error
    instead of surfacing an obscure AttributeError deep inside __next__.

    Note: "source" here means anything implementing __iter__/__next__ —
    a raw oscillator, a Chain (oscillator + modifiers), or another
    WaveAdder. A bare Modifier alone does not qualify, since modifiers
    implement __call__ (transform a value) rather than __next__ (produce
    one); see the module docstring's "Signal Flow" section.
    """
    if source is None:
        raise ValueError(f"{label} cannot be None")
    if not (hasattr(source, "__iter__") and hasattr(source, "__next__")):
        raise TypeError(
            f"The given {label} must implement the iterator protocol "
            f"(`__iter__` and `__next__`). Given: {type(source).__name__}. "
            f"If you're trying to mix in a Modifier (e.g. Volume, Panner) "
            f"on its own, wrap it in a Chain with a source oscillator first."
        )


class Composer(AudioComponent, ABC):
    """Base for components that combine signals (chain, mixer)."""

    def __init__(self, *components: AudioComponent, **kwargs: Any):
        super().__init__(*components, **kwargs)
        self.components = components
        # Explicit override for `ended`; None means "not overridden, compute
        # from children". See the `ended` property/setter on subclasses.
        self._ended: bool | None = None

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
        self,
        n: int = DEFAULT_SAMPLE_RATE,
        reset: bool = False,
        mode: SampleMode = "auto",
    ) -> np.ndarray:
        """Generate n samples using the specified method.

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            reset: If True, reset the composer to initial state before generating.
            mode: Generation mode. Options:
                - "auto": Automatically choose the best method (vectorized for large n)
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
        if mode not in VALID_SAMPLE_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'auto', 'iterator', or 'vectorized'."
            )

        if mode == "auto":
            mode = "vectorized" if n >= AUTO_MODE_VECTORIZE_THRESHOLD else "iterator"

        if mode == "iterator":
            return self.get_samples_iterator(n, reset=reset)

        # mode == "vectorized"
        if reset:
            iter(self)
        return self.get_samples_vectorized(n)


@register_component()
class Chain(Composer):
    """A component that allows for chaining a single signal source with multiple
    modifiers after it.

    For sequential composition of waves. The leading argument (``oscillator``)
    can itself be any iterator-protocol signal source — a raw oscillator, a
    nested Chain, or a WaveAdder mix — not only a plain oscillator.
    """

    descriptor = ComponentDescriptor(
        name="Chain",
        category=ComponentCategory.COMPOSER,
        fluent_api_name="chain",
        description="Chains a signal source with multiple modifiers in sequence.",
        tags=["composer", "chain"],
    )

    def __init__(self, oscillator, *modifiers):
        """Initialize the Chain.

        Args:
            oscillator: instance of an Oscillator, or any other signal source
                that can generate a sequence of numbers via __iter__ and
                __next__ (including a Chain or WaveAdder).
            modifiers: Modifiers or Effects (both implement __call__).
                Examples: Volume(0.5), Panner(0.7), Distortion(drive=2.0)

        Raises:
            TypeError: If oscillator doesn't implement iterator protocol.
            ValueError: If oscillator is None.

        Example:
            >>> from src.engine import SineOscillator, Chain, Volume, Distortion, Panner
            >>> # Mix Modifiers seamlessly - all derive from Modifier!
            >>> chain = Chain(
            ...     SineOscillator(440),
            ...     Volume(0.5),
            ...     Distortion(drive=2.0, mix=0.8),
            ...     Panner(0.7)
            ... )
        """
        # Input validation (before assigning self.oscillator, since
        # __getattr__ below depends on that attribute already existing).
        _validate_iterable_generator(oscillator, label="oscillator")

        # Forward oscillator + modifiers to Composer so self.components
        # reflects the actual children (matches WaveAdder's behavior).
        super().__init__(oscillator, *modifiers)

        self.oscillator: Oscillator | ModulatedOscillator = oscillator
        self.modifiers = modifiers

        iter(self)

    def __getattr__(self, attr):
        # Guard against infinite recursion: __getattr__ only runs when normal
        # lookup fails, so if `oscillator`/`modifiers` themselves are missing
        # (e.g. accessed before __init__ finishes, during unpickling, etc.),
        # `self.oscillator` below would re-trigger __getattr__ on the same
        # attribute and recurse forever.
        if attr in ("oscillator", "modifiers"):
            raise AttributeError(attr)

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
        if self._ended is not None:
            return self._ended
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

        Applies each modifier through its block-processing API when available.
        Modifiers that return tuples (left, right) are automatically detected as
        panners and converted to stereo arrays.

        Note: modifiers with custom __next__ are stepped per-sample in the
        iterator path (see __next__ above) but have no equivalent per-sample
        step here; this path relies on process_block being fully
        self-consistent with whatever state __next__ advances. Keep this in
        mind when adding a new stateful modifier.

        Args:
            n: Number of samples to produce.

        Returns:
            np.ndarray: Array of n consecutive samples (float32).
                - Mono: shape (n, )
                - Stereo: shape (n, 2)
        """
        n = validate_sample_count(n)
        samples = _get_vectorized_samples(self.oscillator, n)

        for modifier in self.modifiers:
            samples = _apply_modifier_to_buffer(modifier, np.asarray(samples))

        return samples.astype(np.float32)


@register_component()
class WaveAdder(Composer):
    """Component that combines the output of multiple signal sources.

    A "signal source" is anything implementing the iterator protocol
    (__iter__/__next__) — this includes plain oscillators, but also Chain
    instances (an oscillator plus its own modifiers) and even other
    WaveAdders. This makes WaveAdder suitable as a general-purpose mixer:
    to combine per-channel processing (volume, pan, effects) with mixing,
    wrap each channel in its own Chain and pass the Chains to WaveAdder:

        >>> from src.engine import SineOscillator, SquareOscillator, Volume, Panner
        >>> mixer = WaveAdder(
        ...     Chain(SineOscillator(440), Volume(0.8), Panner(0.2)),
        ...     Chain(SquareOscillator(220), Volume(1.0), Panner(0.8)),
        ...     stereo=True,
        ... )

    A bare Modifier on its own (e.g. just `Volume(0.5)`, with no source
    feeding it) cannot be passed directly, since modifiers implement
    __call__ (transform an incoming value) rather than __next__ (produce
    the next value from nothing).

    Supports two mixing modes:
    - 'average': Returns the mean (prevents clipping)
    - 'sum': Returns the sum (standard mixer behavior, maintains levels)

    For parallel composition of waves.
    """

    descriptor = ComponentDescriptor(
        name="WaveAdder",
        category=ComponentCategory.COMPOSER,
        fluent_api_name="wave_adder",
        description=(
            "Mixes multiple signal sources together (oscillators, or Chains "
            "combining an oscillator with its own modifiers)."
        ),
        parameters={
            "generators": ParameterDescriptor(
                name="generators",
                default=(),
                description=(
                    "Signal source components to mix. Accepts any component "
                    "implementing the iterator protocol, including plain "
                    "oscillators, Chain instances (oscillator + modifiers), "
                    "or nested WaveAdders."
                ),
            ),
            "stereo": ParameterDescriptor(
                name="stereo",
                default=False,
                choices=(False, True),
                description=(
                    "Whether scalar generator output should be duplicated to stereo."
                ),
            ),
            "mix_mode": ParameterDescriptor(
                name="mix_mode",
                default="average",
                choices=("average", "sum"),
                description="Mixing policy for combined generator output.",
            ),
        },
        tags=["composer", "wave_adder", "mixer"],
    )

    def __init__(self, *generators, stereo: bool = False, mix_mode: str = "average"):
        """Initialize WaveAdder.

        Args:
            *generators: Signal source components to mix — instances of
                generators/oscillators, or any component (including Chain or
                another WaveAdder) that implements the iterator protocol via
                __iter__ and __next__.
            stereo: if True the output will have a tuple of two numbers for the left
                and the right channel each, else only one number.
            mix_mode: 'average' (default) or 'sum'.
                - 'average': divides by number of generators (prevents clipping)
                - 'sum': direct sum (standard mixer behavior)

        Raises:
            ValueError: If no generators provided or invalid mix_mode.
            TypeError: If stereo is not a boolean, or a generator doesn't
                implement the iterator protocol.
        """
        # Input validation
        if len(generators) == 0:
            raise ValueError("WaveAdder requires at least one generator")

        if not isinstance(stereo, bool):
            raise TypeError(f"stereo must be a boolean, got {type(stereo).__name__}")

        if mix_mode not in ("average", "sum"):
            raise ValueError(f"mix_mode must be 'average' or 'sum', got {mix_mode!r}")

        for index, gen in enumerate(generators):
            _validate_iterable_generator(gen, label=f"generators[{index}]")

        super().__init__(*generators)

        self.generators = generators
        self.stereo = stereo
        self.mix_mode = mix_mode

        logger.debug(
            "WaveAdder initialized: %d generators, stereo=%s, mix_mode=%r",
            len(generators),
            stereo,
            mix_mode,
        )

    def _mod_channels(self, _val):
        # int/float/np.number scalar handling. Generators are expected to
        # yield plain Python floats (the convention used throughout this
        # codebase's oscillators), but np.number is included defensively in
        # case a generator yields a numpy scalar (e.g. np.float32) directly
        # without casting first.
        if isinstance(_val, (int, float, np.number)) and not isinstance(_val, bool):
            if self.stereo:
                return _val, _val
            return _val

        if isinstance(_val, Sequence) and not self.stereo:
            if self.mix_mode == "sum":
                return sum(_val)

            # average
            return sum(_val) / len(_val)
        return _val

    def _mix_stereo(self, vals):
        """Mix generator outputs into a stereo (left, right) tuple."""
        left_values, right_values = zip(*vals, strict=False)
        if self.mix_mode == "sum":
            return sum(left_values), sum(right_values)
        # average
        return (
            sum(left_values) / len(left_values),
            sum(right_values) / len(right_values),
        )

    def _mix_mono(self, vals):
        """Mix generator outputs into a single scalar."""
        if self.mix_mode == "sum":
            return sum(vals)
        # average
        return sum(vals) / len(vals)

    def trigger_release(self):
        for gen in self.generators:
            if hasattr(gen, "trigger_release"):
                gen.trigger_release()

    @property
    def ended(self):
        if self._ended is not None:
            return self._ended
        ended = [gen.ended for gen in self.generators if hasattr(gen, "ended")]
        return all(ended)

    @ended.setter
    def ended(self, value: bool) -> None:
        self._ended = value

    def __iter__(self):
        for gen in self.generators:
            iter(gen)
        return self

    def __next__(self):
        vals = [self._mod_channels(next(gen)) for gen in self.generators]
        if self.stereo:
            return self._mix_stereo(vals)
        return self._mix_mono(vals)

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
        # Fast path for single generator (no mixing needed).
        # Uses _get_vectorized_samples (not gen.get_samples directly) so
        # this path supports the same minimal generators (only __next__ /
        # get_samples_vectorized, no .get_samples) that the multi-generator
        # path below already supports.
        if len(self.generators) == 1:
            samples = _get_vectorized_samples(self.generators[0], n)

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
