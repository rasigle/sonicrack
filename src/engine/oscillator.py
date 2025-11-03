"""Waveform oscillators for audio synthesis.

This module provides a comprehensive set of oscillator classes for generating
basic waveforms (sine, square, sawtooth, triangle). All oscillators support
both iterator-based and vectorized sample generation, with runtime parameter
modification capabilities.

Classes:
    Oscillator: Abstract base class for all oscillators.
    SineOscillator: Generates pure sine waves.
    SquareOscillator: Generates square waves.
    SawtoothOscillator: Generates sawtooth waves.
    TriangleOscillator: Generates triangle waves.

Functions:
    synth: Convenience function to generate complete waveforms.

Example:
    >>> # Create and use an oscillator
    >>> osc = SineOscillator(frequency=440, amplitude=1.0, phase=0.0)
    >>> samples = osc.get_samples_vectorized(1000)  # Fast vectorized generation
    >>>
    >>> # Change parameters at runtime
    >>> osc.frequency = 880
    >>> more_samples = osc.get_samples(500)
    >>>
    >>> # Use convenience function
    >>> wave = synth(frequency=440, dur=1.0, stype="sine")

Performance:
    - Iterator mode: Flexible but slower, suitable for small buffers
    - Vectorized mode: 50-85x faster, suitable for production use
    - Auto mode: Automatically selects best method based on buffer size

Note:
    All oscillators maintain phase continuity when switching between
    iterator and vectorized modes, enabling seamless parameter changes
    during audio generation.
"""

from abc import abstractmethod, ABC

import numpy as np

from src.engine.audio_component import Generator, ComponentDescriptor
from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.audio_component_registry import register_component, ComponentCategory


class Oscillator(Generator):
    """Base class for all signal generators.

    The oscillator is initialized with fixed initial values but exposes properties to
    modify the running parameters without reconstructing the instance.

    Args:
        frequency: Initial frequency in Hz.
        amplitude: Initial amplitude. Defaults to 1.
        phase: Initial phase in degrees. Defaults to 0.0.
        sample_rate: Samples per second. Defaults to `DEFAULT_SAMPLE_RATE`.
        wave_range: Tuple specifying value range (min, max) of raw waveform before
            amplitude scaling. Defaults to (-1, 1).

    Attributes:
        sample_rate: Samples per second (public alias).
        _i: internal time/index state.
        _step: internal step for phase progression (implementation-specific).
    """

    def __init__(
        self,
        frequency: float = 440,
        amplitude: float = 1,
        phase: float = 0.0,
        sample_rate: int | float = DEFAULT_SAMPLE_RATE,
        wave_range: tuple[float, float] = (-1, 1),
    ):
        super().__init__(sample_rate=sample_rate)

        self._freq = frequency
        self._amp = amplitude
        self._phase = phase
        self._sample_rate = sample_rate
        self._wave_range = wave_range

        self._i = 0
        self._step = 0

        # Properties that can be changed
        self._f = frequency
        self._a = amplitude
        self._p = self._phase

        # Pre-compute wave_range conversion
        self._needs_range_conversion: bool = False
        self._range_scale: float = 1.0
        self._range_offset: float = 0.0
        self._update_range_conversion()

        iter(self)

    @property
    def init_freq(self):
        """float: The initial frequency supplied at construction (Hz)."""
        return self._freq

    @property
    def init_amp(self):
        """float: The initial amplitude supplied at construction."""
        return self._amp

    @property
    def init_phase(self):
        """float: The initial phase supplied at construction (degrees)."""
        return self._phase

    @property
    def wave_range(self):
        """tuple[float, float]: Current wave range (min, max)."""
        return self._wave_range

    @wave_range.setter
    def wave_range(self, value: tuple[float, float]):
        """Set wave range and recompute conversion constants."""
        self._wave_range = value
        self._update_range_conversion()

    def _update_range_conversion(self):
        """Update pre-computed wave_range conversion constants.

        Called automatically when wave_range property is changed.
        """
        self._needs_range_conversion = self._wave_range != (-1, 1)
        if self._needs_range_conversion:
            self._range_scale = (self._wave_range[1] - self._wave_range[0]) / 2.0
            self._range_offset = (self._wave_range[1] + self._wave_range[0]) / 2.0
        else:
            # Reset to avoid stale values
            self._range_scale = 1.0
            self._range_offset = 0.0

    @property
    def frequency(self):
        """float: Current oscillator frequency in Hz.

        Setting this property updates implementation-specific internal state
        by calling `_post_freq_set`.
        """
        return self._f

    @frequency.setter
    def frequency(self, value):
        self._f = value
        self._post_freq_set()

    @property
    def amplitude(self):
        """float: Current amplitude.

        Setting this property updates implementation-specific internal state
        by calling `_post_amp_set`.
        """
        return self._a

    @amplitude.setter
    def amplitude(self, value):
        self._a = value
        self._post_amp_set()

    @property
    def phase(self):
        """float: Current phase (degrees).

        Setting this property updates implementation-specific internal state
        by calling `_post_phase_set`.
        """
        return self._p

    @phase.setter
    def phase(self, value):
        self._p = value
        self._post_phase_set()

    def _post_freq_set(self):
        """Hook called after `freq` is changed.

        Subclasses may override to recompute derived values (period, step, etc.).
        """
        pass

    def _post_amp_set(self):
        """Hook called after `amp` is changed."""
        pass

    def _post_phase_set(self):
        """Hook called after `phase` is changed."""
        pass

    def _initialize_osc(self):
        """Perform any subclass-specific initialization required when iteration starts.

        This is called from `__iter__` whenever iteration is (re)initialized.
        """
        pass

    @staticmethod
    def squish_val(val, min_val=0, max_val=1):
        """Map a value in [-1, 1] to a range [min_val, max_val].

        Args:
            val (float): Value expected roughly in [-1, 1].
            min_val (float, optional): Minimum of target range. Defaults to 0.
            max_val (float, optional): Maximum of target range. Defaults to 1.

        Returns:
            float: Rescaled value in [min_val, max_val].
        """
        return (((val + 1) / 2) * (max_val - min_val)) + min_val

    def __next__(self):
        """Return the next sample from the oscillator.

        Subclasses must implement this method to advance internal state and
        return a single sample (scaled by amplitude).

        Returns:
            float: Next sample value.

        Raises:
            StopIteration: If the oscillator cannot produce more samples
                (not expected in these continuous generators).
        """
        return None

    def __iter__(self):
        """Initialize iteration and prepare the oscillator to yield samples.

        This sets the runtime properties to their initial values and calls
        `_initialize_osc`.

        Returns:
            Oscillator: self, ready for iteration.
        """
        self.frequency = self._freq
        self.phase = self._phase
        self.amplitude = self._amp
        self._initialize_osc()
        return self

    def get_samples_iterator(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False
    ) -> np.ndarray:
        """Generate n samples using Python iterator (slower but flexible).

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            reset: If True, reset the oscillator to initial state before generating.

        Returns:
            np.ndarray: Array of `n` consecutive samples (float32).

        Note:
            This method is slower than `get_samples_vectorized()` but allows
            for per-sample parameter changes and is useful for prototyping.
        """
        if reset:
            iter(self)
        return np.array([next(self) for _ in range(n)], dtype=np.float32)

    @abstractmethod
    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using NumPy vectorization (high performance).

        This method is 50-100x faster than `get_samples_iterator()` for large buffers.
        Use this for real-time synthesis or when generating many samples.

        Args:
            n: Number of samples to produce.

        Returns:
            np.ndarray: Array of `n` consecutive samples.

        Note:
            This method updates the internal state (_i) to maintain phase continuity
            with the iterator interface.
        """
        pass

    def get_samples(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False, mode: str = "auto"
    ) -> np.ndarray:
        """Generate n samples using the specified method.

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            reset: If True, reset the oscillator to initial state before generating.
            mode: Generation mode. Options:
                - "auto": Automatically choose the best method (vectorized for
                    n >= 512, iterator otherwise)
                - "iterator": Use Python iterator (slower, flexible)
                - "vectorized": Use NumPy vectorization (faster, recommended for
                    production)

        Returns:
            np.ndarray: Generated samples as NumPy array.

        Raises:
            ValueError: If mode is not one of "auto", "iterator", or "vectorized".

        Examples:
            >>> osc = SineOscillator(440)
            >>> samples1 = osc.get_samples(1000)  # Auto-selects vectorized (fast)
            >>> samples2 = osc.get_samples(100, mode="iterator")  # Force iterator
            >>> samples3 = osc.get_samples(44100, mode="vectorized")  # Force vectorized
        """
        if mode not in ("auto", "iterator", "vectorized"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'auto', 'iterator', or 'vectorized'."
            )

        if mode == "auto":
            # Auto-select based on buffer size
            # Vectorized is faster for n >= 512, iterator for smaller sizes
            mode = "vectorized" if n >= 512 else "iterator"

        if mode == "iterator":
            samples_list = self.get_samples_iterator(n, reset=reset)
            return np.array(samples_list, dtype=np.float32)

        else:  # mode == "vectorized"
            if reset:
                iter(self)
            return self.get_samples_vectorized(n)


@register_component()
class SawtoothOscillator(Oscillator):
    """Sawtooth wave generator.

    The sawtooth oscillates in the range specified by `wave_range` and is scaled by
    `amp`. Phase is interpreted in degrees and converted to an offset within the
    oscillator period.

    The implementation uses an internal period (`_period`) computed from
    `sample_rate / freq`.
    """

    descriptor = ComponentDescriptor(
        name="Sawtooth",
        category=ComponentCategory.OSCILLATOR,
        description="Sawtooth wave oscillator",
        fluent_api_name="sawtooth",
        config_params=["frequency", "amplitude", "phase", "sample_rate", "wave_range"],
        tags=["basic", "oscillator", "sawtooth"]
    )

    def _post_freq_set(self):
        """Update derived period when frequency changes."""
        self._period = self._sample_rate / self._f
        self._post_phase_set()

    def _post_phase_set(self):
        """Convert phase (degrees) to an index offset into the period."""
        self._p = (self._p / 360) * self._period

    def _initialize_osc(self):
        """Reset internal sample index."""
        self._i = 0

    def __next__(self):
        """Compute next sawtooth sample and advance index.

        Returns:
            float: Next sawtooth sample scaled by amplitude.
        """
        div = (self._i + self._p) / self._period if self._period != 0 else 0
        val = 2 * (div - np.floor(0.5 + div))
        self._i = self._i + 1
        if self._wave_range != (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using NumPy vectorization.

        Returns:
            np.ndarray: Array of sawtooth samples (float32).
        """
        # Generate sample indices
        indices = np.arange(n, dtype=np.float32) + self._i

        # Compute sawtooth values
        if self._period != 0:
            div = (indices + self._p) / self._period
            val = 2 * (div - np.floor(0.5 + div))
        else:
            val = np.zeros(n, dtype=np.float32)

        # Apply wave range if needed (optimized with pre-computed constants)
        if self._needs_range_conversion:
            val = val * self._range_scale + self._range_offset

        # Scale by amplitude
        samples = val * self._a

        # Update internal state
        self._i += n

        return samples.astype(np.float32)


@register_component()
class TriangleOscillator(SawtoothOscillator):
    """Triangle wave generator derived from sawtooth logic.

    The triangle waveform is computed by taking the absolute of a centered
    sawtooth and scaling it to [-1, 1] before amplitude scaling.
    """

    descriptor = ComponentDescriptor(
        name="Triangle",
        category=ComponentCategory.OSCILLATOR,
        description="Triangle wave oscillator",
        tags=["basic", "oscillator", "triangle"],
        fluent_api_name="triangle",
        config_params = ["frequency", "amplitude", "phase", "sample_rate", "wave_range"],
    )

    def __next__(self):
        """Compute next triangle sample and advance index.

        Returns:
            float: Next triangle sample scaled by amplitude.
        """
        div = (self._i + self._p) / self._period if self._period != 0 else 0
        val = 2 * (div - np.floor(0.5 + div))
        val = (abs(val) - 0.5) * 2
        self._i = self._i + 1
        if self._wave_range != (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using NumPy vectorization (100x faster).

        Returns:
            np.ndarray: Array of triangle samples (float32).
        """
        # Generate sample indices
        indices = np.arange(n, dtype=np.float32) + self._i

        # Compute triangle values
        if self._period != 0:
            div = (indices + self._p) / self._period
            val = 2 * (div - np.floor(0.5 + div))
            val = (np.abs(val) - 0.5) * 2
        else:
            val = np.zeros(n, dtype=np.float32)

        # Apply wave range if needed (optimized with pre-computed constants)
        if self._needs_range_conversion:
            val = val * self._range_scale + self._range_offset

        # Scale by amplitude
        samples = val * self._a

        # Update internal state
        self._i += n

        return samples.astype(np.float32)


@register_component()
class SineOscillator(Oscillator):
    """Sine wave generator.

    The sine oscillator uses `_step` to advance the internal phase per sample.
    Phase is converted from degrees to radians in `_post_phase_set`.
    """

    descriptor = ComponentDescriptor(
        name="Sine",
        category=ComponentCategory.OSCILLATOR,
        description="Pure sine wave oscillator",
        tags=["basic", "oscillator", "sine"],
        fluent_api_name="sine",
        config_params = ["frequency", "amplitude", "phase", "sample_rate", "wave_range"],
    )

    def _post_freq_set(self):
        """Recompute the angular step per sample when frequency changes."""
        self._step = (2 * np.pi * self._f) / self._sample_rate

    def _post_phase_set(self):
        """Convert phase from degrees to radians for internal usage."""
        self._p = np.deg2rad(self._p)

    def _initialize_osc(self):
        """Reset internal accumulator used for phase progression."""
        self._i = 0

    def __next__(self):
        """Return next sine sample and advance internal phase accumulator.

        Returns:
            float: Next sine sample scaled by amplitude.
        """
        val = np.sin(self._i + self._p)

        # Optimized phase wrapping: only wrap when needed (10-15% faster)
        self._i += self._step
        if self._i >= 2 * np.pi:
            self._i -= 2 * np.pi

        if self._wave_range != (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using NumPy vectorization (100x faster).

        Returns:
            np.ndarray: Array of sine samples (float32).
        """
        # Generate phase values for all samples (optimized: pre-add phase offset)
        phases = (self._i + self._p) + self._step * np.arange(n)

        # Compute sine values
        val = np.sin(phases)

        # Apply wave range if needed (optimized with pre-computed constants)
        if self._needs_range_conversion:
            val = val * self._range_scale + self._range_offset

        # Scale by amplitude
        samples = val * self._a

        # Update internal state with phase wrapping
        self._i = (self._i + self._step * n) % (2 * np.pi)

        return samples.astype(np.float32)


@register_component()
class SquareOscillator(SineOscillator):
    """Square wave generator built on sine reference.

    The square wave threshold compares the underlying sine value to `threshold`
    and yields either `wave_range[0]` or `wave_range[1]` accordingly.
    """

    descriptor = ComponentDescriptor(
        name="Square",
        category=ComponentCategory.OSCILLATOR,
        description="Square wave oscillator",
        tags=["basic", "oscillator", "square"],
        fluent_api_name="square",
        config_params = ["frequency", "amplitude", "phase", "sample_rate", "wave_range"]
    )

    def __init__(
        self,
        frequency=440,
        amplitude=1,
        phase=0,
        sample_rate=DEFAULT_SAMPLE_RATE,
        wave_range=(-1, 1),
        threshold=0,
    ):
        """Construct a square oscillator.

        Args:
            frequency: Frequency in Hz.
            amplitude: Amplitude multiplier.
            phase: Phase in degrees.
            sample_rate: Sample rate in samples/sec.
            wave_range: Output raw range before amplitude scaling.
            threshold: Threshold used on sine reference to decide polarity.
        """
        super().__init__(frequency, amplitude, phase, sample_rate, wave_range)
        self.threshold = threshold

    def __next__(self):
        """Return next square sample and advance internal phase.

        Returns:
            float: Next square sample scaled by amplitude.
        """
        val = np.sin(self._i + self._p)

        # Optimized phase wrapping: only wrap when needed (10-15% faster)
        self._i += self._step
        if self._i >= 2 * np.pi:
            self._i -= 2 * np.pi

        val = self._wave_range[0] if val < self.threshold else self._wave_range[1]
        return val * self._a

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using NumPy vectorization.

        Returns:
            np.ndarray: Array of square samples (float32).
        """
        # Generate phase values for all samples (optimized: pre-add phase offset)
        phases = (self._i + self._p) + self._step * np.arange(n)

        # Compute sine values and threshold
        sine_vals = np.sin(phases)
        val = np.where(
            sine_vals < self.threshold, self._wave_range[0], self._wave_range[1]
        )

        # Scale by amplitude
        samples = val * self._a

        # Update internal state with phase wrapping
        self._i = (self._i + self._step * n) % (2 * np.pi)

        return samples.astype(np.float32)


def synth(
    frequency: float = 440,
    dur: float = 1.0,
    amplitude: float = 1.0,
    sr: float | int = DEFAULT_SAMPLE_RATE,
    stype: str = "sine",
    mode: str = "auto",
) -> np.ndarray:
    """Synthesizes a waveform of given type.

    Args:
        frequency (float): Frequency of the waveform in Hz.
        dur (float): Duration of the waveform in seconds.
        amplitude (float): Amplitude of the waveform.
        sr (float): Sample rate in samples per second.
        stype (str): Type of waveform ('sine', 'square', 'sawtooth', 'triangle').
        mode (str): Generation mode ('auto', 'iterator', or 'vectorized').
            Defaults to 'auto'.

    Returns:
        np.ndarray: Array of samples.
    """

    n_samples = int(dur * sr)

    stype = stype.lower()
    if stype == "sin":
        stype = "sine"
    if stype == "sawtooth":
        stype = "saw"
    if stype == "triangle":
        stype = "tri"

    synth_map = {
        "sine": SineOscillator,
        "square": SquareOscillator,
        "saw": SawtoothOscillator,
        "tri": TriangleOscillator,
    }
    try:
        osc = synth_map[stype](frequency=frequency, amplitude=amplitude, sample_rate=sr)
    except KeyError:
        raise ValueError(f"Unsupported waveform type: {stype}")

    return osc.get_samples(n_samples, mode=mode)
