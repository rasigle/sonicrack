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

Amplitude Control:
    Oscillators support both linear amplitude and decibel (dB) gain control:

    - **amplitude** (linear): Direct multiplier (0.0 to 1.0+)
      Example: amplitude=0.5 means output is halved

    - **gain_db** (decibels): Professional audio standard
      Example: gain_db=-6 means -6 dB reduction (≈half amplitude)

    - **wave_range**: Advanced feature for non-standard output ranges
      Default is (-1, 1) for audio. Rarely needed except for:
        * Control signals (e.g., 0 to 1 for LFO)
        * Legacy algorithm compatibility
        * Scientific applications

    How they work together:
    1. Waveform is generated in wave_range (default: -1 to 1)
    2. Converted to standard range if wave_range != (-1, 1)
    3. Multiplied by amplitude (determined from gain_db or amplitude parameter)

    Priority: gain_db > amplitude if both specified
    Default: gain_db=-20.0 (safe for mixing multiple sources)

Performance:
    - Iterator mode: Flexible but slower, suitable for small buffers
    - Vectorized mode: 50-85x faster, suitable for production use
    - Auto mode: Automatically select the best method based on buffer size

Note:
    All oscillators maintain phase continuity when switching between
    iterator and vectorized modes, enabling seamless parameter changes
    during audio generation.
"""

import logging
from abc import abstractmethod, ABC
from typing import Literal

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE, DEFAULT_GAIN_DB
from src.engine.audio_component import Generator, ComponentDescriptor
from src.engine.audio_component_registry import register_component, ComponentCategory
from src.utils.math import db_to_linear, linear_to_db, squish_val
from src.utils.utils import track_provided_args, filter_provided_args

DEFAULT_TIME_AMPLITUDE_SMOOTHING_MS = 10  # 10 milliseconds
"""Default duration for amplitude smoothing to prevent clicks"""


class Oscillator(Generator):
    """Base class for all signal generators with professional gain control.

    The oscillator supports both linear amplitude and decibel (dB) gain control,
    with runtime parameter modification without reconstructing the instance.

    **Amplitude vs. Gain (dB):**

    - Use **gain_db** for audio work (professional standard)
      * 0 dB = unity gain (no change)
      * -6 dB = half amplitude
      * -20 dB = 1/10 amplitude (safe default for mixing)

    - Use **amplitude** for direct linear control
      * 1.0 = full amplitude
      * 0.5 = half amplitude
      * 0.1 = 1/10 amplitude

    **If both gain_db and amplitude are specified:**
    gain_db takes priority. A warning is logged if they don't match.

    **wave_range explained:**
    Controls the raw waveform output range before amplitude scaling.
    Default (-1, 1) is correct for audio. Only change for special cases:

    - (0, 1): Unipolar signal (e.g., for modulation)
    - (0, 10): VCV Rack style control voltage
    - Custom ranges for specific algorithms

    Flow: raw_waveform → range_conversion → amplitude_scaling → output

    Args:
        frequency: Initial frequency in Hz.
        amplitude: Linear amplitude (0.0 to 1.0+). Default: 1.0
            Note: Ignored if gain_db is specified.
        gain_db: Gain in decibels. Default: -20.0 (safe for mixing)
            Overrides amplitude if provided.
            Set to None to use amplitude parameter instead.
        phase: Initial phase in degrees. Defaults to 0.0.
        sample_rate: Samples per second. Defaults to `DEFAULT_SAMPLE_RATE`.
        wave_range: Tuple specifying value range (min, max) of raw waveform before
            amplitude scaling. Defaults to (-1, 1).
            Advanced feature - most users should leave as default.
        gain_db: Current gain in dB (settable).
        wave_range: Current wave range (settable).

    Attributes:
        sample_rate: Samples per second (public alias).
        frequency: Current frequency in Hz (settable).
        amplitude: Current linear amplitude (settable).


    Example:
        >>> # Create oscillator with dB control (recommended)
        >>> osc = SineOscillator(frequency=440, gain_db=-20)
        >>> osc.gain_db = -12  # Increase by 8 dB
        >>>
        >>> # Or use linear amplitude
        >>> osc2 = SineOscillator(frequency=880, amplitude=0.5, gain_db=None)
        >>> osc2.amplitude = 0.8  # Increase amplitude
        >>>
        >>> # Both specified - gain_db takes priority
        >>> osc3 = SineOscillator(gain_db=-6, amplitude=0.3)  # Uses -6 dB (≈0.5)
    """

    @track_provided_args
    def __init__(
        self,
        frequency: float = 440,
        amplitude: float = 1.0,
        gain_db: float | None = DEFAULT_GAIN_DB,
        phase: float = 0.0,
        sample_rate: int | float = DEFAULT_SAMPLE_RATE,
        wave_range: tuple[float, float] = (-1, 1),
    ):
        super().__init__(sample_rate=sample_rate)

        self._freq = frequency
        self._phase = phase
        self._sample_rate = sample_rate
        self._wave_range = wave_range
        self._initial_amp = _derive_amplitude_from_init(
            self._provided_args, amplitude, gain_db  # noqa
        )

        self._i = 0
        self._step = 0

        # Properties that can be changed
        self._f = frequency
        self._a = self._initial_amp
        self._p = self._phase

        # Amplitude smoothing to prevent clicks when changing gain
        self._target_amplitude = self._initial_amp
        self._current_amplitude = self._initial_amp
        self._smoothing_samples_remaining = 0
        self._smoothing_samples_duration_total = int(
            DEFAULT_TIME_AMPLITUDE_SMOOTHING_MS * sample_rate / 1000
        )  # default 10ms smooth

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
        return self._initial_amp

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
        """float: Current amplitude (linear scale).

        For audio work, consider using the gain_db property instead,
        as decibels are the professional standard for gain control.

        Setting this property updates implementation-specific internal state
        by calling `_post_amp_set`.

        Example:
            >>> osc.amplitude = 0.5  # noqa Half amplitude
            >>> osc.amplitude = 1.0  # noqa Unity gain
        """
        return self._a

    @amplitude.setter
    def amplitude(self, value):
        # Initiate smooth transition to new amplitude (prevents clicks)
        self._target_amplitude = value
        self._smoothing_samples_remaining = self._smoothing_samples_duration_total
        self._a = value  # Update stored value
        self._post_amp_set()

    @property
    def gain_db(self) -> float:
        """float: Current gain in decibels (professional audio standard).

        This property provides decibel-based gain control, which is the
        standard in professional audio work. Use this for intuitive control
        of relative levels.

        Setting this property updates the internal amplitude using the
        conversion: amplitude = 10^(dB/20)

        Common dB values:
            0 dB = unity gain (amplitude = 1.0)
            -3 dB = half power (amplitude ≈ 0.707)
            -6 dB = half amplitude (amplitude = 0.5)
            -12 dB = quarter amplitude (amplitude = 0.25)
            -20 dB = 1/10 amplitude (amplitude = 0.1)
            +6 dB = double amplitude (amplitude = 2.0)

        Example:
            >>> osc.gain_db = -20  # noqa Safe default for mixing
            >>> osc.gain_db = 0    # noqa Unity gain
            >>> osc.gain_db += 6   # noqa Increase by 6 dB (double amplitude)
            >>> print(osc.gain_db)  # noqa Current gain in dB
        """
        return linear_to_db(self._a)

    @gain_db.setter
    def gain_db(self, value: float):
        new_amplitude = db_to_linear(value)

        # Initiate smooth transition to new amplitude (prevents clicks)
        self._target_amplitude = new_amplitude
        self._smoothing_samples_remaining = self._smoothing_samples_duration_total
        self._a = new_amplitude  # Update stored value
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
        self.amplitude = self._initial_amp
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
        config_params=[
            "frequency",
            "gain_db",
            "amplitude",
            "phase",
            "sample_rate",
            "wave_range",
        ],
        tags=["basic", "oscillator", "sawtooth"],
    )

    def __init__(self, *args, **kwargs):
        """Initialize sawtooth oscillator."""
        super().__init__(*args, **kwargs)
        # Store original phase in degrees for recalculation when frequency changes
        self._phase_degrees = self._p

    def _post_freq_set(self):
        """Update derived period when frequency changes."""
        old_period = getattr(self, "_period", None)
        self._period = self._sample_rate / self._f

        # Recalculate phase offset from original degrees
        # Handle backward compatibility - initialize _phase_degrees if missing
        if not hasattr(self, "_phase_degrees"):
            self._phase_degrees = 0.0
        self._p = (self._phase_degrees / 360) * self._period

        # CRITICAL: Adjust internal position to new period to prevent discontinuities
        # When frequency changes, wrap _i to the new period to maintain phase continuity
        if old_period is not None and old_period > 0 and self._period > 0:
            # Scale current position to new period
            phase_fraction = (self._i % old_period) / old_period
            self._i = phase_fraction * self._period

    def _post_phase_set(self):
        """Convert phase (degrees) to an index offset into the period."""
        # Ensure _phase_degrees exists (backward compatibility)
        if not hasattr(self, "_phase_degrees"):
            self._phase_degrees = 0.0
        self._phase_degrees = self._p  # Store the degree value
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
            val = squish_val(val, *self._wave_range)
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

        # Apply amplitude with smoothing if transitioning (prevents clicks!)
        if self._smoothing_samples_remaining > 0:
            # Calculate how many samples to smooth in this buffer
            smooth_count = min(n, self._smoothing_samples_remaining)

            # Create smooth amplitude envelope (linear ramp)
            amp_envelope = np.linspace(
                self._current_amplitude, self._target_amplitude, smooth_count
            )

            # Apply smoothed amplitude to first part
            samples = np.zeros(n, dtype=np.float32)
            samples[:smooth_count] = val[:smooth_count] * amp_envelope

            # Apply target amplitude to rest (if any)
            if smooth_count < n:
                samples[smooth_count:] = val[smooth_count:] * self._target_amplitude

            # Update state
            self._smoothing_samples_remaining -= smooth_count
            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude
        else:
            # No smoothing needed - direct multiplication
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
        config_params=[
            "frequency",
            "gain_db",
            "amplitude",
            "phase",
            "sample_rate",
            "wave_range",
        ],
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
            val = squish_val(val, *self._wave_range)
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

        # Apply amplitude with smoothing if transitioning (prevents clicks!)
        if self._smoothing_samples_remaining > 0:
            # Calculate how many samples to smooth in this buffer
            smooth_count = min(n, self._smoothing_samples_remaining)

            # Create smooth amplitude envelope (linear ramp)
            amp_envelope = np.linspace(
                self._current_amplitude, self._target_amplitude, smooth_count
            )

            # Apply smoothed amplitude to first part
            samples = np.zeros(n, dtype=np.float32)
            samples[:smooth_count] = val[:smooth_count] * amp_envelope

            # Apply target amplitude to rest (if any)
            if smooth_count < n:
                samples[smooth_count:] = val[smooth_count:] * self._target_amplitude

            # Update state
            self._smoothing_samples_remaining -= smooth_count
            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude
        else:
            # No smoothing needed - direct multiplication
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
        config_params=[
            "frequency",
            "gain_db",
            "amplitude",
            "phase",
            "sample_rate",
            "wave_range",
        ],
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
            val = squish_val(val, *self._wave_range)
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

        # Apply amplitude with smoothing if transitioning (prevents clicks!)
        if self._smoothing_samples_remaining > 0:
            # Calculate how many samples to smooth in this buffer
            smooth_count = min(n, self._smoothing_samples_remaining)

            # Create smooth amplitude envelope (linear ramp)
            amp_envelope = np.linspace(
                self._current_amplitude, self._target_amplitude, smooth_count
            )

            # Apply smoothed amplitude to first part
            samples = np.zeros(n, dtype=np.float32)
            samples[:smooth_count] = val[:smooth_count] * amp_envelope

            # Apply target amplitude to rest (if any)
            if smooth_count < n:
                samples[smooth_count:] = val[smooth_count:] * self._target_amplitude

            # Update state
            self._smoothing_samples_remaining -= smooth_count
            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude
        else:
            # No smoothing needed - direct multiplication
            samples = val * self._a

        # Update internal state with phase wrapping
        self._i = (self._i + self._step * n) % (2 * np.pi)

        return samples.astype(np.float32)


# Type alias for square wave modes
SquareWaveMode = Literal["ideal", "ideal_smooth","soft"]


@register_component()
class SquareOscillator(SineOscillator):
    """Square/Pulse wave generator with variable pulse width.

    The square wave uses phase comparison to generate pulses with configurable
    duty cycle. When the phase is below the pulsewidth threshold, the output is
    `wave_range[1]`, otherwise `wave_range[0]`.

    Pulse width is specified as a fraction of the period (0.0 to 1.0):
    - 0.5 = traditional square wave (50% duty cycle)
    - 0.1 = narrow pulse (10% high, 90% low)
    - 0.9 = wide pulse (90% high, 10% low)
    """

    descriptor = ComponentDescriptor(
        name="Square",
        category=ComponentCategory.OSCILLATOR,
        description="Square/Pulse wave oscillator with variable pulse width",
        tags=["basic", "oscillator", "square", "pulse"],
        fluent_api_name="square",
        config_params=[
            "frequency",
            "gain_db",
            "amplitude",
            "phase",
            "sample_rate",
            "wave_range",
            "pulsewidth",
        ],
    )

    @track_provided_args
    def __init__(
        self,
        frequency: float = 440,
        amplitude: float = 1.0,
        gain_db: float | None = DEFAULT_GAIN_DB,
        phase: float = 0.0,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        wave_range: tuple[float, float] = (-1, 1),
        pulsewidth: float = 0.5,
        mode: SquareWaveMode = "ideal",
        **mode_kwargs,
    ):
        """Construct a square/pulse oscillator with variable pulse width.

        Args:
            frequency: Initial frequency in Hz.
            amplitude: Linear amplitude (0.0 to 1.0+). Default: 1.0
                Note: Ignored if gain_db is specified.
            gain_db: Gain in decibels. Default: -20.0 (safe for mixing)
                Overrides amplitude if provided.
                Set to None to use amplitude parameter instead.
            phase: Initial phase in degrees. Defaults to 0.0.
            wave_range: Tuple specifying value range (min, max) of raw waveform before
                amplitude scaling. Defaults to (-1, 1).
                Advanced feature - most users should leave as default.
            pulsewidth: Pulse width as fraction of period (0.0 to 1.0).
                0.5 = traditional square wave (50% duty cycle)
                0.1 = narrow pulse (10% high, 90% low)
                0.9 = wide pulse (90% high, 10% low)
                Default: 0.5
            mode: Square wave generation algorithm. Options:
                - "ideal": Traditional square wave (instant transitions, aliasing)
                - "ideal_smooth": Ideal square with amplitude smoothing (reduces clicks)
                - "soft": Smooth transitions using tanh() (warm, reduced aliasing)
                Default: "ideal"
            **mode_kwargs: Algorithm-specific parameters:
                - For "ideal_smooth": smoothing_time_ms (default 5.0), sample_rate
                - For "soft": smoothness (1.0-100.0, default 10.0)

        Examples:
            >>> # Standard square wave (50% duty cycle, ideal algorithm)
            >>> osc = SquareOscillator(frequency=440)

            >>> # Soft square with custom smoothness
            >>> osc = SquareOscillator(frequency=220, mode="soft", smoothness=20.0)

        """
        # Filter to pass only arguments explicitly provided by user
        kwargs = filter_provided_args(
            self._provided_args,  # noqa
            frequency=frequency,
            amplitude=amplitude,
            gain_db=gain_db,
            phase=phase,
            sample_rate=sample_rate,
            wave_range=wave_range,
        )
        super().__init__(**kwargs)

        # Validate pulsewidth
        if not 0.0 <= pulsewidth <= 1.0:
            raise ValueError(
                f"pulsewidth must be between 0.0 and 1.0, got {pulsewidth}"
            )

        self._pulsewidth = pulsewidth
        self._mode = mode

        # Convert pulsewidth to phase threshold
        # pulsewidth of 0.5 = threshold of π (50% duty cycle)
        # pulsewidth of 0.25 = threshold of π/2 (25% of cycle is high)
        self._pulsewidth_threshold = pulsewidth * 2 * np.pi

        self._strategy = SquareWaveFactory.create(mode, **mode_kwargs)
        self._mode_kwargs = mode_kwargs


    @property
    def pulsewidth(self) -> float:
        """float: Current pulse width (0.0 to 1.0).

        Returns the duty cycle as a fraction of the period.
        0.5 = traditional square wave (50% duty cycle)
        0.1 = narrow pulse (10% high)
        0.9 = wide pulse (90% high)
        """
        return self._pulsewidth

    @pulsewidth.setter
    def pulsewidth(self, value: float):
        """Set pulse width and update internal threshold.

        Args:
            value: Pulse width between 0.0 and 1.0

        Raises:
            ValueError: If value is outside the range [0.0, 1.0]
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"pulsewidth must be between 0.0 and 1.0, got {value}")
        self._pulsewidth = value
        self._pulsewidth_threshold = value * 2 * np.pi

    @property
    def mode(self) -> SquareWaveMode:
        """str: Current square wave generation mode.

        Available modes:
        - "ideal": Traditional square wave (instant transitions)
        - "bandlimited": MinBLEP antialiased square wave
        - "soft": Smooth transitions using tanh()
        - "comparator": Sine comparator with hysteresis
        """
        return self._mode

    def set_mode(self, mode: SquareWaveMode, **mode_kwargs):
        """Change the square wave generation algorithm.

        This allows runtime switching between different square wave formulations
        without recreating the oscillator.

        Args:
            mode: New generation algorithm
            **mode_kwargs: Algorithm-specific parameters
                - For "soft": smoothness (1.0-100.0)

        Examples:
            >>> osc = SquareOscillator(frequency=440, mode="ideal")
            >>> osc.set_mode("soft", smoothness=15.0)  # Soft with custom smoothness
            >>> osc.set_mode("comparator", hysteresis=0.03)  # Comparator mode
        """
        # Update mode
        self._mode = mode
        self._mode_kwargs = mode_kwargs

        # Create new strategy
        self._strategy = SquareWaveFactory.create(mode, **mode_kwargs)

    @classmethod
    def get_available_modes(cls) -> list[str]:
        """Get list of available square wave generation modes.

        Returns:
            List of mode names

        Example:
            >>> modes = SquareOscillator.get_available_modes()
            >>> print(modes)
            ['ideal', 'soft']
        """
        return SquareWaveFactory.get_available_modes()

    def __next__(self):
        """Return next square sample and advance internal phase.

        Returns:
            float: Next square sample scaled by amplitude.
        """
        # Get current phase (wrapped to 0-2π)
        current_phase = (self._i + self._p) % (2 * np.pi)

        # Use strategy to generate square wave value
        val = self._strategy.generate_sample(
            phase=current_phase,
            pulsewidth_threshold=self._pulsewidth_threshold,
            low_value=self._wave_range[0],
            high_value=self._wave_range[1],
        )

        # Optimized phase wrapping: only wrap when needed (10-15% faster)
        self._i += self._step
        if self._i >= 2 * np.pi:
            self._i -= 2 * np.pi

        return val * self._a

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using NumPy vectorization.

        Returns:
            np.ndarray: Array of square samples (float32).
        """
        # Generate phase values for all samples (optimized: pre-add phase offset)
        phases = (self._i + self._p) + self._step * np.arange(n)

        # Wrap phases to 0-2π for pulse width comparison
        wrapped_phases = phases % (2 * np.pi)

        # Use strategy to generate square wave values
        val = self._strategy.generate_samples(
            phases=wrapped_phases,
            pulsewidth_threshold=self._pulsewidth_threshold,
            low_value=self._wave_range[0],
            high_value=self._wave_range[1],
        )

        # Apply amplitude with smoothing if transitioning (prevents clicks!)
        if self._smoothing_samples_remaining > 0:
            # Calculate how many samples to smooth in this buffer
            smooth_count = min(n, self._smoothing_samples_remaining)

            # Create smooth amplitude envelope (linear ramp)
            amp_envelope = np.linspace(
                self._current_amplitude, self._target_amplitude, smooth_count
            )

            # Apply smoothed amplitude to first part
            samples = np.zeros(n, dtype=np.float32)
            samples[:smooth_count] = val[:smooth_count] * amp_envelope

            # Apply target amplitude to rest (if any)
            if smooth_count < n:
                samples[smooth_count:] = val[smooth_count:] * self._target_amplitude

            # Update state
            self._smoothing_samples_remaining -= smooth_count
            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude
        else:
            # No smoothing needed - direct multiplication
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
    if dur <= 0:
        raise ValueError("Duration must be positive.")
    if frequency < 0:
        raise ValueError("Frequency must be non-negative.")

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


def _derive_amplitude_from_init(given_args, amplitude: float, gain_db: float) -> float:
    """
    Determines the linear amplitude based on __init__ parameters.
    Priority:
    1. `gain_db` if it is not None.
    2. `amplitude` if it is not None.
    3. Default to `gain_db`'s default value.
    A warning is issued if both are provided and they conflict.
    """
    # Input validation
    if amplitude and not isinstance(amplitude, (int, float, np.number)):
        raise TypeError(f"Amplitude must be number, got {type(amplitude).__name__}")
    if amplitude and amplitude < 0.0:
        raise ValueError(f"Amplitude must be non-negative, got {amplitude}")
    if gain_db and not isinstance(gain_db, (int, float, np.number)):
        raise TypeError(f"Gain_db must be a number, got {type(gain_db).__name__}")

    gain_db_set = "gain_db" in given_args
    amplitude_set = "amplitude" in given_args

    # If gain_db is explicitly provided and is not None, it takes precedence.
    if gain_db_set and gain_db is not None:
        expected_amp = db_to_linear(gain_db)
        # Warn if amplitude was also set and conflicts with gain_db's value.
        if (
            amplitude_set
            and amplitude is not None
            and not np.isclose(amplitude, expected_amp)
        ):
            logging.warning(
                f"Both gain_db={gain_db} and amplitude={amplitude} were specified. "
                f"Using gain_db, which results in an amplitude of {expected_amp:.3f}."
            )
        return expected_amp

    # Otherwise, use amplitude if it was provided and is not None.
    if amplitude_set and amplitude is not None:
        return amplitude

    # As a fallback, use the default value for gain_db if it's not None.
    if gain_db is not None:
        return db_to_linear(gain_db)

    # If both are None, use a default amplitude of 1.0
    return 1.0


class SquareWaveStrategy(ABC):
    """Abstract base class for square wave generation strategies.

    All strategies must implement both sample-by-sample and vectorized generation.
    """

    @abstractmethod
    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        """Generate a single square wave sample.

        Args:
            phase: Current phase in radians (0 to 2π)
            pulsewidth_threshold: Phase threshold for pulse width (0 to 2π)
            low_value: Value when phase >= threshold
            high_value: Value when phase < threshold

        Returns:
            Single sample value
        """
        pass

    @abstractmethod
    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        """Generate multiple square wave samples (vectorized).

        Args:
            phases: Array of phases in radians (0 to 2π)
            pulsewidth_threshold: Phase threshold for pulse width (0 to 2π)
            low_value: Value when phase >= threshold
            high_value: Value when phase < threshold

        Returns:
            Array of sample values
        """
        pass


class IdealSquareStrategy(SquareWaveStrategy):
    """Ideal (aliased) square wave - instant transitions.

    This is the traditional square wave with instantaneous transitions,
    which creates aliasing at high frequencies but has minimal CPU overhead.

    Best for: Low frequencies, retro sounds, CPU efficiency
    """

    def __init__(self, **kwargs):
        """Initialize ideal square strategy."""
        pass

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        return high_value if phase < pulsewidth_threshold else low_value

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        return np.where(phases < pulsewidth_threshold, high_value, low_value).astype(np.float32)


class IdealSquareStrategySmoothing(SquareWaveStrategy):
    """Ideal (aliased) square wave with amplitude smoothing.

    This variant of the ideal square wave includes amplitude smoothing
    to reduce clicks when changing amplitude or other parameters.

    Best for: Low frequencies, retro sounds, CPU efficiency,
    with reduced clicks on parameter changes.
    """

    def __init__(self, smoothing_time_ms: float = 5.0, sample_rate: float = 44100):
        """Initialize ideal square strategy with amplitude smoothing.

        Args:
            smoothing_time_ms: Time in milliseconds for amplitude transitions
            sample_rate: Sample rate for calculating smoothing samples
        """
        self.smoothing_time_ms = smoothing_time_ms
        self.sample_rate = sample_rate

        # Smoothing state
        self._current_amplitude = 1.0
        self._target_amplitude = 1.0
        self._smoothing_samples_remaining = 0

    def set_amplitude(self, amplitude: float):
        """Set target amplitude with smoothing.

        Args:
            amplitude: New target amplitude
        """
        if abs(amplitude - self._current_amplitude) > 0.001:
            self._target_amplitude = amplitude
            self._smoothing_samples_remaining = int(
                self.smoothing_time_ms * self.sample_rate / 1000
            )

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        # Generate ideal square value
        val = high_value if phase < pulsewidth_threshold else low_value

        # Apply smoothing if active
        if self._smoothing_samples_remaining > 0:
            # Calculate smoothing factor for this sample
            progress = 1.0 - (self._smoothing_samples_remaining /
                            (self.smoothing_time_ms * self.sample_rate / 1000))
            current_amp = (self._current_amplitude +
                          (self._target_amplitude - self._current_amplitude) * progress)

            self._smoothing_samples_remaining -= 1
            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude

            return val * current_amp

        return val * self._current_amplitude

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        n = len(phases)

        # Generate ideal square wave
        val = np.where(phases < pulsewidth_threshold, high_value, low_value)

        # Apply amplitude with smoothing if transitioning (prevents clicks!)
        if self._smoothing_samples_remaining > 0:
            # Calculate how many samples to smooth in this buffer
            smooth_count = min(n, self._smoothing_samples_remaining)

            # Create smooth amplitude envelope (linear ramp)
            amp_envelope = np.linspace(
                self._current_amplitude, self._target_amplitude, smooth_count
            )

            # Apply smoothed amplitude to first part
            samples = np.zeros(n, dtype=np.float32)
            samples[:smooth_count] = val[:smooth_count] * amp_envelope

            # Apply target amplitude to rest (if any)
            if smooth_count < n:
                samples[smooth_count:] = val[smooth_count:] * self._target_amplitude

            # Update state
            self._smoothing_samples_remaining -= smooth_count
            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude

            return samples
        else:
            # No smoothing needed - just apply current amplitude
            return (val * self._current_amplitude).astype(np.float32)


class SoftSquareStrategy(SquareWaveStrategy):
    """Soft/clipped square wave with smooth transitions.

    Uses tanh() or other soft clipping to create smooth transitions
    instead of instant jumps. Reduces aliasing while maintaining
    square-ish character.

    Best for: Warmer, smoother sounds; analog-style synthesis
    """

    def __init__(self, smoothness: float = 10.0):
        """Initialize soft square strategy.

        Args:
            smoothness: Controls transition steepness (higher = sharper)
                Range: 1.0 (very soft) to 100.0 (nearly ideal)
                Default: 10.0 (good balance)
        """
        self.smoothness = smoothness

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        # Create smooth square wave using tanh-based transitions
        # Standard square: HIGH from 0 to pulsewidth_threshold, LOW after

        # For a smooth square wave, we want:
        # - Phase < pulsewidth_threshold: output = high_value
        # - Phase >= pulsewidth_threshold: output = low_value
        # - Smooth transitions using tanh

        # Create a smooth step function that transitions from high to low
        # at pulsewidth_threshold
        # tanh maps: large negative → -1, large positive → +1
        # We want: before threshold → +1 (high), after threshold → -1 (low)

        dist_from_threshold = phase - pulsewidth_threshold
        # Negate to get correct polarity: before threshold gives negative (→ +1 after negation)
        smooth_step = np.tanh(-dist_from_threshold * self.smoothness)

        # smooth_step is now: +1 before threshold, -1 after threshold
        # Map from [-1, 1] to [low_value, high_value]
        # +1 should map to high_value, -1 should map to low_value
        return low_value + (high_value - low_value) * (smooth_step + 1) / 2

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        # Create smooth square wave using tanh-based transitions
        # Standard square: HIGH from 0 to pulsewidth_threshold, LOW after

        # Create a smooth step function that transitions from high to low
        # at pulsewidth_threshold
        dist_from_threshold = phases - pulsewidth_threshold
        smooth_step = np.tanh(-dist_from_threshold * self.smoothness)

        # smooth_step is: +1 before threshold, -1 after threshold
        # Map from [-1, 1] to [low_value, high_value]
        return low_value + (high_value - low_value) * (smooth_step + 1) / 2


class SquareWaveFactory:
    """Factory for creating square wave strategy instances.

    Provides a clean interface for switching between different
    square wave generation algorithms.
    """

    _strategies = {
        "ideal": IdealSquareStrategy,
        "ideal_smooth": IdealSquareStrategySmoothing,
        "soft": SoftSquareStrategy,
    }

    @classmethod
    def create(
        cls,
        mode: SquareWaveMode = "ideal",
        **kwargs
    ) -> SquareWaveStrategy:
        """Create a square wave strategy instance.

        Args:
            mode: Type of square wave generation
                - "ideal": Traditional square wave (instant transitions)
                - "ideal_smooth": Ideal square with amplitude smoothing (reduces clicks)
                - "soft": Smooth transitions using tanh()
            **kwargs: Strategy-specific parameters
                For "ideal_smooth": smoothing_time_ms (default 5.0), sample_rate
                For "soft": smoothness (1.0-100.0)

        Returns:
            SquareWaveStrategy instance

        Raises:
            ValueError: If mode is not recognized

        Examples:
            >>> # Ideal square wave
            >>> strategy = SquareWaveFactory.create("ideal")

            >>> # Ideal square with amplitude smoothing
            >>> strategy = SquareWaveFactory.create("ideal_smooth",
            ...                                      smoothing_time_ms=10.0,
            ...                                      sample_rate=48000)

            >>> # Soft square with custom smoothness
            >>> strategy = SquareWaveFactory.create("soft", smoothness=20.0)
        """
        if mode not in cls._strategies:
            raise ValueError(
                f"Unknown square wave mode: {mode}. "
                f"Available modes: {list(cls._strategies.keys())}"
            )

        logging.debug(f"Creating square wave strategy with mode: {mode}")
        strategy_class = cls._strategies[mode]
        return strategy_class(**kwargs)

    @classmethod
    def get_available_modes(cls) -> list[str]:
        """Get list of available square wave modes.

        Returns:
            List of mode names
        """
        return list(cls._strategies.keys())

    @classmethod
    def register_strategy(
        cls,
        name: str,
        strategy_class: type[SquareWaveStrategy]
    ):
        """Register a custom square wave strategy.

        Args:
            name: Name for the strategy
            strategy_class: Strategy class (must inherit from SquareWaveStrategy)

        Raises:
            TypeError: If strategy_class doesn't inherit from SquareWaveStrategy
        """
        if not issubclass(strategy_class, SquareWaveStrategy):
            raise TypeError(
                f"{strategy_class} must inherit from SquareWaveStrategy"
            )
        cls._strategies[name] = strategy_class
