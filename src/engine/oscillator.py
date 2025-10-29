"""Variable parameter versions of common wave generators.

This module provides `Oscillator` base class and concrete oscillator
implementations (sine, sawtooth, triangle, square). Instances are iterable
and support per-instance parameter changes via properties.

Example:
    >>> osc = SineOscillator(440, amp=1.0, phase=0.0)
    >>> samples = osc.get_samples(512)

Note:
    The concrete oscillators compute values on each iteration and support
    changing `freq`, `amp` and `phase` at runtime via the provided properties.
"""

from abc import abstractmethod, ABC

import numpy as np

from constants import DEFAULT_SAMPLE_RATE


class Oscillator(ABC):
    """Base class for all signal generators.

    The oscillator is initialized with fixed initial values but exposes properties to
    modify the running parameters without reconstructing the instance.

    Args:
        freq: Initial frequency in Hz.
        amp: Initial amplitude. Defaults to 1.
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
        freq: float = 440,
        amp: float = 1,
        phase: float = 0.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        wave_range: tuple[float, float] = (-1, 1),
    ):
        self.sample_rate = sample_rate  # Samples per second

        self._freq = freq
        self._amp = amp
        self._phase = phase
        self._sample_rate = sample_rate
        self._wave_range = wave_range

        self._i = 0
        self._step = 0

        # Properties that can be changed
        self._f = freq
        self._a = amp
        self._p = self._phase

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
    def freq(self):
        """float: Current oscillator frequency in Hz.

        Setting this property updates implementation-specific internal state
        by calling `_post_freq_set`.
        """
        return self._f

    @freq.setter
    def freq(self, value):
        self._f = value
        self._post_freq_set()

    @property
    def amp(self):
        """float: Current amplitude.

        Setting this property updates implementation-specific internal state
        by calling `_post_amp_set`.
        """
        return self._a

    @amp.setter
    def amp(self, value):
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

    @abstractmethod
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

    @abstractmethod
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
        self.freq = self._freq
        self.phase = self._phase
        self.amp = self._amp
        self._initialize_osc()
        return self

    def get_samples(self, n: int = DEFAULT_SAMPLE_RATE, it: bool = False):
        """Return the next *n* samples from this generator.

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            it: If True, return an iterator instead of a list.

        Returns:
            list[float]: List of `n` consecutive samples produced by calling
            `next(self)` repeatedly.

        Note:
            If `it` is True, the method returns an iterator instead of a list.
        """
        if it:
            iter(self)
        return [next(self) for _ in range(n)]


class SawtoothOscillator(Oscillator):
    """Sawtooth wave generator.

    The sawtooth oscillates in the range specified by `wave_range` and is scaled by
    `amp`. Phase is interpreted in degrees and converted to an offset within the
    oscillator period.

    The implementation uses an internal period (`_period`) computed from
    `sample_rate / freq`.
    """

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


class TriangleOscillator(SawtoothOscillator):
    """Triangle wave generator derived from sawtooth logic.

    The triangle waveform is computed by taking the absolute of a centered
    sawtooth and scaling it to [-1, 1] before amplitude scaling.
    """

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


class SineOscillator(Oscillator):
    """Sine wave generator.

    The sine oscillator uses `_step` to advance the internal phase per sample.
    Phase is converted from degrees to radians in `_post_phase_set`.
    """

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
        self._i = self._i + self._step
        if self._wave_range != (-1, 1):
            val = self.squish_val(val, *self._wave_range)
        return val * self._a


class SquareOscillator(SineOscillator):
    """Square wave generator built on sine reference.

    The square wave threshold compares the underlying sine value to `threshold`
    and yields either `wave_range[0]` or `wave_range[1]` accordingly.
    """

    def __init__(
        self,
        freq=440,
        amp=1,
        phase=0,
        sample_rate=44_100,
        wave_range=(-1, 1),
        threshold=0,
    ):
        """Construct a square oscillator.

        Args:
            freq: Frequency in Hz.
            amp: Amplitude multiplier.
            phase: Phase in degrees.
            sample_rate: Sample rate in samples/sec.
            wave_range: Output raw range before amplitude scaling.
            threshold: Threshold used on sine reference to decide polarity.
        """
        super().__init__(freq, amp, phase, sample_rate, wave_range)
        self.threshold = threshold

    def __next__(self):
        """Return next square sample and advance internal phase.

        Returns:
            float: Next square sample scaled by amplitude.
        """
        val = np.sin(self._i + self._p)
        self._i = self._i + self._step
        if val < self.threshold:
            val = self._wave_range[0]
        else:
            val = self._wave_range[1]
        return val * self._a


def synth(
        freq: float = 440,
        dur: float = 1.0,
        amp: float = 1.0,
        sr: float | int = DEFAULT_SAMPLE_RATE,
        stype: str = "sine"
) -> np.ndarray:
    """Synthesizes a waveform of given type.

    Args:
        freq (float): Frequency of the waveform in Hz.
        dur (float): Duration of the waveform in seconds.
        amp (float): Amplitude of the waveform.
        sr (float): Sample rate in samples per second.
        stype (str): Type of waveform ('sine', 'square', 'sawtooth', 'triangle').
    """

    n_samples = int(dur * sr)

    stype= stype.lower()
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
        osc = synth_map[stype](freq=freq, amp=amp, sample_rate=sr)
    except KeyError:
        raise ValueError(f"Unsupported waveform type: {stype}")

    return np.array(osc.get_samples(n_samples))
