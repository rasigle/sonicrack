"""Fixed parameter versions of common wave generators."""

import itertools
from enum import Enum

import numpy as np


class WaveForm(Enum):
    SINE = 0
    SAW = 1
    TRIANGLE = 2
    SQUARE = 3


DEFAULT_SAMPLE_RATE: int = 44100


class Signal:
    """Base class for all signals."""

    def __init__(self, sr: float = DEFAULT_SAMPLE_RATE):
        self.rate = sr  # Samples per second

    def samples(self, t_start: float, t_offs=None, n: int = 1) -> np.ndarray:
        """Return the next n samples from this generator."""
        raise NotImplementedError("samples() not implemented in base class")


class Saw(Signal):
    """Sawtooth VCO."""

    def __init__(self, f: float, sr: float = DEFAULT_SAMPLE_RATE):
        """Make a new sawtooth generator.

        Args:
            f: Frequency in Hz of the Signal

        """
        super().__init__(sr)
        self.tmul = f / sr

    def samples(self, t_start: float, t_offs: float = None, n: int = 1) -> np.ndarray:
        """Return the next n samples from this generator.

        Args:
            t_start: Start time in seconds.
            t_offs: Time offset (phase) in seconds.
            n: Number of samples to generate.
        """
        times = make_times(t_start, n, t_offs)
        a = self.tmul * times
        return 2.0 * (a - np.floor(0.5 + a))


class Triangle(Signal):
    """Triangle VCO."""

    def __init__(self, f, sr=DEFAULT_SAMPLE_RATE):
        """Make a new triangle generator."""
        super().__init__(sr)
        self.tmul = f / sr

    def samples(self, t_start: float, t_offs: float = None, n: int = 1) -> np.ndarray:
        """Return the next n samples from this generator."""
        times = make_times(t_start, n, t_offs)
        a = times * self.tmul
        return 2.0 * np.abs(2.0 * (a - np.floor(a + 0.5))) - 1.0


class Sine(Signal):
    """Sine VCO."""

    def __init__(self, f, sr=DEFAULT_SAMPLE_RATE):
        """Make a new sine generator."""
        super().__init__(sr)
        self.period = 2 * np.pi * f / sr

    def samples(
        self, t_start: float = 0.0, t_offs: float = None, n: int = 1
    ) -> np.ndarray:
        """Return the next n samples from this generator."""
        times = make_times(t_start, n, t_offs)
        return np.sin(times * self.period)


class Square(Signal):
    """Square VCO."""

    def __init__(self, f: float, sr: float = DEFAULT_SAMPLE_RATE):
        """Make a new square generator."""
        super().__init__(sr)
        self.tmul = f / sr

    def samples(self, t_start: float, t_offs: float = None, n: int = 1) -> np.ndarray:
        """Return the next n samples from this generator."""
        # return np.sign(np.sin(2 * np.pi * freq * t))
        times = make_times(t_start, n, t_offs)
        a = self.tmul * times
        return 2.0 * (2.0 * np.floor(a) - np.floor(2.0 * a)) + 1.0


def generate_waveform(waveform_type: WaveForm, freq, phase, samplesize, sr):
    """Generate waveform samples for given phase and frequency."""

    if waveform_type == WaveForm.SINE:
        return Sine(freq, sr).samples(t_start=phase, n=samplesize)

    elif waveform_type == WaveForm.SQUARE:
        return Square(freq, sr).samples(t_start=phase, n=samplesize)

    elif waveform_type == WaveForm.SAW:
        return Saw(freq, sr).samples(t_start=phase, n=samplesize)

    elif waveform_type == WaveForm.TRIANGLE:
        return Triangle(freq, sr).samples(t_start=phase, n=samplesize)

    else:
        t = (np.arange(samplesize) + phase) / sr
        return np.zeros_like(t)


def make_times(t_start: float, n: int, t_offs: float = 0.0) -> np.ndarray:
    """Return a sequence of n time points for wave sampling.

    Args:
        t_start: Start time in seconds.
        t_offs: Time offset (phase) in seconds.
        n: Number of samples to generate.

    Returns:
        A numpy array of time points.
    """
    times = np.linspace(t_start, t_start + n, num=n, endpoint=False, dtype=np.float32)
    if t_offs is not None:
        times += t_offs
    return times


def get_sin_oscillator(
    freq: float,
    amp: float = 1,
    phase: float = 0.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
):
    phase = (phase / 360) * 2 * np.pi
    increment = (2 * np.pi * freq) / sample_rate
    return (np.sin(phase + v) * amp for v in itertools.count(start=0, step=increment))


def get_n(iterator, n: int = DEFAULT_SAMPLE_RATE):
    return [next(iterator) for i in range(n)]
