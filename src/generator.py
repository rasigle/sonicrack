from enum import Enum

import numpy as np


class WaveForm(Enum):
    SINE = 0
    SAW = 1
    TRIANGLE = 2
    SQUARE = 3



DEFAULT_SAMPLE_RATE: float = 44100.


def make_times(t_start: float, n: int, t_offs: float = 0.) -> np.ndarray:
    """Return a sequence of n time points for wave sampling.

    Args:
        t_start: Start time in seconds.
        t_offs: Time offset (phase) in seconds.
        n: Number of samples to generate.

    Returns:
        A numpy array of time points.
    """
    times = np.linspace(t_start, t_start + n, num = n, endpoint = False, dtype = np.float32)
    if t_offs is not None:
        times += t_offs
    return times


class Signal:
    """Base class for all signals."""

    def __init__(self, rate: float = DEFAULT_SAMPLE_RATE):
        self.rate = rate  # Samples per second

    def samples(self, t_start: float, t_offs = None, n: int = 1) -> np.ndarray:
        """Return the next n samples from this generator."""
        raise NotImplementedError("samples() not implemented in base class")


class Saw(Signal):
    """Sawtooth VCO."""
    def __init__(self, f: float, rate: float = DEFAULT_SAMPLE_RATE):
        """Make a new sawtooth generator.

        Args:
            f: Frequency in Hz of the Signal

        """
        super().__init__(rate)
        self.tmul = f / rate

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
    def __init__(self, f, rate = DEFAULT_SAMPLE_RATE):
        """Make a new triangle generator."""
        super().__init__(rate)
        self.tmul = f / rate

    def samples(self, t_start: float, t_offs: float = None, n: int = 1) -> np.ndarray:
        """Return the next n samples from this generator."""
        times = make_times(t_start, n, t_offs)
        a = times * self.tmul
        return 2.0 * np.abs(2.0 * (a - np.floor(a + 0.5))) - 1.0

class Sine(Signal):
    """Sine VCO."""
    def __init__(self, f, rate = DEFAULT_SAMPLE_RATE):
        """Make a new sine generator."""
        super().__init__(rate)
        self.period = 2 * np.pi * f / rate

    def samples(self, t_start: float, t_offs: float = None, n: int = 1) -> np.ndarray:
        """Return the next n samples from this generator."""
        times = make_times(t_start, n, t_offs)
        return np.sin(times * self.period)


class Square(Signal):
    """Square VCO."""
    def __init__(self, f: float, rate: float = DEFAULT_SAMPLE_RATE):
        """Make a new square generator."""
        super().__init__(rate)
        self.tmul = f / rate

    def samples(self, t_start: float, t_offs: float = None, n: int = 1) -> np.ndarray:
        """Return the next n samples from this generator."""
        # return np.sign(np.sin(2 * np.pi * freq * t))
        times = make_times(t_start, n, t_offs)
        a = self.tmul * times
        return 2.0 * (2.0 * np.floor(a) - np.floor(2.0 * a)) + 1.0



def envelope(t, released, release_start):
    """Compute ADSR amplitude at time t."""

    # 🎚️ ADSR envelope settings
    ATTACK = 0.05
    DECAY = 0.1
    SUSTAIN_LEVEL = 0.4
    RELEASE = 0.3

    if not released:
        if t < ATTACK:
            return t / ATTACK
        elif t < ATTACK + DECAY:
            return 1 - (1 - SUSTAIN_LEVEL) * ((t - ATTACK) / DECAY)
        else:
            return SUSTAIN_LEVEL
    else:
        rel_t = t - release_start
        if rel_t < RELEASE:
            return SUSTAIN_LEVEL * (1 - rel_t / RELEASE)
        else:
            return 0.0


def generate_waveform(waveform_type: WaveForm, freq, phase, samplesize, rate):
    """Generate waveform samples for given phase and frequency."""

    if waveform_type == WaveForm.SINE:
        return Sine(freq, rate).samples(t_start=phase, n=samplesize)

    elif waveform_type == WaveForm.SQUARE:
        return Square(freq, rate).samples(t_start=phase, n=samplesize)

    elif waveform_type == WaveForm.SAW:
        return Saw(freq, rate).samples(t_start=phase, n=samplesize)

    elif waveform_type ==  WaveForm.TRIANGLE:
        return Triangle(freq, rate).samples(t_start=phase, n=samplesize)

    else:
        t = (np.arange(samplesize) + phase) / rate
        return np.zeros_like(t)
