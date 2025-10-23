"""Fixed parameter versions of common wave generators."""

import itertools

import numpy as np

from constants import DEFAULT_SAMPLE_RATE


class Signal:
    """Base class for all signal generators.

    Parameters of the signal are fixed at initialization and cannot be changed
    afterward.
    """

    def __init__(
        self,
        freq: float,
        amp: float = 1,
        phase_deg: float = 0.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        self.sample_rate = sample_rate  # Samples per second

        self.freq = freq
        self.amp = amp
        self.phase_rad = (phase_deg / 360) * 2 * np.pi
        self.osc = self._oscillator()

    def _oscillator(self):
        """Generator that yields samples indefinitely."""
        raise NotImplementedError("Subclasses must implement _oscillator method.")

    def get_samples(self, n: int = DEFAULT_SAMPLE_RATE):
        """Return the next n samples from this generator."""
        print(self.freq)
        return [next(self.osc) for _ in range(n)]


class Saw(Signal):
    """Sawtooth VCO."""

    def __init__(
        self,
        freq: float,
        amp: float = 1.0,
        phase_deg: float = 0.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        """Make a new sawtooth generator."""
        super().__init__(freq, amp, phase_deg, sample_rate)

    def _oscillator(self):
        period = self.sample_rate / self.freq
        phase_offset = self.phase_rad * (self.sample_rate / (2 * np.pi))
        sample_index = 0
        while True:
            pos_in_period = (sample_index + phase_offset) % period
            yield (2 * self.amp / period) * pos_in_period - self.amp
            sample_index += 1


class Triangle(Signal):
    """Triangle VCO."""

    def __init__(
        self,
        freq: float,
        amp: float = 1.0,
        phase_deg: float = 0.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        """Make a new triangle generator."""
        super().__init__(freq, amp, phase_deg, sample_rate)

    def _oscillator(self):
        period = self.sample_rate / self.freq
        half_period = period / 2
        phase_offset = self.phase_rad * (self.sample_rate / (2 * np.pi))
        sample_index = 0
        while True:
            pos_in_period = (sample_index + phase_offset) % period
            if pos_in_period < half_period:
                yield (2 * self.amp / half_period) * pos_in_period - self.amp
            else:
                yield (-2 * self.amp / half_period) * (
                    pos_in_period - half_period
                ) + self.amp
            sample_index += 1


class Sine(Signal):
    """Sine VCO."""

    def __init__(
        self,
        freq: float,
        amp: float = 1.0,
        phase_deg: float = 0.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        """Make a new sine generator."""
        super().__init__(freq, amp, phase_deg, sample_rate)

    def _oscillator(self):
        increment = (2 * np.pi * self.freq) / self.sample_rate
        return (
            np.sin(self.phase_rad + v) * self.amp
            for v in itertools.count(start=0, step=increment)
        )


class Square(Signal):
    """Square VCO."""

    def __init__(
        self,
        freq: float,
        amp: float = 1.0,
        phase_deg: float = 0.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        """Make a new square generator."""
        super().__init__(freq, amp, phase_deg, sample_rate)

    def _oscillator(self):
        period = self.sample_rate / self.freq
        half_period = period / 2
        phase_offset = self.phase_rad * (self.sample_rate / (2 * np.pi))
        sample_index = 0
        while True:
            pos_in_period = (sample_index + phase_offset) % period
            yield self.amp if pos_in_period < half_period else -self.amp
            sample_index += 1
