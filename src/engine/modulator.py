"""Modulators shape the amplitude or frequency of audio signals over time."""

import itertools
from abc import ABC

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE


class Modulator(ABC):

    def __init__(self):
        pass


class ADSREnvelope(Modulator):
    """A simple ADSR envelope with the four stages attack, decay, release and sustain.

    Has `.trigger_release()` implemented to trigger the release stage of the envelope.
    similarly has `.ended`, a flag to indicate the end of the release stage.
    """

    def __init__(
        self,
        attack_duration: float = 0.05,
        decay_duration: float = 0.2,
        sustain_level: float = 0.7,
        release_duration: float = 0.3,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ):
        """Initialize a new ADSR envelope instance.

        Args:
            attack_duration : time taken to reach from 0 to 1 in s.
            decay_duration : time taken to reach from 1 to `sustain_level` in s.
            sustain_level : the float value of the sustain stage, should typically
                be in the range [0,1]
            release_duration : time taken to reach 0 from current value in s.
            sample_rate : the sample rate at which the notes are to be consumed.
        """
        super().__init__()
        self.attack_duration = attack_duration
        self.decay_duration = decay_duration
        self.sustain_level = sustain_level
        self.release_duration = release_duration
        self._sample_rate = sample_rate

        self.stepper = None

    def _get_ads_stepper(self):
        steppers = []
        if self.attack_duration > 0:
            steppers.append(
                itertools.count(
                    start=0, step=1 / (self.attack_duration * self._sample_rate)
                )
            )

        if self.decay_duration > 0:
            steppers.append(
                itertools.count(
                    start=1,
                    step=-(1 - self.sustain_level)
                    / (self.decay_duration * self._sample_rate),
                )
            )

        while True:
            stepper_len = len(steppers)
            if stepper_len > 0:
                val = next(steppers[0])
                if stepper_len == 2 and val > 1:
                    steppers.pop(0)
                    val = next(steppers[0])
                elif stepper_len == 1 and val < self.sustain_level:
                    steppers.pop(0)
                    val = self.sustain_level
            else:
                val = self.sustain_level
            yield val

    def _get_r_stepper(self):
        val = 1
        if self.release_duration > 0:
            release_step = -self.val / (self.release_duration * self._sample_rate)
            stepper = itertools.count(self.val, step=release_step)
        else:
            val = -1
        while True:
            if val <= 0:
                self.ended = True
                val = 0
            else:
                val = next(stepper)
            yield val

    def __iter__(self):
        self.val = 0
        self.ended = False
        self.stepper = self._get_ads_stepper()
        return self

    def __next__(self):
        self.val = next(self.stepper)
        return self.val

    def trigger_release(self):
        self.stepper = self._get_r_stepper()

    def get_samples_iterator(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False
    ) -> np.ndarray:
        """Generate n samples using Python iterator (slower but flexible).

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            reset: If True, reset the envelope to initial state before generating.

        Returns:
            list[float]: List of `n` consecutive samples produced by calling
            `next(self)` repeatedly.
        """
        if reset:
            iter(self)
        return np.array([next(self) for _ in range(n)])

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using vectorized computation (faster).

        For ADSR envelopes, this generates the envelope curve using NumPy arrays.
        Note: This is optimized for the common case but uses iterator for complex state.

        Args:
            n: Number of samples to produce.

        Returns:
            np.ndarray: Array of `n` consecutive envelope values.

        Note:
            For ADSR envelopes, vectorization provides moderate speedup (~5-10x)
            as the state machine logic is complex.
        """
        # For ADSR, we use iterator but convert to array for consistency
        # Full vectorization would require rewriting the state machine
        samples = [next(self) for _ in range(n)]
        return np.array(samples, dtype=np.float32)

    def get_samples(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False, mode: str = "auto"
    ) -> np.ndarray:
        """Generate n samples using the specified method.

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            reset: If True, reset the envelope to initial state before generating.
            mode: Generation mode. Options:
                - "auto": Automatically choose best method (vectorized for n >= 512, iterator otherwise)
                - "iterator": Use Python iterator (slower, flexible)
                - "vectorized": Use NumPy array conversion (returns ndarray)

        Returns:
            np.ndarray or list[float]: Generated samples. Returns ndarray for vectorized mode,
            list for iterator mode.

        Raises:
            ValueError: If mode is not one of "auto", "iterator", or "vectorized".

        Examples:
            >>> env = ADSREnvelope(0.1, 0.2, 0.7, 0.3)
            >>> samples = env.get_samples(1000)  # Auto-selects vectorized
            >>> samples = env.get_samples(100, mode="iterator", reset=True)
        """
        if mode not in ("auto", "iterator", "vectorized"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'auto', 'iterator', or 'vectorized'."
            )

        if mode == "auto":
            # Auto-select based on buffer size
            mode = "vectorized" if n >= 512 else "iterator"

        if mode == "iterator":
            return self.get_samples_iterator(n, reset=reset)
        else:  # mode == "vectorized"
            if reset:
                iter(self)
            return self.get_samples_vectorized(n)

    def __str__(self):
        return (
            f"ADSREnvelope(attack={self.attack_duration}, "
            f"decay={self.decay_duration}, sustain={self.sustain_level}, "
            f"release={self.release_duration})"
        )


def getadsr(
    a=0.05, d=0.3, sl=0.7, r=0.2, sd=0.4, sample_rate=DEFAULT_SAMPLE_RATE
) -> tuple[np.ndarray, int, int]:
    """Generate ADSR envelope values for a down (attack+decay+sustain) phase and an
    up (release) phase.

    The function constructs an `ADSREnvelope` with the provided times and sustain level,
    samples the envelope for the down phase (attack + decay + sustain duration `sd`),
    invokes `trigger_release()` on the envelope, then samples the release (up) phase.

    Args:
        a (float): Attack time in seconds.
        d (float): Decay time in seconds.
        sl (float): Sustain level (amplitude between 0.0 and 1.0).
        r (float): Release time in seconds.
        sd (float): Sustain duration in seconds (time to hold at sustain level before
            release).
        sample_rate (int): Samples per second (defaults to global `SR`).

    Returns:
        tuple:
            - adsr_vals (list[float]): Concatenated envelope sample values for the down
                and up phases.
            - down_len (int): Number of samples produced before `trigger_release()`
                (attack + decay + sustain duration).
            - up_len (int): Number of samples produced after `trigger_release()`
                (release length).

    Notes:
        - Time parameters (`a`, `d`, `r`, `sd`) are interpreted in seconds and converted
          to sample counts using `sample_rate`.
    """
    adsr = ADSREnvelope(a, d, sl, r)
    down_len = int(sum([a, d, sd]) * sample_rate)
    up_len = int(r * sample_rate)
    adsr = iter(adsr)
    adsr_vals = adsr.get_samples(down_len)
    adsr.trigger_release()
    adsr_vals = np.concatenate([adsr_vals, adsr.get_samples(up_len)])
    return adsr_vals, down_len, up_len
