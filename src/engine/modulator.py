import itertools
from abc import ABC

from constants import DEFAULT_SAMPLE_RATE


class Modulator(ABC):

    def __init__(self):
        pass


class ADSREnvelope(Modulator):
    """
    A simple ADSR envelope with the four stages attack, decay, release and sustain.

    Has `.trigger_release()` implemented to trigger the release stage of the envelope.
    similarly has `.ended`, a flag to indicate the end of the release stage.
    """

    def __init__(self, attack_duration: float = 0.05,
                 decay_duration: float = 0.2, sustain_level: float = 0.7,
                 release_duration: float = 0.3,
                 sample_rate: float = DEFAULT_SAMPLE_RATE):
        """
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

    def __str__(self):
        return (f"ADSREnvelope(attack={self.attack_duration}, "
                f"decay={self.decay_duration}, sustain={self.sustain_level}, "
                f"release={self.release_duration})")
