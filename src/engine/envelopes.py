import numpy as np
import itertools

from engine.generators import DEFAULT_SAMPLE_RATE


class ADSREnvelope:
    """
    A simple ADSR envelope with the four stages attack, decay, release and sustain.

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
        """
        attack_duration : time taken to reach from 0 to 1 in s.
        decay_duration : time taken to reach from 1 to `sustain_level` in s.
        sustain_level : the float value of the sustain stage, should typically
            be in the range [0,1]
        release_duration : time taken to reach 0 from current value in s.
        sample_rate : the sample rate at which the notes are to be consumed.
        """
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


def generate_adsr_envelope(t, attack, decay, sustain, release, total_time):
    env = np.zeros_like(t)
    attack_end = attack
    decay_end = attack + decay
    release_start = total_time - release
    for i, ti in enumerate(t):
        if ti < attack_end:
            env[i] = ti / attack_end if attack_end > 0 else 1
        elif ti < decay_end:
            env[i] = 1 - (ti - attack_end) / max(decay, 1e-9) * (1 - sustain)
        elif ti < release_start:
            env[i] = sustain
        else:
            env[i] = sustain * (1 - (ti - release_start) / max(release, 1e-9))
    return env
