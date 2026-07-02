"""ADSR envelope generator for audio modulation."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator
from enum import StrEnum
from typing import Literal

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    ComponentCategory,
    ComponentDescriptor,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.core.registry import register_component
from src.engine.core.sample_mode import SampleMode, VALID_SAMPLE_MODES
from src.engine.dsp.modulators.base import Modulator
from src.engine.utils.validation import validate_sample_count, validate_sample_rate

logger = logging.getLogger(__name__)


class ADSRPhase(StrEnum):
    """Internal ADSR envelope phase states."""

    IDLE = "idle"
    RETRIGGER_RESET = "retrigger_reset"
    ATTACK = "attack"
    DECAY = "decay"
    SUSTAIN = "sustain"
    RELEASE = "release"


@register_component()
class ADSREnvelope(Modulator):
    """A simple ADSR envelope with the four stages attack, decay, release and sustain.

    Has `.trigger_release()` implemented to trigger the release stage of the envelope.
    similarly has `.ended`, a flag to indicate the end of the release stage.

    Retrigger behavior controls what happens when a new note-on/gate rising edge
    arrives while the envelope is already active:

    - ``"legato"`` starts the next attack from the current envelope level. This is
      the smoothest mode and avoids amplitude dips when gates overlap or retrigger
      before release has finished.
    - ``"punch"`` first performs a short click-safe reset ramp to zero, then starts
      attack from zero. This restores a percussive VCA chop for rhythmic gates
      without the one-sample discontinuity that causes clicks.
    """

    descriptor = ComponentDescriptor(
        name="ADSREnvelope",
        category=ComponentCategory.MODULATOR,
        description="ADSR envelope generator",
        tags=["envelope", "modulator", "adsr"],
        parameters=make_parameter_descriptors(
            "attack_duration",
            "decay_duration",
            "sustain_level",
            "release_duration",
            "sample_rate",
            attack_duration=ParameterDescriptor(
                name="attack_duration",
                default=0.05,
                minimum=0.0,
                unit="s",
                description="Attack duration.",
            ),
            decay_duration=ParameterDescriptor(
                name="decay_duration",
                default=0.2,
                minimum=0.0,
                unit="s",
                description="Decay duration.",
            ),
            sustain_level=ParameterDescriptor(
                name="sustain_level",
                default=0.7,
                minimum=0.0,
                maximum=1.0,
                unit="level",
                description="Sustain level.",
            ),
            release_duration=ParameterDescriptor(
                name="release_duration",
                default=0.3,
                minimum=0.0,
                unit="s",
                description="Release duration.",
            ),
        ),
        fluent_api_name="adsr",
    )

    def __init__(
        self,
        attack_duration: float = 0.05,
        decay_duration: float = 0.2,
        sustain_level: float = 0.7,
        release_duration: float = 0.3,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        retrigger_mode: Literal["legato", "punch"] = "legato",
    ):
        """Initialize a new ADSR envelope instance.

        Args:
            attack_duration : time taken to reach from 0 to 1 in s.
            decay_duration : time taken to reach from 1 to `sustain_level` in s.
            sustain_level : the float value of the sustain stage, should typically
                be in the range [0,1]
            release_duration : time taken to reach 0 from current value in s.
            sample_rate : the sample rate at which the notes are to be consumed.
            retrigger_mode : "legato" keeps retriggers smooth from the current
                envelope level; "punch" uses a short click-safe reset before attack.
        """
        # Store as private attributes - access through properties
        self._attack_duration = attack_duration
        self._decay_duration = decay_duration
        self.sustain_level = sustain_level
        self._release_duration = release_duration
        self._sample_rate = validate_sample_rate(sample_rate)
        self.retrigger_mode = retrigger_mode
        super().__init__(sample_rate=sample_rate)

        self.ended = True  # Start in ended state
        self.val: float = 0.0  # Initialize current value
        self._attack_start_value = 0.0
        self._release_start_value = 0.0
        self._retrigger_reset_start_value = 0.0

        # Pre-compute phase durations in samples (performance optimization)
        # These will be set by _update_phase_samples()
        self._attack_samples = 0
        self._decay_samples = 0
        self._release_samples = 0
        self._update_phase_samples()

        # Vectorization state tracking
        # Start in idle/ended state, not attack! (prevents spurious triggers)
        self._phase = ADSRPhase.IDLE
        self._phase_position = 0  # Position within current phase (in samples)
        self._stepper: Iterator[float] | None = None
        self._retrigger_reset_samples = max(1, int(0.002 * self._sample_rate))

    @property
    def attack_duration(self) -> float:
        """float: Attack duration in seconds."""
        return self._attack_duration

    @attack_duration.setter
    def attack_duration(self, value: float):
        """Set attack duration and update pre-computed samples."""
        self._attack_duration = value
        self._attack_samples = int(value * self._sample_rate)

    @property
    def decay_duration(self) -> float:
        """float: Decay duration in seconds."""
        return self._decay_duration

    @decay_duration.setter
    def decay_duration(self, value: float):
        """Set decay duration and update pre-computed samples."""
        self._decay_duration = value
        self._decay_samples = int(value * self._sample_rate)

    @property
    def release_duration(self) -> float:
        """float: Release duration in seconds."""
        return self._release_duration

    @release_duration.setter
    def release_duration(self, value: float):
        """Set release duration and update pre-computed samples."""
        self._release_duration = value
        self._release_samples = int(value * self._sample_rate)

    @property
    def sample_rate(self) -> float:
        """float: Sample rate in samples per second."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float):
        """Set sample rate and update all pre-computed samples."""
        self._sample_rate = validate_sample_rate(value)
        self._update_phase_samples()
        self._retrigger_reset_samples = max(1, int(0.002 * self._sample_rate))

    def _get_ads_stepper(self):
        steppers = []
        if self.attack_duration > 0:
            attack_step = (1.0 - self._attack_start_value) / (
                self.attack_duration * self._sample_rate
            )
            steppers.append(
                itertools.count(
                    start=self._attack_start_value,
                    step=attack_step,
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
        stepper = None
        if self.release_duration > 0:
            self._release_start_value = self.val
            release_step = -self._release_start_value / (
                self.release_duration * self._sample_rate
            )
            stepper = itertools.count(self._release_start_value, step=release_step)
        else:
            val = -1
        while True:
            if val <= 0:
                self.ended = True
                val = 0
            else:
                if stepper is None:
                    return
                val = next(stepper)
            yield val

    def __iter__(self):
        # Only initialize stepper if not in idle state
        # This prevents spurious triggers when creating iterator
        if self._phase != ADSRPhase.IDLE:
            self.val = 0.0
            self._attack_start_value = 0.0
            self._release_start_value = 0.0
            self._retrigger_reset_start_value = 0.0
            self.ended = False
            self._stepper = self._get_ads_stepper()
            self._phase = ADSRPhase.ATTACK
        self._phase_position = 0
        return self

    def __next__(self):
        # Handle idle state
        if self._phase == ADSRPhase.IDLE:
            self.val = 0.0
            return 0.0

        # Ensure stepper exists
        if self._stepper is None:
            self._stepper = self._get_ads_stepper()
        assert self._stepper is not None

        self.val = next(self._stepper)
        self._phase_position += 1

        # Update phase tracking using pre-computed values (optimized)
        if (
            self._phase == ADSRPhase.ATTACK
            and self._phase_position >= self._attack_samples
        ):
            self._phase = ADSRPhase.DECAY
            self._phase_position = 0
        elif (
            self._phase == ADSRPhase.DECAY
            and self._phase_position >= self._decay_samples
        ):
            self._phase = ADSRPhase.SUSTAIN
            self._phase_position = 0

        return self.val

    def trigger_release(self):
        """Trigger the release phase of the envelope."""
        self._release_start_value = self.val
        self._stepper = self._get_r_stepper()
        self._phase = ADSRPhase.RELEASE
        self._phase_position = 0

    def trigger_note_on(self):
        """Trigger note on and start attack from the current envelope level.

        Re-triggering while the envelope is already active must not force the output
        to zero. The VCA would otherwise jump in one sample and create an audible
        click on rhythmic gate signals.
        """
        if self.ended or self._phase == ADSRPhase.IDLE:
            self._attack_start_value = 0.0

        elif self.retrigger_mode == "punch":
            self._retrigger_reset_start_value = self.val
            self.ended = False
            self._phase = ADSRPhase.RETRIGGER_RESET
            self._phase_position = 0
            self._stepper = None
            return
        else:
            self._attack_start_value = self.val

        self.ended = False
        self._phase = ADSRPhase.ATTACK
        self._phase_position = 0
        self.val = self._attack_start_value
        self._stepper = self._get_ads_stepper()

    def trigger_note_off(self):
        """Trigger note off - starts release phase.

        This is an alias for trigger_release(), compatible with MIDI note off.
        """
        self.trigger_release()

    def get_samples(
        self,
        n: int = DEFAULT_SAMPLE_RATE,
        reset: bool = False,
        mode: SampleMode = "auto",
    ) -> np.ndarray:
        """Generate n samples using vectorized computation (recommended).

        This method always uses the high-performance vectorized implementation,
        providing 50-100x speedup over iterator mode. The 'mode' parameter is
        kept for API compatibility but all modes use vectorized internally.

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            reset: If True, reset the envelope to initial state before generating.
            mode: Generation mode (kept for compatibility, all use vectorized):
                - "auto": Uses vectorized (default, recommended)
                - "vectorized": Uses vectorized
                - "iterator": Uses vectorized (not iterator despite name)

        Returns:
            np.ndarray: Generated samples as NumPy array.

        Raises:
            ValueError: If mode is not one of "auto", "iterator", or "vectorized".

        Examples:
            >>> env = ADSREnvelope(0.1, 0.2, 0.7, 0.3)
            >>> samples1 = env.get_samples(1000)  # Fast vectorized
            >>> samples2 = env.get_samples(100)  # Also vectorized

        Note:
            For the reference iterator implementation (75x slower), use
            get_samples_iterator() directly. This is only useful for testing
            or educational purposes.
        """
        if mode not in VALID_SAMPLE_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'auto', 'iterator', or 'vectorized'."
            )

        # Only if iterator mode is explicitly requested, use it
        if mode == "iterator":
            return self._get_samples_iterator(n, reset)

        # Use vectorized mode
        if reset:
            iter(self)
        return self._get_samples_vectorized(n)

    def __str__(self):
        return (
            f"ADSREnvelope(attack={self.attack_duration}, "
            f"decay={self.decay_duration}, sustain={self.sustain_level}, "
            f"release={self.release_duration})"
        )

    def _get_samples_iterator(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False
    ) -> np.ndarray:
        """Generate n samples using Python iterator (reference implementation).

        This is the reference implementation that shows how ADSR works step-by-step.
        It is much slower than vectorized mode (75x slower) but useful for:
        - Educational purposes
        - Debugging envelope behavior
        - Verifying vectorized implementation correctness

        For production use, use get_samples() or get_samples_vectorized() instead.

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            reset: If True, reset the envelope to initial state before generating.

        Returns:
            np.ndarray: Array of `n` consecutive samples.

        Note:
            This method is ~75x slower than get_samples_vectorized().
            Use only for testing, debugging, or educational purposes.
        """
        n = validate_sample_count(n)
        if reset:
            iter(self)
        return np.array([next(self) for _ in range(n)], np.float32)

    def _get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using true vectorized NumPy computation.

        This is a fully vectorized implementation that computes ADSR envelope
        phases using NumPy operations, providing speedup over iterator.
        Properly handles state continuity across calls and all ADSR phases.

        Args:
            n: Number of samples to produce.

        Returns:
            np.ndarray: Array of `n` consecutive envelope values.

        Note:
            Maintains state continuity by tracking current phase and position.
            Supports mid-envelope calls, release phase, and phase transitions.
        """

        n = validate_sample_count(n)
        # Initialize stepper if needed
        if self._stepper is None and self._phase not in (
            ADSRPhase.IDLE,
            ADSRPhase.RETRIGGER_RESET,
        ):
            iter(self)

        samples = np.zeros(n, dtype=np.float32)
        idx = 0
        remaining = n

        # Use pre-computed phase durations (optimized - no recalculation)
        attack_samples = self._attack_samples
        decay_samples = self._decay_samples
        release_samples = self._release_samples

        # Process samples through current and subsequent phases
        while remaining > 0 and not self.ended:
            if self._phase == ADSRPhase.IDLE:
                # Idle phase: output zeros until triggered
                samples[idx : idx + remaining] = 0.0
                self.val = 0.0
                break  # Stay in idle, don't advance

            if self._phase == ADSRPhase.RETRIGGER_RESET:
                samples_in_phase = self._retrigger_reset_samples - self._phase_position
                chunk_size = min(remaining, samples_in_phase)

                if chunk_size > 0:
                    start_progress = (
                        self._phase_position / self._retrigger_reset_samples
                    )
                    end_progress = (
                        self._phase_position + chunk_size
                    ) / self._retrigger_reset_samples
                    start_val = self._retrigger_reset_start_value * (
                        1.0 - start_progress
                    )
                    end_val = self._retrigger_reset_start_value * (1.0 - end_progress)
                    samples[idx : idx + chunk_size] = np.linspace(
                        start_val,
                        end_val,
                        chunk_size,
                        endpoint=False,
                        dtype=np.float32,
                    )

                    self._phase_position += chunk_size
                    self.val = samples[idx + chunk_size - 1]
                    idx += chunk_size
                    remaining -= chunk_size

                if self._phase_position >= self._retrigger_reset_samples:
                    self._attack_start_value = 0.0
                    self.val = 0.0
                    self._phase = ADSRPhase.ATTACK
                    self._phase_position = 0
                    self._stepper = self._get_ads_stepper()

            if self._phase == ADSRPhase.ATTACK:
                # Attack phase: current level -> 1
                if attack_samples > 0:
                    samples_in_phase = attack_samples - self._phase_position
                    chunk_size = min(remaining, samples_in_phase)

                    if chunk_size > 0:
                        # Generate attack curve
                        start_progress = self._phase_position / attack_samples
                        end_progress = (
                            self._phase_position + chunk_size
                        ) / attack_samples
                        attack_range = 1.0 - self._attack_start_value
                        start_val = self._attack_start_value + (
                            start_progress * attack_range
                        )
                        end_val = self._attack_start_value + (
                            end_progress * attack_range
                        )
                        samples[idx : idx + chunk_size] = np.linspace(
                            start_val,
                            end_val,
                            chunk_size,
                            endpoint=False,
                            dtype=np.float32,
                        )

                        self._phase_position += chunk_size
                        self.val = samples[idx + chunk_size - 1]
                        idx += chunk_size
                        remaining -= chunk_size

                    # Check if attack phase completed
                    if self._phase_position >= attack_samples:
                        self._phase = ADSRPhase.DECAY
                        self._phase_position = 0
                        self.val = 1.0
                        self._attack_start_value = 0.0
                else:
                    # Zero attack time, skip to decay
                    self._phase = ADSRPhase.DECAY
                    self._phase_position = 0
                    self.val = 1.0
                    self._attack_start_value = 0.0

            elif self._phase == ADSRPhase.DECAY:
                # Decay phase: 1 -> sustain_level
                if decay_samples > 0:
                    samples_in_phase = decay_samples - self._phase_position
                    chunk_size = min(remaining, samples_in_phase)

                    if chunk_size > 0:
                        # Generate decay curve
                        start_val = 1.0 - (self._phase_position / decay_samples) * (
                            1.0 - self.sustain_level
                        )
                        end_val = 1.0 - (
                            (self._phase_position + chunk_size) / decay_samples
                        ) * (1.0 - self.sustain_level)
                        samples[idx : idx + chunk_size] = np.linspace(
                            start_val,
                            end_val,
                            chunk_size,
                            endpoint=False,
                            dtype=np.float32,
                        )

                        self._phase_position += chunk_size
                        self.val = samples[idx + chunk_size - 1]
                        idx += chunk_size
                        remaining -= chunk_size

                    # Check if decay phase completed
                    if self._phase_position >= decay_samples:
                        self._phase = ADSRPhase.SUSTAIN
                        self._phase_position = 0
                        self.val = self.sustain_level
                else:
                    # Zero decay time, skip to sustain
                    self._phase = ADSRPhase.SUSTAIN
                    self._phase_position = 0
                    self.val = self.sustain_level

            elif self._phase == ADSRPhase.SUSTAIN:
                # Sustain phase: hold at sustain_level
                samples[idx : idx + remaining] = self.sustain_level
                self.val = self.sustain_level
                self._phase_position += remaining
                idx += remaining
                remaining = 0

            elif self._phase == ADSRPhase.RELEASE:
                # Release phase: current value -> 0
                if release_samples > 0:
                    samples_in_phase = release_samples - self._phase_position
                    chunk_size = min(remaining, samples_in_phase)

                    if chunk_size > 0:
                        # Generate release curve from current val to 0
                        # Note: val is set when trigger_release() is called
                        start_val = self._release_start_value * (
                            1.0 - self._phase_position / release_samples
                        )
                        end_val = self._release_start_value * (
                            1.0 - (self._phase_position + chunk_size) / release_samples
                        )
                        samples[idx : idx + chunk_size] = np.linspace(
                            start_val,
                            end_val,
                            chunk_size,
                            endpoint=False,
                            dtype=np.float32,
                        )

                        self._phase_position += chunk_size
                        self.val = samples[idx + chunk_size - 1]
                        idx += chunk_size
                        remaining -= chunk_size

                    # Check if release phase completed
                    if self._phase_position >= release_samples:
                        self.ended = True
                        self.val = 0.0
                        # Fill remaining with zeros
                        if remaining > 0:
                            samples[idx : idx + remaining] = 0.0
                            idx += remaining
                            remaining = 0
                else:
                    # Zero release time, end immediately
                    self.ended = True
                    self.val = 0.0
                    if remaining > 0:
                        samples[idx : idx + remaining] = 0.0
                        idx += remaining
                        remaining = 0

        # If envelope ended, fill remaining with zeros
        if self.ended and remaining > 0:
            samples[idx:] = 0.0

        return samples

    def _update_phase_samples(self):
        """Update pre-computed phase sample counts.

        Called automatically when duration or sample_rate properties change.
        """
        self._attack_samples = int(self._attack_duration * self._sample_rate)
        self._decay_samples = int(self._decay_duration * self._sample_rate)
        self._release_samples = int(self._release_duration * self._sample_rate)


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
    adsr = ADSREnvelope(a, d, sl, r, sample_rate)
    adsr.trigger_note_on()  # Trigger envelope to start attack phase
    down_len = int(sum([a, d, sd]) * sample_rate)
    up_len = int(r * sample_rate)
    iter(adsr)
    adsr_vals = adsr.get_samples(down_len)
    adsr.trigger_release()
    adsr_vals = np.concatenate([adsr_vals, adsr.get_samples(up_len)])
    return adsr_vals, down_len, up_len
