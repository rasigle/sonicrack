"""Envelope generators and modulators for audio synthesis.

This module provides components that generate time-varying control signals
used to modulate audio parameters such as amplitude, frequency, and filters.
The primary implementation is the ADSR (Attack-Decay-Sustain-Release) envelope,
a fundamental building block in subtractive synthesis.

Classes:
    Modulator: Abstract base class for all modulators.
    ADSREnvelope: Classic ADSR envelope generator with configurable phases.

Functions:
    getadsr: Convenience function to generate complete ADSR envelope arrays.

Example:
    >>> # Create an ADSR envelope
    >>> env = ADSREnvelope(
    ...     attack_duration=0.1,
    ...     decay_duration=0.2,
    ...     sustain_level=0.7,
    ...     release_duration=0.3
    ... )
    >>>
    >>> # Generate envelope values
    >>> samples = env.get_samples(1000)
    >>>
    >>> # Trigger release phase
    >>> env.trigger_release()
    >>> release_samples = env.get_samples(500)
    >>>
    >>> # Check if envelope has completed
    >>> if env.ended:
    ...     print("Envelope finished")

Typical Use:
    ADSR envelopes are commonly used with ModulatedOscillator to create
    expressive synthesis voices with natural attack and decay characteristics.

Note:
    Modulators maintain internal state and should be reset (via iteration)
    when reusing for multiple notes or synthesis events.
"""

import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.audio_component import AudioComponent, ComponentDescriptor, Generator
from src.engine.audio_component_registry import ComponentCategory, register_component
from src.engine.validation import validate_sample_count, validate_sample_rate
from src.utils.logging_config import get_engine_logger

logger = get_engine_logger("modulator")


class Modulator(Generator):
    """Base class for all modulators.

    Modulators generate time-varying control signals used to modulate
    parameters of other audio components (e.g., amplitude, frequency).
    """

    descriptor = ComponentDescriptor(
        name="Modulator",
        category=ComponentCategory.MODULATOR,
        description="Base class for modulators",
        tags=["modulator", "base"],
        config_params=["sample_rate"],
        fluent_api_name="modulator",
    )

    def __init__(self, sample_rate: float = DEFAULT_SAMPLE_RATE):
        """Initialize a new Modulator instance.

        Args:
            sample_rate : the sample rate at which the notes are to be consumed.
        """
        super().__init__(sample_rate=sample_rate)


@register_component()
class ADSREnvelope(Modulator):
    """A simple ADSR envelope with the four stages attack, decay, release and sustain.

    Has `.trigger_release()` implemented to trigger the release stage of the envelope.
    similarly has `.ended`, a flag to indicate the end of the release stage.
    """

    descriptor = ComponentDescriptor(
        name="ADSREnvelope",
        category=ComponentCategory.MODULATOR,
        description="ADSR envelope generator",
        tags=["envelope", "modulator", "adsr"],
        config_params=[
            "attack_duration",
            "decay_duration",
            "sustain_level",
            "release_duration",
            "sample_rate",
        ],
        fluent_api_name="adsr",
    )

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
        # Store as private attributes - access through properties
        self._attack_duration = attack_duration
        self._decay_duration = decay_duration
        self.sustain_level = sustain_level
        self._release_duration = release_duration
        self._sample_rate = validate_sample_rate(sample_rate)
        super().__init__(sample_rate=sample_rate)

        self.ended = True  # Start in ended state
        self.val: float = 0.0  # Initialize current value

        # Pre-compute phase durations in samples (performance optimization)
        # These will be set by _update_phase_samples()
        self._attack_samples = 0
        self._decay_samples = 0
        self._release_samples = 0
        self._update_phase_samples()

        # Vectorization state tracking
        # Start in idle/ended state, not attack! (prevents spurious triggers)
        self._phase = (
            "idle"  # Current phase: 'idle', 'attack', 'decay', 'sustain', 'release'
        )
        self._phase_position = 0  # Position within current phase (in samples)
        self._stepper: Iterator[float] | None = None

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
        stepper = None
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
                if stepper is None:
                    return
                val = next(stepper)
            yield val

    def __iter__(self):
        # Only initialize stepper if not in idle state
        # This prevents spurious triggers when creating iterator
        if self._phase != "idle":
            self.val = 0.0
            self.ended = False
            self._stepper = self._get_ads_stepper()
            self._phase = "attack"
        self._phase_position = 0
        return self

    def __next__(self):
        # Handle idle state
        if self._phase == "idle":
            self.val = 0.0
            return 0.0

        # Ensure stepper exists
        if self._stepper is None:
            self._stepper = self._get_ads_stepper()
        assert self._stepper is not None

        self.val = next(self._stepper)
        self._phase_position += 1

        # Update phase tracking using pre-computed values (optimized)
        if self._phase == "attack" and self._phase_position >= self._attack_samples:
            self._phase = "decay"
            self._phase_position = 0
        elif self._phase == "decay" and self._phase_position >= self._decay_samples:
            self._phase = "sustain"
            self._phase_position = 0

        return self.val

    def trigger_release(self):
        """Trigger the release phase of the envelope."""
        self._stepper = self._get_r_stepper()
        self._phase = "release"
        self._phase_position = 0

    def trigger_note_on(self):
        """Trigger note on - resets envelope to attack phase.

        This is an alias for resetting the envelope, compatible with MIDI note on.
        Transitions from idle state to attack phase.
        """
        self.ended = False
        self._phase = "attack"
        self._phase_position = 0
        self.val = 0

    def trigger_note_off(self):
        """Trigger note off - starts release phase.

        This is an alias for trigger_release(), compatible with MIDI note off.
        """
        self.trigger_release()

    def get_samples(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False, mode: str = "auto"
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
        if mode not in ("auto", "iterator", "vectorized"):
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
        if self._stepper is None:
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

            if self._phase == "idle":
                # Idle phase: output zeros until triggered
                samples[idx : idx + remaining] = 0.0
                self.val = 0.0
                break  # Stay in idle, don't advance

            if self._phase == "attack":
                # Attack phase: 0 -> 1
                if attack_samples > 0:
                    samples_in_phase = attack_samples - self._phase_position
                    chunk_size = min(remaining, samples_in_phase)

                    if chunk_size > 0:
                        # Generate attack curve
                        start_val = self._phase_position / attack_samples
                        end_val = (self._phase_position + chunk_size) / attack_samples
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
                        self._phase = "decay"
                        self._phase_position = 0
                        self.val = 1.0
                else:
                    # Zero attack time, skip to decay
                    self._phase = "decay"
                    self._phase_position = 0
                    self.val = 1.0

            elif self._phase == "decay":
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
                        self._phase = "sustain"
                        self._phase_position = 0
                        self.val = self.sustain_level
                else:
                    # Zero decay time, skip to sustain
                    self._phase = "sustain"
                    self._phase_position = 0
                    self.val = self.sustain_level

            elif self._phase == "sustain":
                # Sustain phase: hold at sustain_level
                samples[idx : idx + remaining] = self.sustain_level
                self.val = self.sustain_level
                self._phase_position += remaining
                idx += remaining
                remaining = 0

            elif self._phase == "release":
                # Release phase: current value -> 0
                if release_samples > 0:
                    samples_in_phase = release_samples - self._phase_position
                    chunk_size = min(remaining, samples_in_phase)

                    if chunk_size > 0:
                        # Generate release curve from current val to 0
                        # Note: val is set when trigger_release() is called
                        start_val = self.val * (
                            1.0 - self._phase_position / release_samples
                        )
                        end_val = self.val * (
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


class GateTriggeredADSR(Modulator):
    """ADSR envelope triggered by a gate signal.

    This wrapper monitors a gate signal (0.0 or 1.0) and triggers the ADSR
    envelope accordingly:
    - Gate 0→1 transition: Trigger note on (attack phase)
    - Gate 1→0 transition: Trigger note off (release phase)

    Perfect for MIDI keyboard control where the gate signal comes from
    MIDI Input [Gate] output.

    Example:
        >>> # Example gate source object with a get_samples() method omitted for brevity
        >>> adsr = ADSREnvelope(attack_duration=0.1, release_duration=0.3)
        >>> gate_source = ...
        >>> gate_adsr = GateTriggeredADSR(adsr, gate_source)
        >>> samples = gate_adsr.get_samples(1000)
    """

    def __init__(self, adsr_envelope: ADSREnvelope, gate_source: AudioComponent):
        """Initialize gate-triggered ADSR.

        Args:
            adsr_envelope: The ADSR envelope to trigger
            gate_source: Component that provides gate signal
                (must have get_samples method)
        """
        super().__init__()
        self.adsr = adsr_envelope
        self.gate_source = gate_source

        # Initialize previous_gate from current gate state to prevent false triggers
        # Try multiple methods to get initial gate value
        initial_gate = 0.0
        if hasattr(gate_source, "cv_converter"):
            # For CVGateOutput
            initial_gate = gate_source.cv_converter.gate
        elif hasattr(gate_source, "gate"):
            # For other gate sources with gate property
            initial_gate = gate_source.gate

        self.previous_gate = float(initial_gate)
        self.ended = self.adsr.ended

        logger.debug(
            f"GateTriggeredADSR initialized with initial gate={self.previous_gate}"
        )

    def get_samples(self, n: int, *args: Any, **kwargs: Any) -> np.ndarray:
        """Generate envelope samples, checking gate for triggers.

        Args:
            n: Number of samples to generate
            **kwargs: Additional arguments (ignored, for compatibility with
                ModulatedVolume)

        Returns:
            Envelope output (0.0 to 1.0)
        """
        n = validate_sample_count(n)
        # Get gate signal
        if hasattr(self.gate_source, "get_gate_samples"):
            # For CV converters with specialized gate method
            gate_samples = self.gate_source.get_gate_samples(n)
        elif hasattr(self.gate_source, "get_samples"):
            # For generic audio components
            gate_samples = self.gate_source.get_samples(n)
        else:
            logger.warning("Gate source has no get_samples method")
            gate_samples = np.zeros(n)

        # Check for gate transitions (look at first sample for now)
        # In a more sophisticated implementation, we'd check each sample
        current_gate = gate_samples[0] if len(gate_samples) > 0 else 0.0

        # Detect rising edge (note on) - require transition from clearly low to clearly
        # high
        if self.previous_gate < 0.3 and current_gate > 0.7:
            logger.debug("Gate rising edge detected - triggering note on")
            self.adsr.trigger_note_on()

        # Detect falling edge (note off) - require transition from clearly high to
        # clearly low
        elif self.previous_gate > 0.7 and current_gate < 0.3:
            logger.debug("Gate falling edge detected - triggering note off")
            self.adsr.trigger_note_off()

        self.previous_gate = current_gate

        # Generate ADSR envelope samples
        _ = args, kwargs
        samples = self.adsr.get_samples(n)
        self.ended = self.adsr.ended
        return samples

    def __iter__(self):
        """Make GateTriggeredADSR iterable for use as modulator."""
        # Reset gate state
        self.previous_gate = 0.0

        # Initialize gate source iterator if it has __iter__
        if hasattr(self.gate_source, "__iter__"):
            iter(self.gate_source)

        # Initialize ADSR iterator
        if hasattr(self.adsr, "__iter__"):
            iter(self.adsr)
        self.ended = self.adsr.ended
        return self

    def __next__(self) -> float:
        """Get next envelope value, checking gate for triggers.

        Returns:
            Current envelope value (0.0 to 1.0)
        """
        # Get current gate value
        if hasattr(self.gate_source, "__next__"):
            current_gate = next(self.gate_source)
        elif hasattr(self.gate_source, "gate"):
            # For CV converters with gate property
            current_gate = self.gate_source.gate
        else:
            current_gate = 0.0

        # Use hysteresis for gate detection to prevent false triggers
        # Detect rising edge (note on) - require clear transition
        if self.previous_gate < 0.3 and current_gate > 0.7:
            logger.debug("Gate rising edge detected - triggering note on")
            self.adsr.trigger_note_on()

        # Detect falling edge (note off) - require clear transition
        elif self.previous_gate > 0.7 and current_gate < 0.3:
            logger.debug("Gate falling edge detected - triggering note off")
            self.adsr.trigger_note_off()

        self.previous_gate = current_gate

        # Get next ADSR value
        if hasattr(self.adsr, "__next__"):
            value = next(self.adsr)
            self.ended = self.adsr.ended
            return value

        # Fallback to get_samples
        samples = self.adsr.get_samples(1)
        self.ended = self.adsr.ended
        return samples[0] if len(samples) > 0 else 0.0

    def trigger_note_on(self):
        """Manually trigger note on."""
        self.adsr.trigger_note_on()
        self.ended = self.adsr.ended

    def trigger_note_off(self):
        """Manually trigger note off."""
        self.adsr.trigger_note_off()
        self.ended = self.adsr.ended

    # Parameter forwarding for hot-swap support
    @property
    def attack_duration(self) -> float:
        """Get attack duration from underlying ADSR."""
        return self.adsr.attack_duration

    @attack_duration.setter
    def attack_duration(self, value: float):
        """Set attack duration on underlying ADSR."""
        self.adsr.attack_duration = value

    @property
    def decay_duration(self) -> float:
        """Get decay duration from underlying ADSR."""
        return self.adsr.decay_duration

    @decay_duration.setter
    def decay_duration(self, value: float):
        """Set decay duration on underlying ADSR."""
        self.adsr.decay_duration = value

    @property
    def sustain_level(self) -> float:
        """Get sustain level from underlying ADSR."""
        return self.adsr.sustain_level

    @sustain_level.setter
    def sustain_level(self, value: float):
        """Set sustain level on underlying ADSR."""
        self.adsr.sustain_level = value

    @property
    def release_duration(self) -> float:
        """Get release duration from underlying ADSR."""
        return self.adsr.release_duration

    @release_duration.setter
    def release_duration(self, value: float):
        """Set release duration on underlying ADSR."""
        self.adsr.release_duration = value
