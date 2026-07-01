"""Gate-triggered ADSR envelope wrapper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from src.engine.dsp.modulators.base import Modulator
from src.engine.utils.validation import validate_sample_count

if TYPE_CHECKING:
    from src.engine.core.component import AudioComponent
    from src.engine.dsp.modulators.adsr_envelope import ADSREnvelope


logger = logging.getLogger(__name__)


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
