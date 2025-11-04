"""Gate-triggered ADSR envelope wrapper.

This component wraps an ADSR envelope and triggers it based on a gate signal
(0 or 1) from another component like MIDI Input.
"""

import numpy as np
import logging

from src.engine.audio_component import AudioComponent
from src.engine.modulator import ADSREnvelope

logger = logging.getLogger(__name__)


class GateTriggeredADSR(AudioComponent):
    """ADSR envelope triggered by a gate signal.

    This wrapper monitors a gate signal (0.0 or 1.0) and triggers the ADSR
    envelope accordingly:
    - Gate 0→1 transition: Trigger note on (attack phase)
    - Gate 1→0 transition: Trigger note off (release phase)

    Perfect for MIDI keyboard control where the gate signal comes from
    MIDI Input [Gate] output.

    Example:
        >>> from src.engine.midi import MIDIToCV, NoteOnMessage
        >>>
        >>> # Create MIDI to CV and ADSR
        >>> cv = MIDIToCV()
        >>> adsr = ADSREnvelope(attack_duration=0.1, release_duration=0.3)
        >>> gate_adsr = GateTriggeredADSR(adsr, cv.get_gate_samples)
        >>>
        >>> # Trigger with MIDI
        >>> cv.process_message(NoteOnMessage(0.0, 0, 60, 100))
        >>> samples = gate_adsr.get_samples(1000)  # ADSR attacks
        >>>
        >>> cv.process_message(NoteOffMessage(0.1, 0, 60))
        >>> samples = gate_adsr.get_samples(1000)  # ADSR releases
    """

    def __init__(self, adsr_envelope: ADSREnvelope, gate_source: AudioComponent):
        """Initialize gate-triggered ADSR.

        Args:
            adsr_envelope: The ADSR envelope to trigger
            gate_source: Component that provides gate signal (must have get_samples method)
        """
        super().__init__()
        self.adsr = adsr_envelope
        self.gate_source = gate_source

        # Initialize previous_gate from current gate state to prevent false triggers
        # Try multiple methods to get initial gate value
        initial_gate = 0.0
        if hasattr(gate_source, 'cv_converter'):
            # For CVGateOutput
            initial_gate = gate_source.cv_converter.gate
        elif hasattr(gate_source, 'gate'):
            # For other gate sources with gate property
            initial_gate = gate_source.gate

        self.previous_gate = initial_gate

        logger.debug(f"GateTriggeredADSR initialized with initial gate={self.previous_gate}")

    def get_samples(self, num_samples: int, **kwargs) -> np.ndarray:
        """Generate envelope samples, checking gate for triggers.

        Args:
            num_samples: Number of samples to generate
            **kwargs: Additional arguments (ignored, for compatibility with ModulatedVolume)

        Returns:
            Envelope output (0.0 to 1.0)
        """
        # Get gate signal
        if hasattr(self.gate_source, 'get_gate_samples'):
            # For CV converters with specialized gate method
            gate_samples = self.gate_source.get_gate_samples(num_samples)
        elif hasattr(self.gate_source, 'get_samples'):
            # For generic audio components
            gate_samples = self.gate_source.get_samples(num_samples)
        else:
            logger.warning("Gate source has no get_samples method")
            gate_samples = np.zeros(num_samples)

        # Check for gate transitions (look at first sample for now)
        # In a more sophisticated implementation, we'd check each sample
        current_gate = gate_samples[0] if len(gate_samples) > 0 else 0.0

        # Use hysteresis for gate detection to prevent false triggers from noise
        # Threshold: 0.5 for detection, but require significant change
        GATE_THRESHOLD = 0.5

        # Detect rising edge (note on) - require transition from clearly low to clearly high
        if self.previous_gate < 0.3 and current_gate > 0.7:
            logger.debug("Gate rising edge detected - triggering note on")
            self.adsr.trigger_note_on()

        # Detect falling edge (note off) - require transition from clearly high to clearly low
        elif self.previous_gate > 0.7 and current_gate < 0.3:
            logger.debug("Gate falling edge detected - triggering note off")
            self.adsr.trigger_note_off()

        self.previous_gate = current_gate

        # Generate ADSR envelope samples
        return self.adsr.get_samples(num_samples)

    def __iter__(self):
        """Make GateTriggeredADSR iterable for use as modulator."""
        # Reset gate state
        self.previous_gate = 0.0

        # Initialize gate source iterator if it has __iter__
        if hasattr(self.gate_source, '__iter__'):
            iter(self.gate_source)

        # Initialize ADSR iterator
        if hasattr(self.adsr, '__iter__'):
            iter(self.adsr)
        return self

    def __next__(self) -> float:
        """Get next envelope value, checking gate for triggers.

        Returns:
            Current envelope value (0.0 to 1.0)
        """
        # Get current gate value
        if hasattr(self.gate_source, '__next__'):
            current_gate = next(self.gate_source)
        elif hasattr(self.gate_source, 'gate'):
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
        if hasattr(self.adsr, '__next__'):
            return next(self.adsr)
        else:
            # Fallback to get_samples
            samples = self.adsr.get_samples(1)
            return samples[0] if len(samples) > 0 else 0.0

    @property
    def ended(self) -> bool:
        """Check if envelope has ended."""
        return self.adsr.ended

    def trigger_note_on(self):
        """Manually trigger note on."""
        self.adsr.trigger_note_on()

    def trigger_note_off(self):
        """Manually trigger note off."""
        self.adsr.trigger_note_off()

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

