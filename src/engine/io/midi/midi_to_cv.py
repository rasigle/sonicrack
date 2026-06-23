"""MIDI to Control Voltage (CV) converter.

This module converts MIDI messages to control voltage signals that can be used
to control oscillators and other synthesis parameters in the modular system.

CV Outputs:
    - Gate: 0 or 1 signal indicating note on/off
    - Pitch: Frequency in Hz corresponding to MIDI note
    - Velocity: 0.0 to 1.0 normalized velocity
    - Mod Wheel: 0.0 to 1.0 from CC#1
    - Expression: 0.0 to 1.0 from CC#11

Example:
    >>> from engine.io.midi import MIDIToCV, NoteOnMessage
    >>>
    >>> converter = MIDIToCV()
    >>> msg = NoteOnMessage(timestamp=0.0, channel=0, note=60, velocity=100)
    >>> converter.process_message(msg)
    >>>
    >>> # Get current state
    >>> print(converter.gate)        # 1.0 (note is on)
    >>> print(converter.frequency)   # 261.63 Hz (middle C)
    >>> print(converter.velocity)    # 0.787 (100/127)
"""

import logging

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.audio_component import AudioComponent
from src.engine.io.midi.messages import (
    ControlChangeMessage,
    MIDIMessage,
    NoteOffMessage,
    NoteOnMessage,
    PitchBendMessage,
)

logger = logging.getLogger(__name__)


class MIDIToCV(AudioComponent):
    """Convert MIDI messages to control voltage outputs.

    This component maintains the current state of MIDI controls and
    generates constant CV signals based on the most recent messages.

    Outputs:
        - gate: 1.0 when note is on, 0.0 when off
        - frequency: Frequency in Hz of current note
        - velocity: Normalized velocity (0.0-1.0)
        - mod_wheel: CC#1 value (0.0-1.0)
        - expression: CC#11 value (0.0-1.0)
        - pitch_bend: Pitch bend in semitones (-2.0 to +2.0 by default)

    Example:
        >>> converter = MIDIToCV()
        >>>
        >>> # Process note on
        >>> msg = NoteOnMessage(0.0, 0, 60, 100)
        >>> converter.process_message(msg)
        >>>
        >>> # Generate CV signal (constant values)
        >>> cv_samples = converter.get_samples(1000)
        >>> assert np.all(cv_samples == converter.frequency)
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        pitch_bend_range: float = 2.0,
    ):
        """Initialize MIDI to CV converter.

        Args:
            sample_rate: Audio sample rate
            pitch_bend_range: Pitch bend range in semitones (default ±2)
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.pitch_bend_range = pitch_bend_range

        # Current state
        self.gate: float = 0.0
        self.frequency: float = 440.0  # A4 default
        self.velocity: float = 0.0
        self.mod_wheel: float = 0.0
        self.expression: float = 1.0
        self.pitch_bend: float = 0.0  # In semitones
        self.current_note: int | None = None

        logger.debug("MIDIToCV initialized")

    def process_message(self, msg: MIDIMessage):
        """Process a MIDI message and update CV state.

        Args:
            msg: MIDI message to process
        """
        if isinstance(msg, NoteOnMessage):
            self._handle_note_on(msg)
        elif isinstance(msg, NoteOffMessage):
            self._handle_note_off(msg)
        elif isinstance(msg, ControlChangeMessage):
            self._handle_cc(msg)
        elif isinstance(msg, PitchBendMessage):
            self._handle_pitch_bend(msg)

    def _handle_note_on(self, msg: NoteOnMessage):
        """Handle note on message."""
        # Velocity 0 is note off
        if msg.velocity == 0:
            self._handle_note_off(NoteOffMessage(msg.timestamp, msg.channel, msg.note))
            return

        self.current_note = msg.note
        self.gate = 1.0
        self.velocity = msg.normalize_velocity()
        self._update_frequency()

        logger.debug(f"Note ON: {msg.note}, freq={self.frequency:.2f}Hz")

    def _handle_note_off(self, msg: NoteOffMessage):
        """Handle note off message."""
        # Only turn off gate if this is the current note
        if msg.note == self.current_note:
            self.gate = 0.0
            logger.debug(f"Note OFF: {msg.note}")

    def _handle_cc(self, msg: ControlChangeMessage):
        """Handle control change message."""
        normalized = msg.normalize_value()

        if msg.controller == 1:  # Mod wheel
            self.mod_wheel = normalized
            logger.debug(f"Mod Wheel: {self.mod_wheel:.3f}")
        elif msg.controller == 11:  # Expression
            self.expression = normalized
            logger.debug(f"Expression: {self.expression:.3f}")

    def _handle_pitch_bend(self, msg: PitchBendMessage):
        """Handle pitch bend message."""
        self.pitch_bend = msg.to_semitones(int(self.pitch_bend_range))
        self._update_frequency()
        logger.debug(f"Pitch Bend: {self.pitch_bend:.3f} semitones")

    def _update_frequency(self):
        """Update frequency based on current note and pitch bend."""
        if self.current_note is not None:
            # Apply pitch bend - midi_to_frequency formula works with float
            note_with_bend = float(self.current_note) + self.pitch_bend
            # The formula works fine with fractional MIDI notes for microtonal tuning
            self.frequency = 440.0 * (2 ** ((note_with_bend - 69) / 12))

    def get_samples(self, num_samples: int) -> np.ndarray:
        """Generate constant CV output.

        This returns the frequency as a constant signal. For gate/velocity/etc,
        access the attributes directly.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Array of frequency values (constant)
        """
        return np.full(num_samples, self.frequency, dtype=np.float32)

    def get_gate_samples(self, num_samples: int) -> np.ndarray:
        """Get gate signal (0 or 1).

        Returns:
            Array of gate values (constant 0.0 or 1.0)
        """
        return np.full(num_samples, self.gate, dtype=np.float32)

    def get_velocity_samples(self, num_samples: int) -> np.ndarray:
        """Get velocity signal (0.0 to 1.0).

        Returns:
            Array of velocity values (constant)
        """
        return np.full(num_samples, self.velocity, dtype=np.float32)

    def reset(self):
        """Reset all CV outputs to default state."""
        self.gate = 0.0
        self.frequency = 440.0
        self.velocity = 0.0
        self.mod_wheel = 0.0
        self.expression = 1.0
        self.pitch_bend = 0.0
        self.current_note = None
        logger.debug("MIDIToCV reset")
