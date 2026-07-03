"""MIDI to Control Voltage (CV) converter.

This module converts MIDI messages to control voltage signals that can be used
to control oscillators and other synthesis parameters in the modular system.

CV Outputs:
    - Gate: 0 or 1 signal indicating note on/off
    - Pitch: 1V/oct pitch CV, with 0V = C4
    - Velocity: 0.0 to 1.0 normalized velocity
    - Mod Wheel: 0.0 to 1.0 from CC#1
    - Expression: 0.0 to 1.0 from CC#11

Example:
    >>> from src.midi_io import MIDIToCV, NoteOnMessage
    >>>
    >>> converter = MIDIToCV()
    >>> msg = NoteOnMessage(timestamp=0.0, channel=0, note=60, velocity=100)
    >>> converter.process_message(msg)
    >>>
    >>> # Get current state
    >>> print(converter.gate)        # 1.0 (note is on)
    >>> print(converter.pitch_cv)    # 0.0V (middle C)
    >>> print(converter.velocity)    # 0.787 (100/127)
"""

import logging
from typing import Any

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.component import AudioComponent
from src.engine.utils.cv import midi_note_to_pitch_cv, pitch_cv_to_frequency
from src.midi_io.messages import (
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
        - pitch_cv: 1V/oct pitch CV for current note
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
        >>> assert np.all(cv_samples == converter.pitch_cv)
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
        self.pitch_cv: float = float(midi_note_to_pitch_cv(69))  # A4 default
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

        logger.debug(
            "Note ON: %s, pitch_cv=%.3fV, freq=%.2fHz",
            msg.note,
            self.pitch_cv,
            self.frequency,
        )

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

    @property
    def frequency(self) -> float:
        """Frequency in Hz derived from the current 1V/oct pitch CV."""
        return float(pitch_cv_to_frequency(self.pitch_cv))

    def _update_pitch_cv(self):
        """Update pitch CV based on current note and pitch bend."""
        if self.current_note is not None:
            note_with_bend = float(self.current_note) + self.pitch_bend
            self.pitch_cv = float(midi_note_to_pitch_cv(note_with_bend))

    def _update_frequency(self):
        """Compatibility alias for older call sites."""
        self._update_pitch_cv()

    def get_samples(self, n: int, *args: Any, **kwargs: Any) -> np.ndarray:
        """Generate constant CV output.

        This returns pitch CV volts as a constant signal. For gate/velocity/etc,
        access the attributes directly.

        Args:
            n: Number of samples to generate

        Returns:
            Array of 1V/oct pitch CV values (constant)
        """
        return np.full(n, self.pitch_cv, dtype=np.float32)

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
        self.pitch_cv = float(midi_note_to_pitch_cv(69))
        self.velocity = 0.0
        self.mod_wheel = 0.0
        self.expression = 1.0
        self.pitch_bend = 0.0
        self.current_note = None
        logger.debug("MIDIToCV reset")
