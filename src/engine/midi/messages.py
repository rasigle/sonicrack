"""MIDI message classes and types.

This module defines the core MIDI message types used throughout the MIDI system.
All messages include timestamp and channel information.

Standard MIDI message types:
- Note On: Trigger a note with velocity
- Note Off: Release a note
- Control Change (CC): Modify continuous parameters
- Pitch Bend: Pitch wheel modulation
- Program Change: Select instrument/preset
- Aftertouch: Pressure sensitivity

Note Numbers:
- MIDI uses note numbers 0-127
- Middle C (C4) = 60
- A4 (440 Hz) = 69
- Each octave spans 12 semitones

Velocity:
- 0-127 where 0 = silent, 127 = maximum
- Note On with velocity 0 = Note Off (per MIDI spec)

CC Numbers:
- 0-127 for different parameters
- Common: 1 (Mod Wheel), 7 (Volume), 10 (Pan), 11 (Expression)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MIDIMessage:
    """Base class for all MIDI messages.

    Attributes:
        timestamp: Time in seconds when message occurred
        channel: MIDI channel (0-15, channels 1-16 in user interface)
    """

    timestamp: float
    channel: int = 0

    def __post_init__(self):
        """Validate channel range."""
        if not 0 <= self.channel <= 15:
            raise ValueError(f"MIDI channel must be 0-15, got {self.channel}")


@dataclass
class NoteOnMessage(MIDIMessage):
    """MIDI Note On message.

    Triggers a note to start playing with specified velocity.

    Attributes:
        note: MIDI note number (0-127), where 60 = Middle C
        velocity: Note velocity (0-127), where 0 = silent, 127 = loudest
        timestamp: Time in seconds
        channel: MIDI channel (0-15)

    Example:
        >>> msg = NoteOnMessage(timestamp=0.0, note=60, velocity=100)
        >>> print(f"Note {msg.note} on at velocity {msg.velocity}")
        Note 60 on at velocity 100
    """

    note: int = 60
    velocity: int = 100

    def __post_init__(self):
        """Validate note and velocity ranges."""
        super().__post_init__()

        if not 0 <= self.note <= 127:
            raise ValueError(f"MIDI note must be 0-127, got {self.note}")

        if not 0 <= self.velocity <= 127:
            raise ValueError(f"MIDI velocity must be 0-127, got {self.velocity}")

    def to_frequency(self, a4_tuning: float = 440.0) -> float:
        """Convert note to frequency.

        Args:
            a4_tuning: Frequency of A4 in Hz (default 440 Hz)

        Returns:
            Frequency in Hz
        """
        return a4_tuning * 2 ** ((self.note - 69) / 12)

    def normalize_velocity(self) -> float:
        """Get velocity as normalized float 0.0-1.0.

        Returns:
            Velocity scaled to 0.0-1.0
        """
        return self.velocity / 127.0


@dataclass
class NoteOffMessage(MIDIMessage):
    """MIDI Note Off message.

    Triggers a note to stop playing (release).

    Attributes:
        note: MIDI note number (0-127)
        velocity: Release velocity (0-127), often unused
        timestamp: Time in seconds
        channel: MIDI channel (0-15)

    Example:
        >>> msg = NoteOffMessage(timestamp=1.5, note=60, velocity=64)
        >>> print(f"Note {msg.note} off")
        Note 60 off
    """

    note: int = 60
    velocity: int = 64  # Release velocity, often ignored

    def __post_init__(self):
        """Validate note and velocity ranges."""
        super().__post_init__()

        if not 0 <= self.note <= 127:
            raise ValueError(f"MIDI note must be 0-127, got {self.note}")

        if not 0 <= self.velocity <= 127:
            raise ValueError(f"MIDI velocity must be 0-127, got {self.velocity}")


@dataclass
class ControlChangeMessage(MIDIMessage):
    """MIDI Control Change (CC) message.

    Modifies continuous parameters like volume, pan, modulation, etc.

    Common CC Numbers:
        1: Modulation Wheel
        7: Volume
        10: Pan
        11: Expression
        64: Sustain Pedal (< 64 = off, >= 64 = on)
        74: Filter Cutoff (often)

    Attributes:
        controller: CC number (0-127)
        value: CC value (0-127)
        timestamp: Time in seconds
        channel: MIDI channel (0-15)

    Example:
        >>> # Modulation wheel to 50%
        >>> msg = ControlChangeMessage(timestamp=0.5, controller=1, value=64)
        >>> print(f"CC {msg.controller} = {msg.value}")
        CC 1 = 64
    """

    controller: int = 1  # CC number
    value: int = 0

    def __post_init__(self):
        """Validate controller and value ranges."""
        super().__post_init__()

        if not 0 <= self.controller <= 127:
            raise ValueError(f"CC controller must be 0-127, got {self.controller}")

        if not 0 <= self.value <= 127:
            raise ValueError(f"CC value must be 0-127, got {self.value}")

    def normalize_value(self) -> float:
        """Get CC value as normalized float 0.0-1.0.

        Returns:
            Value scaled to 0.0-1.0
        """
        return self.value / 127.0

    def is_switch_on(self, threshold: int = 64) -> bool:
        """Check if CC is interpreted as a switch in ON state.

        Common for sustain pedal and other on/off controllers.

        Args:
            threshold: Value above which switch is considered ON (default 64)

        Returns:
            True if value >= threshold, False otherwise
        """
        return self.value >= threshold


@dataclass
class PitchBendMessage(MIDIMessage):
    """MIDI Pitch Bend message.

    Modifies pitch (usually from pitch wheel).

    Attributes:
        value: Pitch bend value (-8192 to +8191, 0 = no bend)
        timestamp: Time in seconds
        channel: MIDI channel (0-15)

    Example:
        >>> msg = PitchBendMessage(timestamp=1.0, value=2048)
        >>> semitones = msg.to_semitones(bend_range=2)
        >>> print(f"Bend: {semitones} semitones")
        Bend: 0.5 semitones
    """

    value: int = 0  # -8192 to +8191

    def __post_init__(self):
        """Validate pitch bend value range."""
        super().__post_init__()

        if not -8192 <= self.value <= 8191:
            raise ValueError(f"Pitch bend must be -8192 to +8191, got {self.value}")

    def normalize_value(self) -> float:
        """Get pitch bend as normalized float -1.0 to +1.0.

        Returns:
            Value scaled to -1.0 to +1.0
        """
        return self.value / 8192.0

    def to_semitones(self, bend_range: int = 2) -> float:
        """Convert pitch bend to semitone shift.

        Args:
            bend_range: Pitch bend range in semitones (default 2, common setting)

        Returns:
            Pitch shift in semitones
        """
        return self.normalize_value() * bend_range


@dataclass
class ProgramChangeMessage(MIDIMessage):
    """MIDI Program Change message.

    Selects an instrument/preset (0-127).

    Attributes:
        program: Program number (0-127)
        timestamp: Time in seconds
        channel: MIDI channel (0-15)
    """

    program: int = 0

    def __post_init__(self):
        """Validate program number range."""
        super().__post_init__()

        if not 0 <= self.program <= 127:
            raise ValueError(f"Program must be 0-127, got {self.program}")


@dataclass
class AftertouchMessage(MIDIMessage):
    """MIDI Channel Aftertouch message.

    Pressure applied after key press (0-127).

    Attributes:
        pressure: Pressure value (0-127)
        timestamp: Time in seconds
        channel: MIDI channel (0-15)
    """

    pressure: int = 0

    def __post_init__(self):
        """Validate pressure range."""
        super().__post_init__()

        if not 0 <= self.pressure <= 127:
            raise ValueError(f"Pressure must be 0-127, got {self.pressure}")

    def normalize_pressure(self) -> float:
        """Get pressure as normalized float 0.0-1.0.

        Returns:
            Pressure scaled to 0.0-1.0
        """
        return self.pressure / 127.0

