"""Comprehensive tests for MIDI messages module."""

import pytest
from src.engine.midi.messages import (
    MIDIMessage,
    NoteOnMessage,
    NoteOffMessage,
    ControlChangeMessage,
    PitchBendMessage,
    ProgramChangeMessage,
    AftertouchMessage,
)


class TestMIDIMessage:
    """Test base MIDIMessage class."""

    def test_init(self):
        """Test basic initialization."""
        msg = MIDIMessage(timestamp=1.5, channel=0)
        assert msg.timestamp == 1.5
        assert msg.channel == 0

    def test_default_channel(self):
        """Test default channel is 0."""
        msg = MIDIMessage(timestamp=0.0)
        assert msg.channel == 0

    def test_channel_validation(self):
        """Test channel range validation."""
        # Valid channels
        MIDIMessage(timestamp=0.0, channel=0)
        MIDIMessage(timestamp=0.0, channel=15)

        # Invalid channels
        with pytest.raises(ValueError, match="MIDI channel must be 0-15"):
            MIDIMessage(timestamp=0.0, channel=-1)

        with pytest.raises(ValueError, match="MIDI channel must be 0-15"):
            MIDIMessage(timestamp=0.0, channel=16)


class TestNoteOnMessage:
    """Test NoteOnMessage class."""

    def test_init(self):
        """Test initialization."""
        msg = NoteOnMessage(timestamp=0.0, note=60, velocity=100, channel=0)
        assert msg.timestamp == 0.0
        assert msg.note == 60
        assert msg.velocity == 100
        assert msg.channel == 0

    def test_defaults(self):
        """Test default values."""
        msg = NoteOnMessage(timestamp=0.0)
        assert msg.note == 60  # Middle C
        assert msg.velocity == 100
        assert msg.channel == 0

    def test_note_validation(self):
        """Test note number validation."""
        # Valid notes
        NoteOnMessage(timestamp=0.0, note=0)
        NoteOnMessage(timestamp=0.0, note=127)

        # Invalid notes
        with pytest.raises(ValueError, match="MIDI note must be 0-127"):
            NoteOnMessage(timestamp=0.0, note=-1)

        with pytest.raises(ValueError, match="MIDI note must be 0-127"):
            NoteOnMessage(timestamp=0.0, note=128)

    def test_velocity_validation(self):
        """Test velocity validation."""
        # Valid velocities
        NoteOnMessage(timestamp=0.0, velocity=0)
        NoteOnMessage(timestamp=0.0, velocity=127)

        # Invalid velocities
        with pytest.raises(ValueError, match="MIDI velocity must be 0-127"):
            NoteOnMessage(timestamp=0.0, velocity=-1)

        with pytest.raises(ValueError, match="MIDI velocity must be 0-127"):
            NoteOnMessage(timestamp=0.0, velocity=128)

    def test_to_frequency(self):
        """Test note to frequency conversion."""
        msg = NoteOnMessage(timestamp=0.0, note=69)  # A4
        assert abs(msg.to_frequency() - 440.0) < 0.01

        msg = NoteOnMessage(timestamp=0.0, note=60)  # Middle C
        assert abs(msg.to_frequency() - 261.63) < 0.01

        msg = NoteOnMessage(timestamp=0.0, note=57)  # A3
        assert abs(msg.to_frequency() - 220.0) < 0.01

    def test_to_frequency_alternate_tuning(self):
        """Test frequency conversion with alternate tuning."""
        msg = NoteOnMessage(timestamp=0.0, note=69)  # A4
        freq = msg.to_frequency(a4_tuning=432.0)
        assert abs(freq - 432.0) < 0.01

    def test_normalize_velocity(self):
        """Test velocity normalization."""
        msg = NoteOnMessage(timestamp=0.0, velocity=0)
        assert msg.normalize_velocity() == 0.0

        msg = NoteOnMessage(timestamp=0.0, velocity=127)
        assert msg.normalize_velocity() == 1.0

        msg = NoteOnMessage(timestamp=0.0, velocity=64)
        assert abs(msg.normalize_velocity() - 0.504) < 0.01


class TestNoteOffMessage:
    """Test NoteOffMessage class."""

    def test_init(self):
        """Test initialization."""
        msg = NoteOffMessage(timestamp=1.0, note=60, velocity=64, channel=1)
        assert msg.timestamp == 1.0
        assert msg.note == 60
        assert msg.velocity == 64
        assert msg.channel == 1

    def test_defaults(self):
        """Test default values."""
        msg = NoteOffMessage(timestamp=0.0)
        assert msg.note == 60
        assert msg.velocity == 64
        assert msg.channel == 0

    def test_note_validation(self):
        """Test note validation."""
        NoteOffMessage(timestamp=0.0, note=0)
        NoteOffMessage(timestamp=0.0, note=127)

        with pytest.raises(ValueError, match="MIDI note must be 0-127"):
            NoteOffMessage(timestamp=0.0, note=128)


class TestControlChangeMessage:
    """Test ControlChangeMessage class."""

    def test_init(self):
        """Test initialization."""
        msg = ControlChangeMessage(
            timestamp=0.0, controller=7, value=100, channel=0
        )
        assert msg.timestamp == 0.0
        assert msg.controller == 7
        assert msg.value == 100
        assert msg.channel == 0

    def test_defaults(self):
        """Test default values."""
        msg = ControlChangeMessage(timestamp=0.0)
        assert msg.controller == 1  # Default modulation wheel
        assert msg.value == 0
        assert msg.channel == 0

    def test_controller_validation(self):
        """Test controller number validation."""
        ControlChangeMessage(timestamp=0.0, controller=0)
        ControlChangeMessage(timestamp=0.0, controller=127)

        with pytest.raises(ValueError, match="CC controller must be 0-127"):
            ControlChangeMessage(timestamp=0.0, controller=128)

    def test_value_validation(self):
        """Test value validation."""
        ControlChangeMessage(timestamp=0.0, value=0)
        ControlChangeMessage(timestamp=0.0, value=127)

        with pytest.raises(ValueError, match="CC value must be 0-127"):
            ControlChangeMessage(timestamp=0.0, value=128)

    def test_normalize_value(self):
        """Test value normalization."""
        msg = ControlChangeMessage(timestamp=0.0, value=0)
        assert msg.normalize_value() == 0.0

        msg = ControlChangeMessage(timestamp=0.0, value=127)
        assert msg.normalize_value() == 1.0

        msg = ControlChangeMessage(timestamp=0.0, value=64)
        assert abs(msg.normalize_value() - 0.504) < 0.01

    def test_is_switch_on(self):
        """Test switch detection for CC 64 (sustain pedal)."""
        msg = ControlChangeMessage(timestamp=0.0, controller=64, value=0)
        assert msg.is_switch_on() is False

        msg = ControlChangeMessage(timestamp=0.0, controller=64, value=63)
        assert msg.is_switch_on() is False

        msg = ControlChangeMessage(timestamp=0.0, controller=64, value=64)
        assert msg.is_switch_on() is True

        msg = ControlChangeMessage(timestamp=0.0, controller=64, value=127)
        assert msg.is_switch_on() is True


class TestPitchBendMessage:
    """Test PitchBendMessage class."""

    def test_init(self):
        """Test initialization."""
        msg = PitchBendMessage(timestamp=0.0, value=2048, channel=0)
        assert msg.timestamp == 0.0
        assert msg.value == 2048
        assert msg.channel == 0

    def test_defaults(self):
        """Test default values."""
        msg = PitchBendMessage(timestamp=0.0)
        assert msg.value == 0  # Center
        assert msg.channel == 0

    def test_value_validation(self):
        """Test value range validation."""
        PitchBendMessage(timestamp=0.0, value=-8192)
        PitchBendMessage(timestamp=0.0, value=8191)

        with pytest.raises(ValueError, match="Pitch bend must be -8192 to \\+8191"):
            PitchBendMessage(timestamp=0.0, value=-8193)

        with pytest.raises(ValueError, match="Pitch bend must be -8192 to \\+8191"):
            PitchBendMessage(timestamp=0.0, value=8192)

    def test_normalize(self):
        """Test pitch bend normalization."""
        msg = PitchBendMessage(timestamp=0.0, value=-8192)
        assert msg.normalize_value() == -1.0

        msg = PitchBendMessage(timestamp=0.0, value=0)
        assert msg.normalize_value() == 0.0

        msg = PitchBendMessage(timestamp=0.0, value=8191)
        assert abs(msg.normalize_value() - 1.0) < 0.001

    def test_to_semitones(self):
        """Test conversion to semitones."""
        msg = PitchBendMessage(timestamp=0.0, value=0)
        assert msg.to_semitones(bend_range=2) == 0.0

        msg = PitchBendMessage(timestamp=0.0, value=8191)
        assert abs(msg.to_semitones(bend_range=2) - 2.0) < 0.01

        msg = PitchBendMessage(timestamp=0.0, value=-8192)
        assert abs(msg.to_semitones(bend_range=2) - (-2.0)) < 0.01

        # 12 semitone bend range
        msg = PitchBendMessage(timestamp=0.0, value=4096)
        assert abs(msg.to_semitones(bend_range=12) - 6.0) < 0.1


class TestProgramChangeMessage:
    """Test ProgramChangeMessage class."""

    def test_init(self):
        """Test initialization."""
        msg = ProgramChangeMessage(timestamp=0.0, program=5, channel=2)
        assert msg.timestamp == 0.0
        assert msg.program == 5
        assert msg.channel == 2

    def test_defaults(self):
        """Test default values."""
        msg = ProgramChangeMessage(timestamp=0.0)
        assert msg.program == 0
        assert msg.channel == 0

    def test_program_validation(self):
        """Test program number validation."""
        ProgramChangeMessage(timestamp=0.0, program=0)
        ProgramChangeMessage(timestamp=0.0, program=127)

        with pytest.raises(ValueError, match="Program must be 0-127"):
            ProgramChangeMessage(timestamp=0.0, program=128)


class TestAftertouchMessage:
    """Test AftertouchMessage class."""

    def test_init(self):
        """Test initialization."""
        msg = AftertouchMessage(timestamp=0.0, pressure=80, channel=1)
        assert msg.timestamp == 0.0
        assert msg.pressure == 80
        assert msg.channel == 1

    def test_defaults(self):
        """Test default values."""
        msg = AftertouchMessage(timestamp=0.0)
        assert msg.pressure == 0
        assert msg.channel == 0

    def test_pressure_validation(self):
        """Test pressure validation."""
        AftertouchMessage(timestamp=0.0, pressure=0)
        AftertouchMessage(timestamp=0.0, pressure=127)

        with pytest.raises(ValueError, match="Pressure must be 0-127"):
            AftertouchMessage(timestamp=0.0, pressure=128)

    def test_normalize_pressure(self):
        """Test pressure normalization."""
        msg = AftertouchMessage(timestamp=0.0, pressure=0)
        assert msg.normalize_pressure() == 0.0

        msg = AftertouchMessage(timestamp=0.0, pressure=127)
        assert msg.normalize_pressure() == 1.0

        msg = AftertouchMessage(timestamp=0.0, pressure=64)
        assert abs(msg.normalize_pressure() - 0.504) < 0.01

