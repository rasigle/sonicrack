"""Test MIDI to CV converter component."""

import pytest
import numpy as np

from src.engine.io.midi import (
    MIDIToCV,
    NoteOnMessage,
    NoteOffMessage,
    ControlChangeMessage,
    PitchBendMessage,
)
from src.engine.io.midi import midi_to_frequency


class TestMIDIToCV:
    """Test MIDI to CV conversion."""

    def test_initialization(self):
        """Test CV converter initializes with correct defaults."""
        cv = MIDIToCV()

        assert cv.gate == 0.0
        assert cv.frequency == 440.0  # A4 default
        assert cv.velocity == 0.0
        assert cv.mod_wheel == 0.0
        assert cv.expression == 1.0
        assert cv.pitch_bend == 0.0
        assert cv.current_note is None

    def test_note_on(self):
        """Test note on message updates CV state."""
        cv = MIDIToCV()
        msg = NoteOnMessage(timestamp=0.0, channel=0, note=60, velocity=100)

        cv.process_message(msg)

        assert cv.gate == 1.0
        assert cv.current_note == 60
        assert cv.velocity == pytest.approx(100 / 127, rel=0.01)
        assert cv.frequency == pytest.approx(midi_to_frequency(60), rel=0.01)

    def test_note_off(self):
        """Test note off message clears gate."""
        cv = MIDIToCV()

        # Note on
        cv.process_message(NoteOnMessage(0.0, 0, 60, 100))
        assert cv.gate == 1.0

        # Note off
        cv.process_message(NoteOffMessage(0.1, 0, 60))
        assert cv.gate == 0.0
        assert cv.current_note == 60  # Note number persists

    def test_note_on_velocity_zero_is_note_off(self):
        """Test note on with velocity 0 is treated as note off."""
        cv = MIDIToCV()

        # Note on
        cv.process_message(NoteOnMessage(0.0, 0, 60, 100))
        assert cv.gate == 1.0

        # Note on with velocity 0 (should be note off)
        cv.process_message(NoteOnMessage(0.1, 0, 60, 0))
        assert cv.gate == 0.0

    def test_mod_wheel(self):
        """Test mod wheel CC updates state."""
        cv = MIDIToCV()
        msg = ControlChangeMessage(0.0, 0, controller=1, value=64)

        cv.process_message(msg)

        assert cv.mod_wheel == pytest.approx(64 / 127, rel=0.01)

    def test_expression(self):
        """Test expression CC updates state."""
        cv = MIDIToCV()
        msg = ControlChangeMessage(0.0, 0, controller=11, value=100)

        cv.process_message(msg)

        assert cv.expression == pytest.approx(100 / 127, rel=0.01)

    def test_pitch_bend(self):
        """Test pitch bend updates frequency."""
        cv = MIDIToCV()

        # Note on middle C
        cv.process_message(NoteOnMessage(0.0, 0, 60, 100))

        # Pitch bend up 1 semitone (8192 / 2 with ±2 semitone range)
        msg = PitchBendMessage(0.1, 0, value=4096)
        cv.process_message(msg)

        # Should be +1 semitone higher
        expected_freq = midi_to_frequency(61)
        assert cv.frequency == pytest.approx(expected_freq, rel=0.01)

    def test_pitch_bend_range(self):
        """Test custom pitch bend range."""
        cv = MIDIToCV(pitch_bend_range=12.0)  # Full octave

        cv.process_message(NoteOnMessage(0.0, 0, 60, 100))

        # Max pitch bend up (should be +12 semitones = 1 octave)
        msg = PitchBendMessage(0.1, 0, value=8191)
        cv.process_message(msg)

        expected_freq = midi_to_frequency(72)  # C5
        assert cv.frequency == pytest.approx(expected_freq, rel=0.01)

    def test_get_samples(self):
        """Test generating CV samples."""
        cv = MIDIToCV()
        cv.process_message(NoteOnMessage(0.0, 0, 69, 100))  # A4

        samples = cv.get_samples(100)

        assert len(samples) == 100
        assert np.all(samples == 440.0)  # Constant frequency

    def test_get_gate_samples(self):
        """Test generating gate samples."""
        cv = MIDIToCV()
        cv.process_message(NoteOnMessage(0.0, 0, 60, 100))

        gate_samples = cv.get_gate_samples(100)

        assert len(gate_samples) == 100
        assert np.all(gate_samples == 1.0)

    def test_get_velocity_samples(self):
        """Test generating velocity samples."""
        cv = MIDIToCV()
        cv.process_message(NoteOnMessage(0.0, 0, 60, 64))

        vel_samples = cv.get_velocity_samples(100)

        assert len(vel_samples) == 100
        expected_vel = 64 / 127
        assert np.all(vel_samples == pytest.approx(expected_vel, rel=0.01))

    def test_reset(self):
        """Test reset clears all state."""
        cv = MIDIToCV()

        # Set some state
        cv.process_message(NoteOnMessage(0.0, 0, 60, 100))
        cv.process_message(ControlChangeMessage(0.1, 0, 1, 64))

        assert cv.gate == 1.0
        assert cv.current_note == 60

        # Reset
        cv.reset()

        assert cv.gate == 0.0
        assert cv.frequency == 440.0
        assert cv.velocity == 0.0
        assert cv.mod_wheel == 0.0
        assert cv.expression == 1.0
        assert cv.pitch_bend == 0.0
        assert cv.current_note is None

    def test_multiple_notes_last_priority(self):
        """Test that last note takes priority (monophonic behavior)."""
        cv = MIDIToCV()

        # Play C
        cv.process_message(NoteOnMessage(0.0, 0, 60, 100))
        assert cv.frequency == pytest.approx(midi_to_frequency(60), rel=0.01)

        # Play E (should replace C)
        cv.process_message(NoteOnMessage(0.1, 0, 64, 100))
        assert cv.frequency == pytest.approx(midi_to_frequency(64), rel=0.01)

        # Release E
        cv.process_message(NoteOffMessage(0.2, 0, 64))
        assert cv.gate == 0.0  # Gate turns off

        # C is still the "current note" but gate is off
        # This is monophonic behavior - no note stacking

    def test_different_channels(self):
        """Test that CV converter responds to all channels (omni mode)."""
        cv = MIDIToCV()

        # Note on channel 0
        cv.process_message(NoteOnMessage(0.0, 0, 60, 100))
        assert cv.gate == 1.0

        # Note off on channel 0
        cv.process_message(NoteOffMessage(0.1, 0, 60))
        assert cv.gate == 0.0

        # Note on channel 5
        cv.process_message(NoteOnMessage(0.2, 5, 64, 100))
        assert cv.gate == 1.0
        assert cv.frequency == pytest.approx(midi_to_frequency(64), rel=0.01)
