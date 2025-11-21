"""Tests for MIDI utility functions."""

import pytest
from engine.io.midi import (
    midi_to_frequency,
    frequency_to_midi,
    note_name_to_midi,
    midi_to_note_name,
    note_name_to_frequency,
    get_note_range,
    transpose,
)


class TestMIDIToFrequency:
    """Test MIDI note to frequency conversion."""

    def test_a4_is_440hz(self):
        """A4 (note 69) should be 440 Hz."""
        assert midi_to_frequency(69) == 440.0

    def test_c4_middle_c(self):
        """C4 (Middle C, note 60) frequency."""
        freq = midi_to_frequency(60)
        assert abs(freq - 261.626) < 0.01

    def test_octave_doubles_frequency(self):
        """One octave up should double the frequency."""
        c4 = midi_to_frequency(60)
        c5 = midi_to_frequency(72)
        assert abs(c5 / c4 - 2.0) < 0.001

    def test_alternate_tuning(self):
        """Test with non-standard A4 tuning."""
        # A4 = 432 Hz (alternative tuning)
        assert midi_to_frequency(69, a4_tuning=432.0) == 432.0

    def test_invalid_note_raises(self):
        """Invalid MIDI note should raise ValueError."""
        with pytest.raises(ValueError):
            midi_to_frequency(-1)
        with pytest.raises(ValueError):
            midi_to_frequency(128)


class TestFrequencyToMIDI:
    """Test frequency to MIDI note conversion."""

    def test_440hz_is_a4(self):
        """440 Hz should be A4 (note 69)."""
        assert frequency_to_midi(440.0) == 69

    def test_middle_c(self):
        """261.63 Hz should be C4 (note 60)."""
        assert frequency_to_midi(261.63) == 60

    def test_roundtrip(self):
        """Converting MIDI->freq->MIDI should return original."""
        for note in [21, 40, 60, 80, 100]:
            freq = midi_to_frequency(note)
            back = frequency_to_midi(freq)
            assert back == note

    def test_invalid_frequency_raises(self):
        """Invalid frequency should raise ValueError."""
        with pytest.raises(ValueError):
            frequency_to_midi(0)
        with pytest.raises(ValueError):
            frequency_to_midi(-100)


class TestNoteNameToMIDI:
    """Test note name parsing."""

    def test_middle_c(self):
        """C4 should be note 60."""
        assert note_name_to_midi("C4") == 60

    def test_concert_a(self):
        """A4 should be note 69."""
        assert note_name_to_midi("A4") == 69

    def test_sharps(self):
        """Sharp notes should parse correctly."""
        assert note_name_to_midi("C#4") == 61
        assert note_name_to_midi("F#5") == 78

    def test_flats(self):
        """Flat notes should parse correctly."""
        assert note_name_to_midi("Db4") == 61
        assert note_name_to_midi("Bb3") == 58

    def test_enharmonic_equivalents(self):
        """C# and Db should be the same MIDI note."""
        assert note_name_to_midi("C#4") == note_name_to_midi("Db4")
        assert note_name_to_midi("F#5") == note_name_to_midi("Gb5")

    def test_low_notes(self):
        """Test very low notes."""
        assert note_name_to_midi("C0") == 12
        assert note_name_to_midi("C-1") == 0

    def test_high_notes(self):
        """Test very high notes."""
        assert note_name_to_midi("G9") == 127

    def test_invalid_format_raises(self):
        """Invalid note name format should raise."""
        with pytest.raises(ValueError):
            note_name_to_midi("H4")  # H is not a note
        with pytest.raises(ValueError):
            note_name_to_midi("C")  # Missing octave
        with pytest.raises(ValueError):
            note_name_to_midi("4C")  # Wrong order

    def test_out_of_range_raises(self):
        """Out of range notes should raise."""
        with pytest.raises(ValueError):
            note_name_to_midi("C10")  # Too high
        with pytest.raises(ValueError):
            note_name_to_midi("C-2")  # Too low


class TestMIDIToNoteName:
    """Test MIDI to note name conversion."""

    def test_middle_c(self):
        """Note 60 should be C4."""
        assert midi_to_note_name(60) == "C4"

    def test_concert_a(self):
        """Note 69 should be A4."""
        assert midi_to_note_name(69) == "A4"

    def test_sharps(self):
        """Black keys should use sharps by default."""
        assert midi_to_note_name(61) == "C#4"
        assert midi_to_note_name(78) == "F#5"

    def test_flats(self):
        """Black keys should use flats when requested."""
        assert midi_to_note_name(61, use_sharps=False) == "Db4"
        assert midi_to_note_name(78, use_sharps=False) == "Gb5"

    def test_roundtrip(self):
        """Converting name->MIDI->name should return original."""
        notes = ["C4", "A#5", "Bb3", "F#2", "G7"]
        for note in notes:
            midi = note_name_to_midi(note)
            # May not be exact due to enharmonic equivalents
            back_midi = note_name_to_midi(midi_to_note_name(midi))
            assert midi == back_midi


class TestNoteNameToFrequency:
    """Test direct note name to frequency conversion."""

    def test_a4_440hz(self):
        """A4 should be 440 Hz."""
        assert note_name_to_frequency("A4") == 440.0

    def test_middle_c(self):
        """C4 frequency."""
        freq = note_name_to_frequency("C4")
        assert abs(freq - 261.626) < 0.01


class TestGetNoteRange:
    """Test note range generation."""

    def test_one_octave(self):
        """One octave should have 13 notes (inclusive)."""
        notes = get_note_range("C4", "C5")
        assert len(notes) == 13
        assert notes[0] == 60  # C4
        assert notes[-1] == 72  # C5

    def test_chromatic_scale(self):
        """C to B should have 12 notes."""
        notes = get_note_range("C4", "B4")
        assert len(notes) == 12

    def test_single_note(self):
        """Range of one note."""
        notes = get_note_range("A4", "A4")
        assert notes == [69]

    def test_invalid_range_raises(self):
        """End before start should raise."""
        with pytest.raises(ValueError):
            get_note_range("C5", "C4")


class TestTranspose:
    """Test note transposition."""

    def test_octave_up(self):
        """Transpose up one octave."""
        assert transpose(60, 12) == 72  # C4 -> C5

    def test_octave_down(self):
        """Transpose down one octave."""
        assert transpose(60, -12) == 48  # C4 -> C3

    def test_fifth_up(self):
        """Transpose up a perfect fifth (7 semitones)."""
        assert transpose(60, 7) == 67  # C4 -> G4

    def test_clamping_high(self):
        """Transposing too high should clamp to 127."""
        assert transpose(120, 20) == 127

    def test_clamping_low(self):
        """Transposing too low should clamp to 0."""
        assert transpose(10, -20) == 0

    def test_invalid_note_raises(self):
        """Invalid input note should raise."""
        with pytest.raises(ValueError):
            transpose(-1, 5)
        with pytest.raises(ValueError):
            transpose(128, 5)
