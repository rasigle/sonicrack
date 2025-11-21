"""Tests for MIDI file reader."""

import pytest

from src.engine.io.midi import MIDIFile, ControlChangeMessage
from src.engine.io.midi.input import MIDO_AVAILABLE

# Skip all tests if mido not available
pytestmark = pytest.mark.skipif(not MIDO_AVAILABLE, reason="mido not installed")


@pytest.fixture
def sample_midi_file(tmp_path):
    """Create a simple MIDI file for testing."""
    if not MIDO_AVAILABLE:
        return None

    import mido

    # Create a simple MIDI file
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Add some notes (C major scale)
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
    ticks_per_note = 480  # Quarter note at default tempo

    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=100, time=0))
        track.append(
            mido.Message("note_off", note=note, velocity=0, time=ticks_per_note)
        )

    # Save to temp file
    filepath = tmp_path / "test.mid"
    mid.save(str(filepath))

    return filepath


class TestMIDIFile:
    """Test MIDI file reader."""

    def test_load_file(self, sample_midi_file):
        """Test loading a MIDI file."""
        midi = MIDIFile(sample_midi_file)

        assert midi.filepath == sample_midi_file
        assert len(midi.messages) > 0
        assert midi.duration > 0

    def test_nonexistent_file_raises(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            MIDIFile("nonexistent.mid")

    def test_get_duration(self, sample_midi_file):
        """Test getting file duration."""
        midi = MIDIFile(sample_midi_file)
        duration = midi.get_duration()

        assert duration > 0
        assert duration < 10  # Should be short test file

    def test_get_tempo(self, sample_midi_file):
        """Test getting tempo."""
        midi = MIDIFile(sample_midi_file)
        tempo = midi.get_tempo()

        # Default tempo is 120 BPM
        assert 100 <= tempo <= 140

    def test_get_notes_in_range(self, sample_midi_file):
        """Test getting notes in time range."""
        midi = MIDIFile(sample_midi_file)

        # Get all messages
        all_messages = midi.get_notes_in_range(0, midi.duration)
        assert len(all_messages) == len(midi.messages)

        # Get first half
        half_duration = midi.duration / 2
        first_half = midi.get_notes_in_range(0, half_duration)
        assert len(first_half) <= len(all_messages)

    def test_get_note_count(self, sample_midi_file):
        """Test counting notes."""
        midi = MIDIFile(sample_midi_file)
        note_count = midi.get_note_count()

        # Should have note on + note off for each note
        assert note_count > 0
        assert note_count % 2 == 0  # Equal note on/off

    def test_get_track_count(self, sample_midi_file):
        """Test getting track count."""
        midi = MIDIFile(sample_midi_file)
        track_count = midi.get_track_count()

        assert track_count >= 1

    def test_get_message_types(self, sample_midi_file):
        """Test getting message type counts."""
        midi = MIDIFile(sample_midi_file)
        types = midi.get_message_types()

        assert isinstance(types, dict)
        assert "NoteOnMessage" in types or "NoteOffMessage" in types

    def test_get_used_channels(self, sample_midi_file):
        """Test getting used channels."""
        midi = MIDIFile(sample_midi_file)
        channels = midi.get_used_channels()

        assert isinstance(channels, list)
        assert len(channels) > 0
        assert all(0 <= ch <= 15 for ch in channels)

    def test_get_note_range(self, sample_midi_file):
        """Test getting note range."""
        midi = MIDIFile(sample_midi_file)
        low, high = midi.get_note_range()

        # C major scale from C4 (60) to C5 (72)
        assert 0 <= low <= 127
        assert 0 <= high <= 127
        assert low <= high

    def test_channel_filter(self, sample_midi_file):
        """Test filtering messages by channel."""
        midi = MIDIFile(sample_midi_file)

        # Get messages for channel 0
        ch0_messages = midi.get_notes_in_range(0, midi.duration, channel=0)

        # All should be on channel 0
        assert all(msg.channel == 0 for msg in ch0_messages)

    def test_repr(self, sample_midi_file):
        """Test string representation."""
        midi = MIDIFile(sample_midi_file)
        repr_str = repr(midi)

        assert "MIDIFile" in repr_str
        assert "test.mid" in repr_str

    def test_len(self, sample_midi_file):
        """Test len() returns message count."""
        midi = MIDIFile(sample_midi_file)

        assert len(midi) == len(midi.messages)
        assert len(midi) > 0


class TestMIDIFileWithComplexFile:
    """Test with more complex MIDI file."""

    @pytest.fixture
    def complex_midi_file(self, tmp_path):
        """Create a more complex MIDI file."""
        if not MIDO_AVAILABLE:
            return None

        import mido

        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Add tempo change
        track.append(mido.MetaMessage("set_tempo", tempo=500000))  # 120 BPM

        # Add notes on different channels (all at time 0)
        for channel in range(2):
            for note in [60, 64, 67]:  # C major chord
                track.append(
                    mido.Message(
                        "note_on",
                        note=note,
                        velocity=80 + channel * 10,
                        channel=channel,
                        time=0,
                    )
                )

        # Add CC message after 240 ticks (delta time from last message)
        track.append(
            mido.Message(
                "control_change", control=7, value=100, channel=0, time=240  # Volume
            )
        )

        # Add note offs after another 240 ticks (total 480 from note_on)
        for channel in range(2):
            for i, note in enumerate([60, 64, 67]):
                # First note_off has delta time 240, rest have 0
                delta_time = 240 if channel == 0 and i == 0 else 0
                track.append(
                    mido.Message(
                        "note_off",
                        note=note,
                        velocity=0,
                        channel=channel,
                        time=delta_time,
                    )
                )

        filepath = tmp_path / "complex.mid"
        mid.save(str(filepath))

        return filepath

    def test_multiple_channels(self, complex_midi_file):
        """Test file with multiple channels."""
        midi = MIDIFile(complex_midi_file)

        channels = midi.get_used_channels()
        assert len(channels) >= 2
        assert 0 in channels
        assert 1 in channels

    def test_get_channel_messages(self, complex_midi_file):
        """Test getting messages for specific channel."""
        midi = MIDIFile(complex_midi_file)

        ch0_messages = midi.get_channel_messages(0)
        ch1_messages = midi.get_channel_messages(1)

        assert len(ch0_messages) > 0
        assert len(ch1_messages) > 0
        assert all(msg.channel == 0 for msg in ch0_messages)
        assert all(msg.channel == 1 for msg in ch1_messages)

    def test_control_change_messages(self, complex_midi_file):
        """Test that CC messages are parsed."""
        midi = MIDIFile(complex_midi_file)

        # Debug: print all message types
        for msg in midi.messages:
            print(f"Message type: {type(msg).__name__}, {msg}")

        cc_messages = [
            msg for msg in midi.messages if isinstance(msg, ControlChangeMessage)
        ]

        assert (
            len(cc_messages) > 0
        ), f"Expected CC messages but got: {[type(m).__name__ for m in midi.messages]}"
        assert cc_messages[0].controller == 7  # Volume


class TestMIDIFileErrors:
    """Test error handling."""

    def test_invalid_file_raises(self, tmp_path):
        """Test that invalid MIDI file raises error."""
        # Create a text file, not a MIDI file
        invalid_file = tmp_path / "invalid.mid"
        invalid_file.write_text("This is not a MIDI file")

        with pytest.raises(IOError):
            MIDIFile(invalid_file)

    def test_mido_not_available(self, monkeypatch, tmp_path):
        """Test error when mido not available."""
        # Create a dummy file
        dummy_file = tmp_path / "dummy.mid"
        dummy_file.write_bytes(b"MThd" + b"\x00" * 20)  # Fake MIDI header

        # Patch MIDO_AVAILABLE before importing
        import src.engine.io.midi.file_reader

        # Need to patch at the module level where it's checked
        monkeypatch.setattr(src.engine.io.midi.file_reader, "MIDO_AVAILABLE", False)

        with pytest.raises(RuntimeError, match="mido.*not installed"):
            MIDIFile(str(dummy_file))
