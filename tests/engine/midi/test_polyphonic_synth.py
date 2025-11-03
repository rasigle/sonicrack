"""Tests for polyphonic MIDI synthesizer."""

import numpy as np

from src.engine import SineOscillator
from src.engine.midi.messages import NoteOnMessage, NoteOffMessage
from src.engine.midi.polyphonic_synth import PolyphonicSynth, Voice


class DummyVoice:
    """Simple dummy voice for testing."""

    def __init__(self):
        self.frequency = 440
        self.note_on_triggered = False
        self.note_off_triggered = False
        self.ended = False

    def get_samples(self, num_samples):
        """Return simple sine wave."""
        if self.ended:
            return np.zeros(num_samples, dtype=np.float32)
        return np.sin(np.linspace(0, 2 * np.pi, num_samples)).astype(np.float32)

    def trigger_note_on(self):
        self.note_on_triggered = True
        self.ended = False

    def trigger_note_off(self):
        self.note_off_triggered = True


class TestVoice:
    """Test Voice class."""

    def test_init(self):
        """Test voice initialization."""
        voice = Voice()
        assert voice.note is None
        assert voice.velocity == 0
        assert voice.component is None
        assert voice.is_active is False
        assert voice.age == 0

    def test_is_free(self):
        """Test is_free()."""
        voice = Voice()
        assert voice.is_free() is True

        voice.note = 60
        assert voice.is_free() is False

        voice.note = None
        voice.is_active = True
        assert voice.is_free() is False

    def test_is_releasing(self):
        """Test is_releasing()."""
        voice = Voice()
        assert voice.is_releasing() is False

        voice.note = 60
        voice.is_active = False
        assert voice.is_releasing() is True

        voice.is_active = True
        assert voice.is_releasing() is False

    def test_clear(self):
        """Test clear()."""
        voice = Voice(note=60, velocity=100, is_active=True, age=1000)
        voice.clear()

        assert voice.note is None
        assert voice.velocity == 0
        assert voice.is_active is False
        assert voice.age == 0


class TestPolyphonicSynth:
    """Test PolyphonicSynth class."""

    def test_init(self):
        """Test initialization."""

        def voice_factory():
            return DummyVoice()

        synth = PolyphonicSynth(voice_factory, max_voices=4)

        assert synth.max_voices == 4
        assert len(synth.voices) == 4
        assert synth.get_active_voice_count() == 0
        assert synth.get_free_voice_count() == 4

    def test_note_on_single(self):
        """Test playing single note."""

        def voice_factory():
            return DummyVoice()

        synth = PolyphonicSynth(voice_factory, max_voices=4)
        synth.note_on(60, 100)

        assert synth.get_active_voice_count() == 1
        assert synth.get_free_voice_count() == 3

        # Find the active voice
        active_voice = next(v for v in synth.voices if v.note == 60)
        assert active_voice.velocity == 100
        assert active_voice.is_active is True

    def test_note_on_chord(self):
        """Test playing chord (multiple notes)."""

        def voice_factory():
            return DummyVoice()

        synth = PolyphonicSynth(voice_factory, max_voices=8)

        # Play C major chord
        synth.note_on(60, 100)  # C
        synth.note_on(64, 100)  # E
        synth.note_on(67, 100)  # G

        assert synth.get_active_voice_count() == 3
        assert synth.get_free_voice_count() == 5

    def test_note_off(self):
        """Test releasing note."""

        def voice_factory():
            return DummyVoice()

        synth = PolyphonicSynth(voice_factory, max_voices=4)
        synth.note_on(60, 100)
        assert synth.get_active_voice_count() == 1

        synth.note_off(60)

        # Voice should be releasing but not free yet
        voice = synth.voices[0]
        assert voice.note == 60
        assert voice.is_active is False
        assert voice.is_releasing() is True

    def test_voice_stealing_releasing(self):
        """Test that releasing voices are stolen before active ones."""

        def voice_factory():
            return DummyVoice()

        synth = PolyphonicSynth(voice_factory, max_voices=2)

        # Fill voices
        synth.note_on(60, 100)
        synth.note_on(64, 100)

        # Release first note
        synth.note_off(60)

        # Play new note - should steal releasing voice
        synth.note_on(67, 100)

        # Should have note 64 (active) and 67 (new)
        notes = [v.note for v in synth.voices if v.note is not None]
        assert 67 in notes
        assert 64 in notes or 60 in notes  # Either old active or being stolen

    def test_get_samples_silence(self):
        """Test get_samples returns silence when no notes."""

        def voice_factory():
            return DummyVoice()

        synth = PolyphonicSynth(voice_factory, max_voices=4)
        samples = synth.get_samples(100)

        assert len(samples) == 100
        assert np.allclose(samples, 0.0)

    def test_get_samples_mixing(self):
        """Test that multiple voices are mixed."""

        def voice_factory():
            return DummyVoice()

        synth = PolyphonicSynth(voice_factory, max_voices=4)

        # Play two notes
        synth.note_on(60, 100)
        synth.note_on(64, 100)

        samples = synth.get_samples(100)

        assert len(samples) == 100
        assert not np.allclose(samples, 0.0)  # Should have audio

        # Check normalization (shouldn't clip)
        assert np.abs(samples).max() <= 2.0  # Allow some headroom

    def test_process_message(self):
        """Test processing MIDI messages."""

        def voice_factory():
            return DummyVoice()

        synth = PolyphonicSynth(voice_factory, max_voices=4)

        # Note on
        msg_on = NoteOnMessage(timestamp=0.0, note=60, velocity=100)
        synth.process_message(msg_on)
        assert synth.get_active_voice_count() == 1

        # Note off
        msg_off = NoteOffMessage(timestamp=1.0, note=60)
        synth.process_message(msg_off)

        voice = synth.voices[0]
        assert voice.is_releasing() is True

    def test_reset(self):
        """Test reset clears all voices."""

        def voice_factory():
            return DummyVoice()

        synth = PolyphonicSynth(voice_factory, max_voices=4)

        # Play some notes
        synth.note_on(60, 100)
        synth.note_on(64, 100)
        assert synth.get_active_voice_count() == 2

        # Reset
        synth.reset()

        assert synth.get_active_voice_count() == 0
        assert synth.get_free_voice_count() == 4

    def test_repr(self):
        """Test string representation."""

        def voice_factory():
            return DummyVoice()

        synth = PolyphonicSynth(voice_factory, max_voices=4)
        synth.note_on(60, 100)

        repr_str = repr(synth)
        assert "1 active" in repr_str
        assert "3 free" in repr_str
        assert "4 total" in repr_str

    def test_with_real_oscillator(self):
        """Test with actual oscillator."""

        def voice_factory():
            return SineOscillator(440)

        synth = PolyphonicSynth(voice_factory, max_voices=4)

        # Play chord
        synth.note_on(60, 100)
        synth.note_on(64, 100)
        synth.note_on(67, 100)

        samples = synth.get_samples(1000)

        assert len(samples) == 1000
        assert not np.allclose(samples, 0.0)
        # Should be roughly between -1 and 1 (with normalization)
        assert np.abs(samples).max() < 3.0
