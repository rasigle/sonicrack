"""Tests for monophonic MIDI synthesizer."""

import numpy as np

from src.engine import SineOscillator
from src.engine.io.midi import MonophonicSynth, NoteOffMessage, NoteOnMessage


class DummyVoice:
    """Simple dummy voice for testing."""

    def __init__(self):
        self.frequency = 440
        self.note_on_triggered = False
        self.note_off_triggered = False

    def get_samples(self, num_samples):
        """Return simple sine wave."""
        return np.sin(np.linspace(0, 2 * np.pi, num_samples))

    def trigger_note_on(self):
        self.note_on_triggered = True

    def trigger_note_off(self):
        self.note_off_triggered = True


class TestMonophonicSynth:
    """Test MonophonicSynth class."""

    def test_init(self):
        """Test initialization."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)

        assert synth.voice is None
        assert synth.current_note is None
        assert synth.current_velocity == 0
        assert synth.is_playing is False

    def test_note_on(self):
        """Test note on."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)
        synth.note_on(60, 100)

        assert synth.current_note == 60
        assert synth.current_velocity == 100
        assert synth.is_playing is True
        assert synth.voice is not None

    def test_note_off(self):
        """Test note off."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)
        synth.note_on(60, 100)
        synth.note_off(60)

        assert synth.is_playing is False

    def test_note_off_wrong_note(self):
        """Test note off for different note doesn't stop current note."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)
        synth.note_on(60, 100)
        synth.note_off(62)  # Different note

        assert synth.is_playing is True  # Still playing
        assert synth.current_note == 60

    def test_velocity_zero_is_note_off(self):
        """Test that velocity 0 triggers note off."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)
        synth.note_on(60, 100)
        assert synth.is_playing is True

        synth.note_on(60, 0)  # Velocity 0 = note off
        assert synth.is_playing is False

    def test_monophonic_behavior(self):
        """Test that new note replaces old note."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)

        # Play first note
        synth.note_on(60, 100)
        assert synth.current_note == 60

        # Play second note (should replace first)
        synth.note_on(64, 100)
        assert synth.current_note == 64
        assert synth.is_playing is True

    def test_get_samples_no_voice(self):
        """Test get_samples returns silence when no voice."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)
        samples = synth.get_samples(100)

        assert len(samples) == 100
        assert np.allclose(samples, 0.0)

    def test_get_samples_with_voice(self):
        """Test get_samples returns audio when voice active."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)
        synth.note_on(60, 100)

        samples = synth.get_samples(100)

        assert len(samples) == 100
        assert not np.allclose(samples, 0.0)  # Should have audio

    def test_process_message_note_on(self):
        """Test processing NoteOnMessage."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)
        msg = NoteOnMessage(timestamp=0.0, note=60, velocity=100)

        synth.process_message(msg)

        assert synth.current_note == 60
        assert synth.is_playing is True

    def test_process_message_note_off(self):
        """Test processing NoteOffMessage."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)

        # Note on
        synth.note_on(60, 100)

        # Note off via message
        msg = NoteOffMessage(timestamp=1.0, note=60)
        synth.process_message(msg)

        assert synth.is_playing is False

    def test_reset(self):
        """Test reset clears state."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)
        synth.note_on(60, 100)

        synth.reset()

        assert synth.voice is None
        assert synth.current_note is None
        assert synth.current_velocity == 0
        assert synth.is_playing is False

    def test_invalid_note_number(self):
        """Test invalid note numbers are ignored."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)

        # Should not crash
        synth.note_on(-1, 100)
        assert synth.voice is None

        synth.note_on(128, 100)
        assert synth.voice is None

    def test_invalid_velocity(self):
        """Test invalid velocities are ignored."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)

        # Should not crash
        synth.note_on(60, -1)
        assert synth.voice is None

        synth.note_on(60, 128)
        assert synth.voice is None

    def test_repr(self):
        """Test string representation."""

        def voice_factory():
            return DummyVoice()

        synth = MonophonicSynth(voice_factory)

        # Idle state
        repr_str = repr(synth)
        assert "idle" in repr_str.lower()

        # Playing state
        synth.note_on(60, 100)
        repr_str = repr(synth)
        assert "60" in repr_str
        assert "100" in repr_str

    def test_with_real_oscillator(self):
        """Test with actual oscillator."""

        def voice_factory():
            return SineOscillator(440)

        synth = MonophonicSynth(voice_factory)
        synth.note_on(60, 100)

        samples = synth.get_samples(1000)

        assert len(samples) == 1000
        assert not np.allclose(samples, 0.0)
        # Should be roughly between -1 and 1 (with some headroom)
        assert np.abs(samples).max() < 2.0
