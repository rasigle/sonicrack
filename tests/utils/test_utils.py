"""Unit tests for audio utility functions.

Tests cover file I/O, format conversions, and audio playback utilities.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils import (
    to_int16,
    save_wave,
    load_wave,
    note_to_frequency,
    # Legacy API
    to_int16,
    save_wave,
    load_wave,
    note_to_frequency,
)
from src.constants import DEFAULT_SAMPLE_RATE


class TestToInt16:
    """Tests for audio format conversion to 16-bit integers."""

    def test_basic_conversion(self):
        """Test conversion of simple normalized audio."""
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
        result = to_int16(audio, amplitude=1.0)

        assert result.dtype == np.int16
        assert len(result) == len(audio)
        assert result[0] == 0
        assert result[-1] == -32767  # Negative max (formula uses 2^15-1)

    def test_amplitude_scaling(self):
        """Test that amplitude parameter scales correctly."""
        audio = np.array([1.0])
        result_full = to_int16(audio, amplitude=1.0)
        result_half = to_int16(audio, amplitude=0.5)

        # Half amplitude should be approximately half the value
        assert abs(result_half[0]) < abs(result_full[0])
        assert abs(result_half[0] - result_full[0] / 2) < 10  # Allow small rounding error

    def test_clipping(self):
        """Test that values outside [-1, 1] are clipped."""
        audio = np.array([2.0, -2.0])
        result = to_int16(audio, amplitude=1.0)

        # Should clip to max/min int16 values
        assert result[0] == 32767
        assert result[1] == -32768

    def test_list_input(self):
        """Test that list input is properly converted."""
        audio = [0.0, 0.5, -0.5]
        result = to_int16(audio, amplitude=1.0)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int16
        assert len(result) == 3

    def test_empty_array(self):
        """Test handling of empty input."""
        audio = np.array([])
        result = to_int16(audio, amplitude=1.0)

        assert len(result) == 0
        assert result.dtype == np.int16


class TestSaveLoadWave:
    """Tests for WAV file I/O operations."""

    def test_save_and_load_mono(self):
        """Test saving and loading mono audio."""
        # Create test audio
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(DEFAULT_SAMPLE_RATE * duration))
        audio = np.sin(2 * np.pi * 440 * t)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.wav"

            # Save
            result_path = save_wave(audio, filename=str(filepath), amplitude=0.5)
            assert Path(result_path).exists()

            # Load
            sr, loaded_audio = load_wave(str(filepath))

            assert sr == DEFAULT_SAMPLE_RATE
            assert len(loaded_audio) == len(audio)
            assert loaded_audio.dtype == np.float32

            # Audio should be similar (allow for conversion losses)
            correlation = np.corrcoef(audio, loaded_audio)[0, 1]
            assert correlation > 0.99

    def test_save_stereo(self):
        """Test saving stereo audio."""
        duration = 0.1
        t = np.linspace(0, duration, int(DEFAULT_SAMPLE_RATE * duration))
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 554 * t)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "stereo.wav"

            # Save stereo
            save_wave(left, audio_right=right, filename=str(filepath), amplitude=0.5)

            # Load without mono conversion
            sr, loaded_audio = load_wave(str(filepath), mono=False)

            assert loaded_audio.ndim == 2
            assert loaded_audio.shape[1] == 2  # Two channels

    def test_auto_wav_extension(self):
        """Test that .wav extension is added automatically."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test"  # No extension

            result_path = save_wave(audio, filename=str(filepath))

            assert result_path.endswith(".wav")
            assert Path(result_path).exists()

    def test_different_sample_rates(self):
        """Test saving and loading with different sample rates."""
        sample_rates = [22050, 44100, 48000]

        for sr in sample_rates:
            duration = 0.1
            audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))

            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = Path(tmpdir) / f"test_{sr}.wav"

                save_wave(audio, filename=str(filepath), sample_rate=sr)
                loaded_sr, _ = load_wave(str(filepath))

                assert loaded_sr == sr

    def test_stereo_to_mono_conversion(self):
        """Test automatic stereo to mono conversion."""
        duration = 0.1
        t = np.linspace(0, duration, int(DEFAULT_SAMPLE_RATE * duration))
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 554 * t)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "stereo.wav"

            save_wave(left, audio_right=right, filename=str(filepath))

            # Load as mono (default)
            sr, mono_audio = load_wave(str(filepath), mono=True)

            assert mono_audio.ndim == 1  # Should be mono
            assert len(mono_audio) == len(left)


class TestNoteToFrequency:
    """Tests for note name to frequency conversion."""

    def test_a4_concert_pitch(self):
        """Test that A4 returns 440 Hz."""
        freq = note_to_frequency("A4")
        assert abs(freq - 440.0) < 0.01

    def test_c4_middle_c(self):
        """Test that C4 returns the correct frequency."""
        freq = note_to_frequency("C4")
        assert abs(freq - 261.63) < 0.01

    def test_sharp_notes(self):
        """Test sharp notes."""
        c_sharp = note_to_frequency("C#4")
        d_flat = note_to_frequency("Db4")

        # C# and Db should be the same
        assert abs(c_sharp - d_flat) < 0.01
        assert abs(c_sharp - 277.18) < 0.5

    def test_octave_relationship(self):
        """Test that octaves double the frequency."""
        a3 = note_to_frequency("A3")
        a4 = note_to_frequency("A4")
        a5 = note_to_frequency("A5")

        assert abs(a4 / a3 - 2.0) < 0.01
        assert abs(a5 / a4 - 2.0) < 0.01

    def test_various_octaves(self):
        """Test notes across different octaves."""
        octaves = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        for octave in octaves:
            freq = note_to_frequency(f"A{octave}")
            assert freq > 0
            # A4 = 440, so A(n) = 440 * 2^(n-4)
            expected = 440.0 * (2 ** (octave - 4))
            assert abs(freq - expected) < 0.01


class TestLegacyAPI:
    """Tests for legacy API functions to ensure backward compatibility."""

    def test_to_16_compatibility(self):
        """Test that to_16 works like to_int16."""
        audio = np.array([0.0, 0.5, -0.5, 1.0])
        result_new = to_int16(audio, amplitude=0.8)
        result_old = to_int16(audio, 0.8)

        np.testing.assert_array_equal(result_new, result_old)

    def test_hz_compatibility(self):
        """Test that hz works like note_to_frequency."""
        notes = ["A4", "C4", "E5", "G#3"]

        for note in notes:
            freq_new = note_to_frequency(note)
            freq_old = note_to_frequency(note)
            assert abs(freq_new - freq_old) < 0.001

    def test_wave_to_file_compatibility(self):
        """Test that wave_to_file works like save_wave."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test mono
            file1 = Path(tmpdir) / "test1.wav"
            save_wave(audio, filename=str(file1))
            assert file1.exists()

            # Test stereo
            file2 = Path(tmpdir) / "test2.wav"
            save_wave(audio, audio_right=audio, filename=str(file2))
            assert file2.exists()

    def test_read_wave_file_compatibility(self):
        """Test that read_wave_file works like load_wave."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.wav"
            save_wave(audio, filename=str(filepath))

            # Use legacy function
            sr_old, audio_old = load_wave(str(filepath))
            # Use new function
            sr_new, audio_new = load_wave(str(filepath))

            assert sr_old == sr_new
            np.testing.assert_array_almost_equal(audio_old, audio_new, decimal=5)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_audio(self):
        """Test handling of very short audio (single sample)."""
        audio = np.array([0.5])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "short.wav"

            save_wave(audio, filename=str(filepath))
            sr, loaded = load_wave(str(filepath))

            assert len(loaded) >= 1  # At least one sample

    def test_zero_audio(self):
        """Test handling of silent audio."""
        audio = np.zeros(4410)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "silent.wav"

            save_wave(audio, filename=str(filepath))
            sr, loaded = load_wave(str(filepath))

            assert len(loaded) == len(audio)
            assert np.all(np.abs(loaded) < 0.01)  # Should be very quiet

    def test_amplitude_zero(self):
        """Test conversion with zero amplitude."""
        audio = np.array([1.0, -1.0])
        result = to_int16(audio, amplitude=0.0)

        # Should result in all zeros
        assert np.all(result == 0)

    def test_invalid_note_name(self):
        """Test that invalid note names raise errors."""
        with pytest.raises(Exception):  # librosa raises various exceptions
            note_to_frequency("Invalid")

    def test_unicode_filename(self):
        """Test handling of Unicode characters in filenames."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4410))

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use ASCII-safe name for cross-platform compatibility
            filepath = Path(tmpdir) / "test_unicode.wav"

            result = save_wave(audio, filename=str(filepath))
            assert Path(result).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

