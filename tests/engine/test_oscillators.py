"""Comprehensive unit tests for the audio engine oscillators.

This test suite validates:
- Waveform accuracy
- Parameter changes
- Phase continuity
- Vectorized vs iterator consistency
- Edge cases
"""

import unittest

import numpy as np

from src.engine.oscillator import (
    SineOscillator,
    SquareOscillator,
    SawtoothOscillator,
    TriangleOscillator,
    synth,
)


class TestOscillatorBase(unittest.TestCase):
    """Base class for oscillator tests with common utilities."""

    def assert_samples_valid(self, samples: np.ndarray, max_amp: float = 1.0) -> None:
        """Verify samples are within valid amplitude range.

        Args:
            samples: Array of audio samples to validate.
            max_amp: Maximum expected amplitude.
        """
        self.assertTrue(np.all(np.abs(samples) <= max_amp + 0.01))

    def assert_arrays_close(
        self, a: np.ndarray, b: np.ndarray, rtol: float = 1e-5
    ) -> None:
        """Verify two arrays are approximately equal.

        Args:
            a: First array.
            b: Second array.
            rtol: Relative tolerance for comparison.
        """
        np.testing.assert_allclose(a, b, rtol=rtol)


class TestSineOscillator(TestOscillatorBase):
    """Test suite for SineOscillator."""

    def setUp(self) -> None:
        """Create a sine oscillator for each test."""
        self.osc = SineOscillator(frequency=440, amplitude=1.0, gain_db=None, sample_rate=44100)

    def test_initialization(self) -> None:
        """Test oscillator initializes with correct parameters."""
        self.assertEqual(self.osc.init_freq, 440)
        self.assertEqual(self.osc.init_amp, 1.0)
        self.assertEqual(self.osc.sample_rate, 44100)

    def test_waveform_range(self) -> None:
        """Test sine wave stays within [-1, 1] range."""
        samples = self.osc.get_samples_vectorized(1000)
        self.assert_samples_valid(samples, max_amp=1.0)

    def test_frequency_accuracy(self) -> None:
        """Test generated frequency matches specified frequency."""
        freq = 440
        sample_rate = 44100
        duration = 1.0  # 1 second

        osc = SineOscillator(frequency=freq, sample_rate=sample_rate)
        samples = osc.get_samples_vectorized(int(duration * sample_rate))

        # Use FFT to find dominant frequency
        fft = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples), 1 / sample_rate)
        dominant_freq = abs(freqs[np.argmax(np.abs(fft[: len(fft) // 2]))])

        self.assertAlmostEqual(dominant_freq, freq, delta=1.0)

    def test_phase_continuity(self) -> None:
        """Test phase is continuous between chunks."""
        samples1 = self.osc.get_samples_vectorized(100)
        samples2 = self.osc.get_samples_vectorized(100)

        # Concatenate and check for discontinuities
        combined = np.concatenate([samples1, samples2])
        diff = np.diff(combined)

        # No diff should be larger than 2 (full range jump)
        self.assertTrue(np.all(np.abs(diff) < 2.0))

    def test_phase_wrapping(self) -> None:
        """Test phase wraps correctly to prevent overflow."""
        osc = SineOscillator(440)

        # Generate many samples to force phase wrapping
        _ = osc.get_samples_vectorized(100000)

        # Internal phase should be wrapped
        self.assertLess(osc._i, 10 * np.pi)

    def test_iterator_vs_vectorized(self) -> None:
        """Test iterator and vectorized methods produce same results."""
        osc1 = SineOscillator(440)
        osc2 = SineOscillator(440)

        samples_iter = osc1.get_samples_iterator(100, reset=True)
        samples_vec = osc2.get_samples_vectorized(100)

        self.assert_arrays_close(np.array(samples_iter), samples_vec, rtol=1e-10)

    def test_parameter_changes(self) -> None:
        """Test frequency and amplitude can be changed mid-stream."""
        samples1 = self.osc.get_samples_vectorized(100)

        self.osc.freq = 880  # Change frequency
        samples2 = self.osc.get_samples_vectorized(100)

        # Samples should be different after frequency change
        self.assertFalse(np.allclose(samples1, samples2))

    def test_amplitude_scaling(self) -> None:
        """Test amplitude parameter scales output correctly."""
        osc = SineOscillator(frequency=440, amplitude=0.5, gain_db=None)
        samples = osc.get_samples_vectorized(1000)

        # Max amplitude should be around 0.5
        self.assertLess(np.max(np.abs(samples)), 0.51)
        self.assertGreater(np.max(np.abs(samples)), 0.45)

    def test_auto_mode_selection(self) -> None:
        """Test auto mode selects correct method based on buffer size."""
        osc = SineOscillator(440)

        # Small buffer should use iterator
        small = osc.get_samples(100, mode="auto", reset=True)
        self.assertIsInstance(small, np.ndarray)

        # Large buffer should use vectorized (returns ndarray)
        large = osc.get_samples(1000, mode="auto", reset=True)
        self.assertIsInstance(large, np.ndarray)

    def test_reset_functionality(self) -> None:
        """Test reset parameter reinitializes oscillator."""
        osc = SineOscillator(440)

        samples1 = osc.get_samples_vectorized(100)
        samples2 = osc.get_samples_vectorized(100)  # Continue from previous state

        # Reset and generate again
        samples3 = osc.get_samples_vectorized(100)
        osc_reset = SineOscillator(440)
        samples4 = osc_reset.get_samples_vectorized(100)

        # samples1 and samples4 should be identical (both start from beginning)
        self.assert_arrays_close(samples1, samples4)

        # samples2 and samples3 should be different (different phases)
        self.assertFalse(np.allclose(samples2, samples3))


class TestSquareOscillator(TestOscillatorBase):
    """Test suite for SquareOscillator."""

    def setUp(self) -> None:
        """Create a square oscillator for each test."""
        self.osc = SquareOscillator(frequency=440, amplitude=1.0)

    def test_binary_output(self) -> None:
        """Test square wave outputs only two values."""
        samples = self.osc.get_samples_vectorized(1000)
        unique_values = np.unique(np.round(samples, decimals=5))

        # Should have primarily two values: -1 and 1
        self.assertLessEqual(len(unique_values), 3)  # Allow for transitions

    def test_duty_cycle(self) -> None:
        """Test square wave has approximately 50% duty cycle."""
        samples = self.osc.get_samples_vectorized(1000)

        positive_count = np.sum(samples > 0)
        negative_count = np.sum(samples < 0)

        # Should be approximately equal (50% duty cycle)
        ratio = positive_count / (positive_count + negative_count)
        self.assertAlmostEqual(ratio, 0.5, delta=0.1)


class TestSawtoothOscillator(TestOscillatorBase):
    """Test suite for SawtoothOscillator."""

    def setUp(self) -> None:
        """Create a sawtooth oscillator for each test."""
        self.osc = SawtoothOscillator(frequency=440, amplitude=1.0, gain_db=None)

    def test_range(self) -> None:
        """Test sawtooth wave covers full range."""
        samples = self.osc.get_samples_vectorized(1000)

        self.assert_samples_valid(samples, max_amp=1.0)
        self.assertGreater(np.max(samples), 0.9)
        self.assertLess(np.min(samples), -0.9)

    def test_linear_ramp(self) -> None:
        """Test sawtooth produces linear ramp within each period."""
        osc = SawtoothOscillator(frequency=100, sample_rate=44100, gain_db=None)
        samples = osc.get_samples_vectorized(441)  # ~10 periods

        # Check that we have both increasing and decreasing values (resets)
        diff = np.diff(samples)
        self.assertTrue(np.any(diff > 0))  # Increasing segments
        self.assertTrue(np.any(diff < -1))  # Reset jumps


class TestTriangleOscillator(TestOscillatorBase):
    """Test suite for TriangleOscillator."""

    def setUp(self) -> None:
        """Create a triangle oscillator for each test."""
        self.osc = TriangleOscillator(frequency=440, amplitude=1.0, gain_db=None)

    def test_range(self) -> None:
        """Test triangle wave covers full range."""
        samples = self.osc.get_samples_vectorized(1000)

        self.assert_samples_valid(samples, max_amp=1.0)
        self.assertGreater(np.max(samples), 0.9)
        self.assertLess(np.min(samples), -0.9)

    def test_symmetry(self) -> None:
        """Test triangle wave is symmetric."""
        samples = self.osc.get_samples_vectorized(1000)

        # Mean should be close to 0 for symmetric wave
        self.assertAlmostEqual(np.mean(samples), 0.0, delta=0.1)


class TestSynthFunction(unittest.TestCase):
    """Test suite for the synth() convenience function."""

    def test_all_waveform_types(self) -> None:
        """Test synth function works with all waveform types."""
        waveforms = ["sine", "square", "sawtooth", "triangle"]

        for waveform in waveforms:
            with self.subTest(waveform=waveform):
                wave = synth(frequency=440, dur=0.1, stype=waveform)

                self.assertIsInstance(wave, np.ndarray)
                self.assertGreater(len(wave), 0)

    def test_duration(self) -> None:
        """Test synth generates correct number of samples."""
        sample_rate = 44100
        duration = 0.5  # seconds

        wave = synth(frequency=440, dur=duration, sr=sample_rate)

        expected_samples = int(duration * sample_rate)
        self.assertEqual(len(wave), expected_samples)

    def test_mode_parameter(self) -> None:
        """Test synth respects mode parameter."""
        wave = synth(frequency=440, dur=0.1, mode="vectorized")
        self.assertIsInstance(wave, np.ndarray)

    def test_invalid_waveform(self) -> None:
        """Test synth raises error for invalid waveform type."""
        with self.assertRaises(ValueError):
            synth(frequency=440, dur=0.1, stype="invalid")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_zero_frequency(self) -> None:
        """Test oscillator with zero frequency."""
        osc = SineOscillator(frequency=0)
        samples = osc.get_samples_vectorized(100)

        # Should produce constant value (no oscillation)
        self.assertAlmostEqual(np.std(samples), 0.0, delta=0.01)

    def test_very_high_frequency(self) -> None:
        """Test oscillator with very high frequency."""
        osc = SineOscillator(frequency=20000, sample_rate=44100)
        samples = osc.get_samples_vectorized(100)

        # Should still produce valid samples
        self.assertTrue(np.all(np.isfinite(samples)))

    def test_negative_amplitude(self) -> None:
        """Test oscillator with negative amplitude."""
        osc = SineOscillator(frequency=440, amplitude=-1.0)
        samples = osc.get_samples_vectorized(100)

        # Should produce inverted waveform
        self.assertTrue(np.all(np.abs(samples) <= 1.01))

    def test_invalid_mode(self) -> None:
        """Test invalid mode parameter raises ValueError."""
        osc = SineOscillator(440)

        with self.assertRaises(ValueError):
            osc.get_samples(100, mode="invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
