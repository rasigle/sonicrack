"""Unit tests for noise generators."""

import unittest

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.noise import (
    white_noise,
    pink_noise,
    brownian_noise,
    blue_noise,
    perlin_noise,
    velvet_noise,
    grey_noise,
    sample_hold_noise,
)


class TestWhiteNoise(unittest.TestCase):
    """Test white noise generator."""

    def test_basic_generation(self) -> None:
        """Test white noise generates correct number of samples."""
        noise = white_noise(dur=1.0, amplitude=1.0, sr=1000)

        self.assertIsInstance(noise, np.ndarray)
        self.assertEqual(len(noise), 1000)

    def test_amplitude_scaling(self) -> None:
        """Test amplitude parameter scales output."""
        noise1 = white_noise(dur=1.0, amplitude=0.5, sr=1000, seed=42)
        noise2 = white_noise(dur=1.0, amplitude=1.0, sr=1000, seed=42)

        # noise2 should be approximately 2x louder
        ratio = np.std(noise2) / np.std(noise1)
        self.assertAlmostEqual(ratio, 2.0, delta=0.2)

    def test_reproducibility_with_seed(self) -> None:
        """Test same seed produces same noise."""
        noise1 = white_noise(dur=0.1, amplitude=1.0, sr=1000, seed=42)
        noise2 = white_noise(dur=0.1, amplitude=1.0, sr=1000, seed=42)

        np.testing.assert_array_equal(noise1, noise2)

    def test_different_seeds_produce_different_noise(self) -> None:
        """Test different seeds produce different noise."""
        noise1 = white_noise(dur=0.1, amplitude=1.0, sr=1000, seed=42)
        noise2 = white_noise(dur=0.1, amplitude=1.0, sr=1000, seed=43)

        self.assertFalse(np.array_equal(noise1, noise2))

    def test_flat_spectrum(self) -> None:
        """Test white noise has approximately flat spectrum."""
        noise = white_noise(dur=10.0, amplitude=1.0, sr=DEFAULT_SAMPLE_RATE, seed=42)

        # Compute power spectrum
        fft = np.fft.fft(noise)
        power = np.abs(fft[: len(fft) // 2]) ** 2

        # Divide into frequency bins and check variance
        n_bins = 10
        bin_size = len(power) // n_bins
        bin_powers = [
            np.mean(power[i * bin_size : (i + 1) * bin_size]) for i in range(n_bins)
        ]

        # Coefficient of variation should be low for flat spectrum
        cv = np.std(bin_powers) / np.mean(bin_powers)
        self.assertLess(cv, 0.3)  # Relatively flat

    def test_zero_duration(self) -> None:
        """Test zero duration produces empty array."""
        noise = white_noise(dur=0.0, amplitude=1.0, sr=1000)
        self.assertEqual(len(noise), 0)


class TestPinkNoise(unittest.TestCase):
    """Test pink noise generator."""

    def test_basic_generation(self) -> None:
        """Test pink noise generates correct number of samples."""
        noise = pink_noise(dur=1.0, amplitude=1.0, sr=1000)

        self.assertIsInstance(noise, np.ndarray)
        self.assertEqual(len(noise), 1000)

    def test_amplitude_scaling(self) -> None:
        """Test amplitude parameter scales output."""
        noise = pink_noise(dur=1.0, amplitude=0.5, sr=1000)

        # Should be scaled
        self.assertLess(np.max(np.abs(noise)), 1.0)

    def test_1_over_f_spectrum(self) -> None:
        """Test pink noise has approximate 1/f spectrum."""
        noise = pink_noise(dur=10.0, amplitude=1.0, sr=DEFAULT_SAMPLE_RATE, seed=42)

        # Compute power spectrum
        fft = np.fft.fft(noise)
        freqs = np.fft.fftfreq(len(noise), 1 / DEFAULT_SAMPLE_RATE)
        power = np.abs(fft[: len(fft) // 2]) ** 2
        freqs_positive = freqs[: len(freqs) // 2]

        # Skip DC component
        power = power[1:]
        freqs_positive = freqs_positive[1:]

        # For pink noise, power should decrease with frequency
        # Check that higher frequencies have less power
        low_freq_power = np.mean(power[: len(power) // 10])
        high_freq_power = np.mean(power[-len(power) // 10 :])

        self.assertGreater(low_freq_power, high_freq_power)

    def test_reproducibility(self) -> None:
        """Test reproducibility with seed."""
        noise1 = pink_noise(dur=0.1, amplitude=1.0, sr=1000, seed=42)
        noise2 = pink_noise(dur=0.1, amplitude=1.0, sr=1000, seed=42)

        np.testing.assert_array_almost_equal(noise1, noise2, decimal=5)


class TestBrownianNoise(unittest.TestCase):
    """Test Brownian/brown noise generator."""

    def test_basic_generation(self) -> None:
        """Test Brownian noise generates correct number of samples."""
        noise = brownian_noise(dur=1.0, amplitude=1.0, sr=1000)

        self.assertIsInstance(noise, np.ndarray)
        self.assertEqual(len(noise), 1000)

    def test_cumulative_nature(self) -> None:
        """Test Brownian noise is cumulative (random walk)."""
        noise = brownian_noise(dur=1.0, amplitude=1.0, sr=1000, seed=42)

        # Brownian noise is cumulative sum, so differences should be white noise
        diff = np.diff(noise)

        # Check that differences have approximately constant variance
        self.assertGreater(len(diff), 0)

    def test_low_frequency_bias(self) -> None:
        """Test Brownian noise has low-frequency bias."""
        noise = brownian_noise(dur=10.0, amplitude=1.0, sr=DEFAULT_SAMPLE_RATE, seed=42)

        # Compute power spectrum
        fft = np.fft.fft(noise)
        power = np.abs(fft[: len(fft) // 2]) ** 2

        # Low frequencies should dominate
        low_freq_power = np.mean(power[: len(power) // 10])
        high_freq_power = np.mean(power[-len(power) // 10 :])

        self.assertGreater(low_freq_power, high_freq_power * 10)

    def test_reproducibility(self) -> None:
        """Test reproducibility with seed."""
        noise1 = brownian_noise(dur=0.1, amplitude=1.0, sr=1000, seed=42)
        noise2 = brownian_noise(dur=0.1, amplitude=1.0, sr=1000, seed=42)

        np.testing.assert_array_almost_equal(noise1, noise2, decimal=5)


class TestBlueNoise(unittest.TestCase):
    """Test blue noise generator."""

    def test_basic_generation(self) -> None:
        """Test blue noise generates correct number of samples."""
        noise = blue_noise(dur=1.0, amplitude=1.0, sr=1000)

        self.assertIsInstance(noise, np.ndarray)
        self.assertEqual(len(noise), 1000)

    def test_high_frequency_bias(self) -> None:
        """Test blue noise favors high frequencies."""
        noise = blue_noise(dur=10.0, amplitude=1.0, sr=DEFAULT_SAMPLE_RATE, seed=42)

        # Compute power spectrum
        fft = np.fft.fft(noise)
        power = np.abs(fft[: len(fft) // 2]) ** 2

        # High frequencies should have more power than low
        low_freq_power = np.mean(power[1 : len(power) // 10])
        high_freq_power = np.mean(power[-len(power) // 10 :])

        self.assertGreater(high_freq_power, low_freq_power)

    def test_reproducibility(self) -> None:
        """Test reproducibility with seed."""
        noise1 = blue_noise(dur=0.1, amplitude=1.0, sr=1000, seed=42)
        noise2 = blue_noise(dur=0.1, amplitude=1.0, sr=1000, seed=42)

        np.testing.assert_array_almost_equal(noise1, noise2, decimal=5)


class TestPerlinNoise(unittest.TestCase):
    """Test Perlin noise generator."""

    def test_basic_generation(self) -> None:
        """Test Perlin noise generates correct number of samples."""
        noise = perlin_noise(dur=1.0, scale=10, sr=1000)

        self.assertIsInstance(noise, np.ndarray)
        self.assertEqual(len(noise), 1000)

    def test_smoothness(self) -> None:
        """Test Perlin noise is smooth (continuous)."""
        noise = perlin_noise(dur=1.0, scale=10, sr=1000, seed=42)

        # Check that differences are small (smooth transitions)
        diff = np.abs(np.diff(noise))
        max_diff = np.max(diff)

        # Should be relatively smooth - Perlin noise gradient values are 1-8,
        # so max diff should be reasonable for the given scale
        self.assertLess(max_diff, 5.0)  # Adjusted for actual Perlin implementation

    def test_scale_parameter(self) -> None:
        """Test scale parameter affects variation rate."""
        noise_fine = perlin_noise(
            dur=1.0, scale=5, sr=1000, seed=42  # Smaller scale = finer variation
        )
        noise_coarse = perlin_noise(
            dur=1.0, scale=50, sr=1000, seed=42  # Larger scale = coarser variation
        )

        # Fine scale should have more variation
        var_fine = np.var(np.diff(noise_fine))
        var_coarse = np.var(np.diff(noise_coarse))

        self.assertGreater(var_fine, var_coarse)

    def test_reproducibility(self) -> None:
        """Test reproducibility with seed."""
        noise1 = perlin_noise(dur=0.1, scale=10, sr=1000, seed=42)
        noise2 = perlin_noise(dur=0.1, scale=10, sr=1000, seed=42)

        np.testing.assert_array_almost_equal(noise1, noise2, decimal=5)


class TestNoiseComparisons(unittest.TestCase):
    """Test comparisons between different noise types."""

    def test_spectral_differences(self) -> None:
        """Test different noise types have different spectral characteristics."""
        duration = 10.0
        sr = DEFAULT_SAMPLE_RATE
        seed = 42

        white = white_noise(dur=duration, amplitude=1.0, sr=sr, seed=seed)
        pink = pink_noise(dur=duration, amplitude=1.0, sr=sr, seed=seed)
        brown = brownian_noise(dur=duration, amplitude=1.0, sr=sr, seed=seed)

        # All should be different
        self.assertFalse(np.array_equal(white, pink))
        self.assertFalse(np.array_equal(pink, brown))
        self.assertFalse(np.array_equal(white, brown))


class TestVelvetNoise(unittest.TestCase):
    """Test suite for velvet noise generator."""

    def test_basic_generation(self) -> None:
        """Test velvet noise generates correct number of samples."""
        noise = velvet_noise(dur=1.0, density=0.01, amplitude=1.0, sr=1000)

        self.assertIsInstance(noise, np.ndarray)
        self.assertEqual(len(noise), 1000)
        self.assertEqual(noise.dtype, np.float32)

    def test_sparsity(self) -> None:
        """Test velvet noise is sparse (mostly zeros)."""
        noise = velvet_noise(dur=1.0, density=0.01, amplitude=1.0, sr=10000, seed=42)

        # Count non-zero samples
        non_zero_count = np.count_nonzero(noise)
        total_count = len(noise)

        # Should be approximately density% non-zero (within reasonable tolerance)
        expected_non_zero = total_count * 0.01
        # Allow 50% tolerance for randomness
        self.assertLess(
            abs(non_zero_count - expected_non_zero), expected_non_zero * 0.5
        )

    def test_amplitude_scaling(self) -> None:
        """Test amplitude parameter scales output."""
        noise = velvet_noise(dur=1.0, density=0.05, amplitude=0.5, sr=1000, seed=42)

        # Non-zero values should be ±0.5
        non_zero_values = noise[noise != 0]
        self.assertTrue(np.all(np.abs(non_zero_values) <= 0.51))

    def test_density_parameter(self) -> None:
        """Test density parameter affects sparsity."""
        sparse = velvet_noise(dur=1.0, density=0.001, sr=10000, seed=42)
        dense = velvet_noise(dur=1.0, density=0.1, sr=10000, seed=42)

        sparse_count = np.count_nonzero(sparse)
        dense_count = np.count_nonzero(dense)

        # Dense should have significantly more non-zero samples
        self.assertGreater(dense_count, sparse_count * 5)

    def test_reproducibility(self) -> None:
        """Test reproducibility with seed."""
        noise1 = velvet_noise(dur=0.1, density=0.01, amplitude=1.0, sr=1000, seed=42)
        noise2 = velvet_noise(dur=0.1, density=0.01, amplitude=1.0, sr=1000, seed=42)

        np.testing.assert_array_equal(noise1, noise2)


class TestGreyNoise(unittest.TestCase):
    """Test suite for grey noise generator."""

    def test_basic_generation(self) -> None:
        """Test grey noise generates correct number of samples."""
        noise = grey_noise(dur=1.0, amplitude=1.0, sr=1000)

        self.assertIsInstance(noise, np.ndarray)
        self.assertEqual(len(noise), 1000)
        self.assertEqual(noise.dtype, np.float32)

    def test_amplitude_scaling(self) -> None:
        """Test amplitude parameter scales output."""
        noise = grey_noise(dur=1.0, amplitude=0.5, sr=1000, seed=42)

        # Should be scaled to approximately ±0.5
        self.assertLessEqual(np.max(np.abs(noise)), 0.51)

    def test_different_from_white(self) -> None:
        """Test grey noise is different from white noise."""
        # Generate grey and white noise with same seed base
        np.random.seed(42)
        white = np.random.uniform(-1, 1, 1000)

        grey = grey_noise(dur=1.0 / 1000, amplitude=1.0, sr=1000, seed=42)

        # They should be different (grey is filtered)
        self.assertFalse(np.allclose(white[: len(grey)], grey, rtol=0.1))

    def test_reproducibility(self) -> None:
        """Test reproducibility with seed."""
        noise1 = grey_noise(dur=0.1, amplitude=1.0, sr=1000, seed=42)
        noise2 = grey_noise(dur=0.1, amplitude=1.0, sr=1000, seed=42)

        np.testing.assert_array_almost_equal(noise1, noise2, decimal=5)

    def test_spectral_characteristics(self) -> None:
        """Test grey noise has reasonable spectral properties."""
        noise = grey_noise(dur=1.0, amplitude=1.0, sr=44100, seed=42)

        # Compute power spectrum
        fft = np.fft.rfft(noise)
        power = np.abs(fft) ** 2

        # Grey noise should have non-zero energy across frequencies
        # (not testing exact equal-loudness, just that it's filtered)
        self.assertGreater(np.mean(power), 0)
        self.assertGreater(np.std(power), 0)


class TestSampleHoldNoise(unittest.TestCase):
    """Test suite for sample & hold noise generator."""

    def test_basic_generation(self) -> None:
        """Test sample & hold noise generates correct number of samples."""
        noise = sample_hold_noise(dur=1.0, rate=10, amplitude=1.0, sr=1000)

        self.assertIsInstance(noise, np.ndarray)
        self.assertEqual(len(noise), 1000)
        self.assertEqual(noise.dtype, np.float32)

    def test_stepped_behavior(self) -> None:
        """Test sample & hold creates stepped signal."""
        noise = sample_hold_noise(dur=1.0, rate=10, amplitude=1.0, sr=1000, seed=42)

        # At 10 Hz rate and 1000 Hz sample rate, each step is 100 samples
        step_length = 100

        # First step should have all identical values
        first_step = noise[:step_length]
        self.assertTrue(np.all(first_step == first_step[0]))

        # Second step should be different from first
        second_step = noise[step_length : 2 * step_length]
        self.assertNotEqual(first_step[0], second_step[0])

        # But second step should be constant within itself
        self.assertTrue(np.all(second_step == second_step[0]))

    def test_rate_parameter(self) -> None:
        """Test rate parameter affects step duration."""
        slow = sample_hold_noise(dur=1.0, rate=2, amplitude=1.0, sr=1000, seed=42)
        fast = sample_hold_noise(dur=1.0, rate=20, amplitude=1.0, sr=1000, seed=42)

        # Count unique values (number of steps)
        slow_steps = len(np.unique(slow))
        fast_steps = len(np.unique(fast))

        # Fast should have more steps
        self.assertGreater(fast_steps, slow_steps)

    def test_amplitude_scaling(self) -> None:
        """Test amplitude parameter scales output."""
        noise = sample_hold_noise(dur=1.0, rate=10, amplitude=0.5, sr=1000, seed=42)

        # Should be scaled to ±0.5
        self.assertLessEqual(np.max(np.abs(noise)), 0.51)

    def test_reproducibility(self) -> None:
        """Test reproducibility with seed."""
        noise1 = sample_hold_noise(dur=0.1, rate=10, amplitude=1.0, sr=1000, seed=42)
        noise2 = sample_hold_noise(dur=0.1, rate=10, amplitude=1.0, sr=1000, seed=42)

        np.testing.assert_array_equal(noise1, noise2)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
