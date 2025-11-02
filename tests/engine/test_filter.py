"""Unit tests for filter module."""

import unittest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.engine.filter import butter, apply_filter
from src.constants import DEFAULT_SAMPLE_RATE


class TestCreateButterFilter(unittest.TestCase):
    """Test Butterworth filter creation."""

    def test_lowpass_filter_creation(self) -> None:
        """Test creating a low-pass filter."""
        b, a = butter(
            order=4,
            cutoff=1000,
            fs=DEFAULT_SAMPLE_RATE,
            btype="low"
        )

        self.assertIsInstance(b, np.ndarray)
        self.assertIsInstance(a, np.ndarray)
        self.assertGreater(len(b), 0)
        self.assertGreater(len(a), 0)

    def test_highpass_filter_creation(self) -> None:
        """Test creating a high-pass filter."""
        b, a = butter(
            order=4,
            cutoff=100,
            fs=DEFAULT_SAMPLE_RATE,
            btype="high"
        )

        self.assertIsInstance(b, np.ndarray)
        self.assertIsInstance(a, np.ndarray)

    def test_bandpass_filter_creation(self) -> None:
        """Test creating a band-pass filter."""
        b, a = butter(
            order=4,
            cutoff=(100, 1000),
            fs=DEFAULT_SAMPLE_RATE,
            btype="band"
        )

        self.assertIsInstance(b, np.ndarray)
        self.assertIsInstance(a, np.ndarray)

    def test_filter_order(self) -> None:
        """Test different filter orders."""
        for order in [2, 4, 6, 8]:
            with self.subTest(order=order):
                b, a = butter(
                    order=order,
                    cutoff=1000,
                    fs=DEFAULT_SAMPLE_RATE,
                    btype="low"
                )

                # Higher order should have more coefficients
                self.assertGreaterEqual(len(b), order)

    def test_cutoff_at_nyquist(self) -> None:
        """Test cutoff frequency at Nyquist limit."""
        nyquist = DEFAULT_SAMPLE_RATE / 2

        b, a = butter(
            order=4,
            cutoff=nyquist * 0.9,  # Just below Nyquist
            fs=DEFAULT_SAMPLE_RATE,
            btype="low"
        )

        self.assertIsInstance(b, np.ndarray)


class TestApplyFilter(unittest.TestCase):
    """Test filter application."""

    def test_lowpass_attenuates_high_freq(self) -> None:
        """Test low-pass filter attenuates high frequencies."""
        # Create signal: low freq + high freq
        t = np.linspace(0, 1, DEFAULT_SAMPLE_RATE, endpoint=False)
        low_freq = np.sin(2 * np.pi * 100 * t)  # 100 Hz
        high_freq = np.sin(2 * np.pi * 5000 * t)  # 5000 Hz
        signal = low_freq + high_freq

        # Create low-pass filter at 1000 Hz
        b, a = butter(
            order=6,
            cutoff=1000,
            fs=DEFAULT_SAMPLE_RATE,
            btype="low"
        )

        # Apply filter
        filtered = apply_filter(b, a, signal)

        # High frequency component should be attenuated
        # Check energy in high frequency band using FFT
        fft_original = np.fft.fft(signal)
        fft_filtered = np.fft.fft(filtered)
        freqs = np.fft.fftfreq(len(signal), 1/DEFAULT_SAMPLE_RATE)

        # Energy above 2000 Hz should be reduced
        high_freq_mask = np.abs(freqs) > 2000
        original_high_energy = np.sum(np.abs(fft_original[high_freq_mask])**2)
        filtered_high_energy = np.sum(np.abs(fft_filtered[high_freq_mask])**2)

        self.assertLess(filtered_high_energy, original_high_energy * 0.1)

    def test_highpass_attenuates_low_freq(self) -> None:
        """Test high-pass filter attenuates low frequencies."""
        # Create signal: low freq + high freq
        t = np.linspace(0, 1, DEFAULT_SAMPLE_RATE, endpoint=False)
        low_freq = np.sin(2 * np.pi * 50 * t)  # 50 Hz
        high_freq = np.sin(2 * np.pi * 3000 * t)  # 3000 Hz
        signal = low_freq + high_freq

        # Create high-pass filter at 1000 Hz
        b, a = butter(
            order=6,
            cutoff=1000,
            fs=DEFAULT_SAMPLE_RATE,
            btype="high"
        )

        # Apply filter
        filtered = apply_filter(b, a, signal)

        # Low frequency component should be attenuated
        fft_original = np.fft.fft(signal)
        fft_filtered = np.fft.fft(filtered)
        freqs = np.fft.fftfreq(len(signal), 1/DEFAULT_SAMPLE_RATE)

        # Energy below 500 Hz should be reduced
        low_freq_mask = np.abs(freqs) < 500
        original_low_energy = np.sum(np.abs(fft_original[low_freq_mask])**2)
        filtered_low_energy = np.sum(np.abs(fft_filtered[low_freq_mask])**2)

        self.assertLess(filtered_low_energy, original_low_energy * 0.1)

    def test_filter_output_length(self) -> None:
        """Test filtered signal has same length as input."""
        signal = np.random.randn(1000)

        b, a = butter(
            order=4,
            cutoff=1000,
            fs=DEFAULT_SAMPLE_RATE,
            btype="low"
        )

        filtered = apply_filter(b, a, signal)

        self.assertEqual(len(filtered), len(signal))

    def test_filter_stability(self) -> None:
        """Test filter doesn't produce NaN or Inf."""
        signal = np.random.randn(1000)

        b, a = butter(
            order=4,
            cutoff=1000,
            fs=DEFAULT_SAMPLE_RATE,
            btype="low"
        )

        filtered = apply_filter(b, a, signal)

        self.assertTrue(np.all(np.isfinite(filtered)))

    def test_zero_phase_filtering(self) -> None:
        """Test filtfilt provides zero-phase filtering."""
        # Create a sharp pulse
        signal = np.zeros(1000)
        signal[500] = 1.0

        b, a = butter(
            order=4,
            cutoff=100,
            fs=1000,
            btype="low"
        )

        filtered = apply_filter(b, a, signal)

        # Peak should still be near center (zero phase delay)
        peak_idx = np.argmax(np.abs(filtered))
        self.assertLess(abs(peak_idx - 500), 50)


class TestBandpassFilter(unittest.TestCase):
    """Test band-pass filter specific functionality."""

    def test_bandpass_passes_band(self) -> None:
        """Test band-pass filter passes frequencies in band."""
        # Create three frequencies
        t = np.linspace(0, 1, DEFAULT_SAMPLE_RATE, endpoint=False)
        low_freq = np.sin(2 * np.pi * 50 * t)  # Below band
        mid_freq = np.sin(2 * np.pi * 500 * t)  # In band
        high_freq = np.sin(2 * np.pi * 5000 * t)  # Above band
        signal = low_freq + mid_freq + high_freq

        # Create band-pass filter: 200-1000 Hz
        b, a = butter(
            order=5,
            cutoff=(200, 1000),
            fs=DEFAULT_SAMPLE_RATE,
            btype="band"
        )

        filtered = apply_filter(b, a, signal)

        # FFT analysis
        fft_filtered = np.fft.fft(filtered)
        freqs = np.fft.fftfreq(len(signal), 1/DEFAULT_SAMPLE_RATE)

        # Energy at 500 Hz should be preserved
        freq_500_idx = np.argmin(np.abs(freqs - 500))
        print(freqs[freq_500_idx])
        self.assertGreater(np.abs(fft_filtered[freq_500_idx]), 100)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_empty_signal(self) -> None:
        """Test filtering empty signal."""
        signal = np.array([])

        b, a = butter(
            order=4,
            cutoff=1000,
            fs=DEFAULT_SAMPLE_RATE,
            btype="low"
        )

        # Should handle gracefully or raise appropriate error
        try:
            filtered = apply_filter(b, a, signal)
            self.assertEqual(len(filtered), 0)
        except ValueError:
            # Also acceptable to raise an error
            pass

    def test_single_sample(self) -> None:
        """Test filtering single sample."""
        signal = np.array([1.0])

        b, a = butter(
            order=2,
            cutoff=1000,
            fs=DEFAULT_SAMPLE_RATE,
            btype="low"
        )

        # Should handle gracefully
        try:
            filtered = apply_filter(b, a, signal)
            self.assertTrue(np.isfinite(filtered[0]))
        except ValueError:
            # May raise error for very short signals
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)

