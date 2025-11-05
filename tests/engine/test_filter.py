"""Tests for Butterworth filter implementation."""

import unittest
import numpy as np

from src.engine.filter import ButterworthFilter, butter, apply_filter
from src.constants import DEFAULT_SAMPLE_RATE


class TestButterworthFilterBasics(unittest.TestCase):
    """Test basic Butterworth filter functionality."""

    def test_initialization(self):
        """Test filter can be initialized with default parameters."""
        filt = ButterworthFilter()
        self.assertEqual(filt.cutoff, 1000.0)
        self.assertEqual(filt.order, 4)
        self.assertEqual(filt.filter_type, "low")
        self.assertEqual(filt.sample_rate, DEFAULT_SAMPLE_RATE)

    def test_initialization_custom_params(self):
        """Test filter initialization with custom parameters."""
        filt = ButterworthFilter(
            cutoff=2000, order=6, filter_type="high", sample_rate=48000
        )
        self.assertEqual(filt.cutoff, 2000)
        self.assertEqual(filt.order, 6)
        self.assertEqual(filt.filter_type, "high")
        self.assertEqual(filt.sample_rate, 48000)

    def test_band_pass_initialization(self):
        """Test band-pass filter initialization."""
        filt = ButterworthFilter(cutoff=(200, 2000), filter_type="band")
        self.assertEqual(filt.cutoff, (200, 2000))
        self.assertEqual(filt.filter_type, "band")

    def test_invalid_filter_type(self):
        """Test that invalid filter type raises ValueError."""
        with self.assertRaises(ValueError):
            ButterworthFilter(filter_type="invalid")

    def test_invalid_band_cutoff(self):
        """Test that invalid band-pass cutoff raises ValueError."""
        # Single value for band-pass
        with self.assertRaises(ValueError):
            ButterworthFilter(cutoff=1000, filter_type="band")

        # Low >= high
        with self.assertRaises(ValueError):
            ButterworthFilter(cutoff=(2000, 1000), filter_type="band")


class TestButterworthFilterProcessing(unittest.TestCase):
    """Test filter signal processing."""

    def test_low_pass_attenuates_high_frequencies(self):
        """Test that low-pass filter attenuates high frequencies."""
        sample_rate = 10000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Create signal with low and high frequency components
        low_freq = 100  # Below cutoff
        high_freq = 4000  # Above cutoff
        signal_in = np.sin(2 * np.pi * low_freq * t) + np.sin(2 * np.pi * high_freq * t)

        # Apply low-pass filter at 500 Hz
        filt = ButterworthFilter(cutoff=500, order=4, sample_rate=sample_rate)
        filtered = filt.scale_vectorized(signal_in.astype(np.float32))

        # Compute FFT to check frequency content
        _ = np.abs(np.fft.rfft(signal_in))
        fft_out = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(signal_in), 1 / sample_rate)

        # Find peaks at low and high frequencies
        low_idx = np.argmin(np.abs(freqs - low_freq))
        high_idx = np.argmin(np.abs(freqs - high_freq))

        # High frequency should be attenuated significantly
        attenuation_ratio = fft_out[high_idx] / fft_out[low_idx]
        self.assertLess(attenuation_ratio, 0.1)  # At least 20dB attenuation

    def test_high_pass_attenuates_low_frequencies(self):
        """Test that high-pass filter attenuates low frequencies."""
        sample_rate = 10000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Create signal with low and high frequency components
        low_freq = 100  # Below cutoff
        high_freq = 4000  # Above cutoff
        signal_in = np.sin(2 * np.pi * low_freq * t) + np.sin(2 * np.pi * high_freq * t)

        # Apply high-pass filter at 500 Hz
        filt = ButterworthFilter(
            cutoff=500, order=4, filter_type="high", sample_rate=sample_rate
        )
        filtered = filt.scale_vectorized(signal_in.astype(np.float32))

        # Compute FFT
        fft_out = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(signal_in), 1 / sample_rate)

        # Find peaks
        low_idx = np.argmin(np.abs(freqs - low_freq))
        high_idx = np.argmin(np.abs(freqs - high_freq))

        # Low frequency should be attenuated
        attenuation_ratio = fft_out[low_idx] / fft_out[high_idx]
        self.assertLess(attenuation_ratio, 0.1)

    def test_band_pass_passes_mid_frequencies(self):
        """Test that band-pass filter passes frequencies in range."""
        sample_rate = 10000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Create signal with low, mid, and high frequencies
        low_freq = 50
        mid_freq = 500
        high_freq = 4000
        signal_in = (
            np.sin(2 * np.pi * low_freq * t)
            + np.sin(2 * np.pi * mid_freq * t)
            + np.sin(2 * np.pi * high_freq * t)
        )

        # Apply band-pass filter 200-1000 Hz
        filt = ButterworthFilter(
            cutoff=(200, 1000), filter_type="band", sample_rate=sample_rate
        )
        filtered = filt.scale_vectorized(signal_in.astype(np.float32))

        # Compute FFT
        fft_out = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(signal_in), 1 / sample_rate)

        # Find peaks
        low_idx = np.argmin(np.abs(freqs - low_freq))
        mid_idx = np.argmin(np.abs(freqs - mid_freq))
        high_idx = np.argmin(np.abs(freqs - high_freq))

        # Mid frequency should be strongest
        self.assertGreater(fft_out[mid_idx], fft_out[low_idx])
        self.assertGreater(fft_out[mid_idx], fft_out[high_idx])

    def test_vectorized_processing_preserves_shape(self):
        """Test that vectorized processing preserves array shape."""
        filt = ButterworthFilter()

        for n in [10, 100, 1000]:
            samples = np.random.randn(n).astype(np.float32)
            result = filt.scale_vectorized(samples)

            self.assertEqual(result.shape, samples.shape)
            self.assertEqual(result.dtype, np.float32)

    def test_empty_array_handling(self):
        """Test filter handles empty arrays gracefully."""
        filt = ButterworthFilter()
        empty = np.array([], dtype=np.float32)
        result = filt.scale_vectorized(empty)

        self.assertEqual(len(result), 0)
        self.assertEqual(result.dtype, np.float32)

    def test_single_sample_call(self):
        """Test __call__ method for single sample processing."""
        filt = ButterworthFilter(cutoff=1000)

        # Process single sample
        sample = 0.5
        result = filt(sample)

        self.assertIsInstance(result, float)

    def test_stereo_sample_call(self):
        """Test __call__ method handles stereo tuples."""
        filt = ButterworthFilter(cutoff=1000)

        # Process stereo sample
        stereo = (0.5, -0.5)
        result = filt(stereo)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestButterworthFilterProperties(unittest.TestCase):
    """Test filter property setters and getters."""

    def test_cutoff_property(self):
        """Test cutoff property getter and setter."""
        filt = ButterworthFilter(cutoff=1000)
        self.assertEqual(filt.cutoff, 1000)

        # Change cutoff
        filt.cutoff = 2000
        self.assertEqual(filt.cutoff, 2000)

    def test_order_property(self):
        """Test order property getter and setter."""
        filt = ButterworthFilter(order=4)
        self.assertEqual(filt.order, 4)

        # Change order
        filt.order = 6
        self.assertEqual(filt.order, 6)

    def test_filter_type_property(self):
        """Test filter_type property getter."""
        filt = ButterworthFilter(filter_type="high")
        self.assertEqual(filt.filter_type, "high")

    def test_changing_cutoff_redesigns_filter(self):
        """Test that changing cutoff actually changes filter behavior."""
        sample_rate = 10000
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        signal_in = np.sin(2 * np.pi * 500 * t)  # 500 Hz signal

        filt = ButterworthFilter(cutoff=1000, sample_rate=sample_rate)
        result1 = filt.scale_vectorized(signal_in.astype(np.float32))

        # Change cutoff to below signal frequency
        filt.cutoff = 300
        result2 = filt.scale_vectorized(signal_in.astype(np.float32))

        # Results should be different (second should be more attenuated)
        self.assertGreater(np.mean(np.abs(result1)), np.mean(np.abs(result2)))


class TestButterworthFilterPerformance(unittest.TestCase):
    """Test filter performance characteristics."""

    def test_vectorized_is_fast(self):
        """Test that vectorized processing is reasonably fast."""
        import time

        filt = ButterworthFilter()
        samples = np.random.randn(44100).astype(np.float32)  # 1 second at 44.1kHz

        start = time.time()
        for _ in range(10):
            _ = filt.scale_vectorized(samples)
        elapsed = time.time() - start

        # Should process 10 seconds of audio in less than 100ms
        self.assertLess(elapsed, 0.1)

    def test_higher_order_still_fast(self):
        """Test that higher order filters are still performant."""
        import time

        filt = ButterworthFilter(order=8)
        samples = np.random.randn(44100).astype(np.float32)

        start = time.time()
        _ = filt.scale_vectorized(samples)
        elapsed = time.time() - start

        # Should still be very fast
        self.assertLess(elapsed, 0.02)


class TestButterworthFilterIntegration(unittest.TestCase):
    """Test filter integration with other components."""

    def test_filter_in_chain(self):
        """Test filter can be used in a processing chain."""
        from src.engine.composer import Chain
        from src.engine.oscillator import SineOscillator

        # Create chain: Oscillator -> Filter
        osc = SineOscillator(frequency=1000, amplitude=1.0, sample_rate=10000)
        filt = ButterworthFilter(cutoff=500, sample_rate=10000)

        chain = Chain(osc, filt)

        # Generate samples
        samples = chain.get_samples(1000, mode="vectorized")

        self.assertEqual(len(samples), 1000)
        self.assertEqual(samples.dtype, np.float32)

        # Signal should be attenuated (1kHz signal through 500Hz low-pass)
        self.assertLess(np.max(np.abs(samples)), 0.5)

    def test_filter_with_noise(self):
        """Test filter can process noise."""
        from src.engine.noise import NoiseGenerator

        noise = NoiseGenerator(noise_type="White", amplitude=1.0, sample_rate=10000)
        filt = ButterworthFilter(cutoff=1000, sample_rate=10000)

        # Get noise samples
        noise_samples = noise.get_samples_vectorized(5000)

        # Filter
        filtered = filt.scale_vectorized(noise_samples)

        # Filtered noise should have less high-frequency content
        # Check via variance (filtered should be smoother)
        noise_diff = np.diff(noise_samples)
        filtered_diff = np.diff(filtered)

        self.assertLess(np.std(filtered_diff), np.std(noise_diff))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility filter functions."""

    def test_butter_function(self):
        """Test butter coefficient design function."""
        b, a = butter(order=4, cutoff=1000, fs=44100, btype="low")

        self.assertIsInstance(b, np.ndarray)
        self.assertIsInstance(a, np.ndarray)
        self.assertEqual(len(b), 5)  # Order 4 = 5 coefficients
        self.assertEqual(len(a), 5)

    def test_butter_band_pass(self):
        """Test butter function with band-pass."""
        b, a = butter(order=4, cutoff=[200, 2000], fs=44100, btype="band")

        self.assertIsInstance(b, np.ndarray)
        self.assertIsInstance(a, np.ndarray)

    def test_apply_filter_function(self):
        """Test apply_filter utility function."""
        signal_in = np.random.randn(1000)
        b, a = butter(order=4, cutoff=1000, fs=10000, btype="low")

        filtered = apply_filter(b, a, signal_in)

        self.assertEqual(len(filtered), len(signal_in))
        self.assertIsInstance(filtered, np.ndarray)


class TestButterworthFilterEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_very_low_cutoff(self):
        """Test filter with very low cutoff frequency."""
        filt = ButterworthFilter(cutoff=10, sample_rate=44100)
        samples = np.random.randn(1000).astype(np.float32)
        result = filt.scale_vectorized(samples)

        self.assertEqual(len(result), len(samples))

    def test_very_high_cutoff(self):
        """Test filter with cutoff near Nyquist."""
        filt = ButterworthFilter(cutoff=20000, sample_rate=44100)
        samples = np.random.randn(1000).astype(np.float32)
        result = filt.scale_vectorized(samples)

        self.assertEqual(len(result), len(samples))

    def test_iterator_not_implemented(self):
        """Test that __next__ raises NotImplementedError."""
        filt = ButterworthFilter()
        iter(filt)  # Initialize iterator

        with self.assertRaises(NotImplementedError):
            next(filt)

    def test_dc_offset_handling(self):
        """Test filter handles DC offset correctly."""
        filt = ButterworthFilter(cutoff=100, filter_type="high")

        # Signal with DC offset
        signal_with_dc = np.ones(1000) + np.random.randn(1000) * 0.1
        filtered = filt.scale_vectorized(signal_with_dc.astype(np.float32))

        # DC component should be removed
        self.assertLess(np.mean(filtered), 0.5)


if __name__ == "__main__":
    unittest.main()
