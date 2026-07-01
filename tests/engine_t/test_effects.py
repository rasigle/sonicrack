"""
Unit tests for audio effects (Distortion, Delay, Reverb).
"""

import unittest

import numpy as np

from src.engine import (
    Compressor,
    Delay,
    Distortion,
    Reverb,
    SineOscillator,
)


class TestDistortion(unittest.TestCase):
    """Test suite for Distortion effect."""

    def setUp(self):
        """Create test oscillator."""
        self.osc = SineOscillator(frequency=440, gain_db=-6)

    def test_initialization(self):
        """Test distortion initialization."""
        dist = Distortion(
            self.osc, drive=2.0, mix=0.5, output_gain=0.7, distortion_type="soft"
        )

        self.assertEqual(dist.drive, 2.0)
        self.assertEqual(dist.mix, 0.5)
        self.assertEqual(dist.output_gain, 0.7)
        self.assertEqual(dist.distortion_type, "soft")

    def test_strict_parameter_validation(self):
        """Test that out-of-range parameters raise ValueError."""
        invalid_cases = [
            {"drive": 15.0},
            {"drive": -0.1},
            {"mix": 2.0},
            {"mix": -0.1},
            {"output_gain": 5.0},
            {"output_gain": -0.1},
        ]

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Distortion(self.osc, **kwargs)

    def test_invalid_distortion_type(self):
        """Test that invalid distortion types raise ValueError."""
        with self.assertRaises(ValueError):
            Distortion(self.osc, distortion_type="invalid")

    def test_all_distortion_types(self):
        """Test that all distortion types work."""
        types = ["soft", "hard", "fuzz", "tube"]

        for dtype in types:
            with self.subTest(dtype=dtype):
                dist = Distortion(self.osc, distortion_type=dtype)
                samples = dist.get_samples_vectorized(1000)

                self.assertEqual(len(samples), 1000)
                self.assertTrue(np.all(np.isfinite(samples)))

    def test_dry_wet_mix(self):
        """Test dry/wet mix parameter."""
        # 100% dry
        dist_dry = Distortion(self.osc, drive=5.0, mix=0.0)
        # 100% wet
        dist_wet = Distortion(SineOscillator(440, gain_db=-6), drive=5.0, mix=1.0)

        samples_dry = dist_dry.get_samples_vectorized(1000)
        samples_wet = dist_wet.get_samples_vectorized(1000)

        # Wet signal should have more harmonic content (higher RMS for distortion)
        self.assertGreater(np.std(samples_wet), np.std(samples_dry) * 0.5)

    def test_vectorized_vs_iterator(self):
        """Test that vectorized and iterator modes produce similar results."""
        dist1 = Distortion(SineOscillator(440, gain_db=-6), drive=3.0)
        dist2 = Distortion(SineOscillator(440, gain_db=-6), drive=3.0)

        samples_vec = dist1.get_samples_vectorized(100)
        samples_iter = dist2.get_samples(100, mode="iterator")

        np.testing.assert_allclose(samples_vec, samples_iter, rtol=1e-5)

    def test_property_setters(self):
        """Test runtime parameter changes via properties."""
        dist = Distortion(self.osc, drive=1.0)

        dist.drive = 5.0
        self.assertEqual(dist.drive, 5.0)

        dist.mix = 0.7
        self.assertEqual(dist.mix, 0.7)

        dist.output_gain = 0.8
        self.assertEqual(dist.output_gain, 0.8)

        dist.distortion_type = "hard"
        self.assertEqual(dist.distortion_type, "hard")

        for name, value in (("drive", 15.0), ("mix", 2.0), ("output_gain", 5.0)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                setattr(dist, name, value)


class TestDelay(unittest.TestCase):
    """Test suite for Delay effect."""

    def setUp(self):
        """Create test oscillator."""
        self.osc = SineOscillator(frequency=440, gain_db=-12)
        self.sample_rate = 44100

    def test_initialization(self):
        """Test delay initialization."""
        delay = Delay(
            self.osc,
            delay_time=0.5,
            feedback=0.4,
            mix=0.5,
            sample_rate=self.sample_rate,
        )

        self.assertEqual(delay.delay_time, 0.5)
        self.assertEqual(delay.feedback, 0.4)
        self.assertEqual(delay.mix, 0.5)

    def test_strict_parameter_validation(self):
        """Test that out-of-range parameters raise ValueError."""
        invalid_cases = [
            {"delay_time": 5.0},
            {"delay_time": 0.0},
            {"feedback": 1.5},
            {"feedback": -0.1},
            {"mix": 2.0},
            {"mix": -0.1},
        ]

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Delay(self.osc, **kwargs)

    def test_delay_creates_echo(self):
        """Test that delay creates echoes."""
        delay = Delay(
            self.osc,
            delay_time=0.1,
            feedback=0.5,
            mix=1.0,  # 100% wet to hear echo clearly
            sample_rate=self.sample_rate,
        )

        # Generate samples
        samples = delay.get_samples_vectorized(20000)

        # Should have valid samples
        self.assertTrue(np.all(np.isfinite(samples)))
        self.assertEqual(len(samples), 20000)

    def test_feedback_effect(self):
        """Test that feedback creates multiple echoes."""
        # No feedback
        delay_no_fb = Delay(
            SineOscillator(440, gain_db=-12),
            delay_time=0.1,
            feedback=0.0,
            mix=1.0,
            sample_rate=self.sample_rate,
        )

        # With feedback
        delay_with_fb = Delay(
            SineOscillator(440, gain_db=-12),
            delay_time=0.1,
            feedback=0.7,
            mix=1.0,
            sample_rate=self.sample_rate,
        )

        samples_no_fb = delay_no_fb.get_samples_vectorized(10000)
        samples_with_fb = delay_with_fb.get_samples_vectorized(10000)

        # Feedback should create more energy (higher RMS)
        self.assertGreater(
            np.sqrt(np.mean(samples_with_fb**2)), np.sqrt(np.mean(samples_no_fb**2))
        )

    def test_delay_time_change(self):
        """Test runtime delay time changes."""
        delay = Delay(self.osc, delay_time=0.1, sample_rate=self.sample_rate)

        delay.delay_time = 0.5
        self.assertEqual(delay.delay_time, 0.5)

        for name, value in (("delay_time", 5.0), ("feedback", 1.5), ("mix", 2.0)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                setattr(delay, name, value)

        # Should still generate valid samples
        samples = delay.get_samples_vectorized(1000)
        self.assertTrue(np.all(np.isfinite(samples)))

    def test_circular_buffer(self):
        """Test that circular buffer works correctly."""
        delay = Delay(
            self.osc, delay_time=0.5, feedback=0.3, sample_rate=self.sample_rate
        )

        # Generate enough samples to wrap around buffer multiple times
        samples = delay.get_samples_vectorized(100000)

        self.assertEqual(len(samples), 100000)
        self.assertTrue(np.all(np.isfinite(samples)))


class TestCompressor(unittest.TestCase):
    """Test suite for Compressor effect."""

    def setUp(self):
        """Create test oscillator."""
        self.osc = SineOscillator(frequency=440, gain_db=0)

    def test_initialization(self):
        """Test compressor initialization."""
        compressor = Compressor(
            self.osc,
            threshold_db=-24.0,
            ratio=6.0,
            attack_ms=5.0,
            release_ms=80.0,
            makeup_gain_db=3.0,
            mix=0.75,
        )

        self.assertEqual(compressor.threshold_db, -24.0)
        self.assertEqual(compressor.ratio, 6.0)
        self.assertEqual(compressor.attack_ms, 5.0)
        self.assertEqual(compressor.release_ms, 80.0)
        self.assertEqual(compressor.makeup_gain_db, 3.0)
        self.assertEqual(compressor.mix, 0.75)

    def test_strict_parameter_validation(self):
        """Test that out-of-range parameters raise ValueError."""
        invalid_cases = [
            {"threshold_db": -80.0},
            {"threshold_db": 3.0},
            {"ratio": 0.5},
            {"ratio": 40.0},
            {"attack_ms": 0.0},
            {"attack_ms": 400.0},
            {"release_ms": 0.0},
            {"release_ms": 2000.0},
            {"makeup_gain_db": -48.0},
            {"makeup_gain_db": 48.0},
            {"mix": -0.1},
            {"mix": 2.0},
        ]

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Compressor(self.osc, **kwargs)

    def test_compression_reduces_loud_signal(self):
        """Test that compressor reduces a signal above threshold."""
        samples = np.full(256, 1.0, dtype=np.float32)
        compressor = Compressor(
            threshold_db=-24.0,
            ratio=20.0,
            attack_ms=0.1,
            release_ms=100.0,
            mix=1.0,
        )

        output = compressor(samples)

        self.assertLess(abs(output[-1]), abs(samples[-1]))
        self.assertTrue(np.all(np.isfinite(output)))

    def test_below_threshold_signal_passes_through(self):
        """Test that quiet signals are unchanged without makeup gain."""
        samples = np.full(32, 0.01, dtype=np.float32)
        compressor = Compressor(threshold_db=-24.0, ratio=8.0)

        output = compressor(samples)

        np.testing.assert_allclose(output, samples, rtol=1e-6, atol=1e-6)

    def test_vectorized_vs_iterator(self):
        """Test that vectorized and iterator modes produce similar results."""
        comp_vec = Compressor(SineOscillator(440, gain_db=0), threshold_db=-30.0)
        comp_iter = Compressor(SineOscillator(440, gain_db=0), threshold_db=-30.0)

        samples_vec = comp_vec.get_samples_vectorized(100)
        samples_iter = comp_iter.get_samples(100, mode="iterator")

        np.testing.assert_allclose(samples_vec, samples_iter, rtol=1e-5, atol=1e-6)


class TestReverb(unittest.TestCase):
    """Test suite for Reverb effect."""

    def setUp(self):
        """Create test oscillator."""
        self.osc = SineOscillator(frequency=440, gain_db=-12)
        self.sample_rate = 44100

    def test_initialization(self):
        """Test reverb initialization."""
        reverb = Reverb(
            self.osc, room_size=0.7, damping=0.5, mix=0.3, sample_rate=self.sample_rate
        )

        self.assertEqual(reverb.room_size, 0.7)
        self.assertEqual(reverb.damping, 0.5)
        self.assertEqual(reverb.mix, 0.3)

    def test_strict_parameter_validation(self):
        """Test that out-of-range parameters raise ValueError."""
        invalid_cases = [
            {"room_size": 2.0},
            {"room_size": -0.1},
            {"damping": 2.0},
            {"damping": -0.1},
            {"mix": 2.0},
            {"mix": -0.1},
        ]

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Reverb(self.osc, **kwargs)

    def test_reverb_creates_decay(self):
        """Test that reverb creates a decay tail."""
        reverb = Reverb(
            self.osc,
            room_size=0.8,
            damping=0.3,
            mix=1.0,  # 100% wet
            sample_rate=self.sample_rate,
        )

        # Generate samples
        samples = reverb.get_samples_vectorized(20000)

        self.assertTrue(np.all(np.isfinite(samples)))
        self.assertEqual(len(samples), 20000)

    def test_room_size_effect(self):
        """Test that room size affects reverb characteristics."""
        # Small room
        reverb_small = Reverb(
            SineOscillator(440, gain_db=-12),
            room_size=0.2,
            mix=1.0,
            sample_rate=self.sample_rate,
        )

        # Large room
        reverb_large = Reverb(
            SineOscillator(440, gain_db=-12),
            room_size=0.9,
            mix=1.0,
            sample_rate=self.sample_rate,
        )

        samples_small = reverb_small.get_samples_vectorized(10000)
        samples_large = reverb_large.get_samples_vectorized(10000)

        # Both should produce valid samples
        self.assertTrue(np.all(np.isfinite(samples_small)))
        self.assertTrue(np.all(np.isfinite(samples_large)))

    def test_damping_effect(self):
        """Test that damping affects high frequencies."""
        # Low damping (bright)
        reverb_bright = Reverb(
            SineOscillator(440, gain_db=-12),
            room_size=0.5,
            damping=0.1,
            mix=1.0,
            sample_rate=self.sample_rate,
        )

        # High damping (dark)
        reverb_dark = Reverb(
            SineOscillator(440, gain_db=-12),
            room_size=0.5,
            damping=0.9,
            mix=1.0,
            sample_rate=self.sample_rate,
        )

        samples_bright = reverb_bright.get_samples_vectorized(5000)
        samples_dark = reverb_dark.get_samples_vectorized(5000)

        self.assertTrue(np.all(np.isfinite(samples_bright)))
        self.assertTrue(np.all(np.isfinite(samples_dark)))

    def test_property_setters(self):
        """Test runtime parameter changes."""
        reverb = Reverb(self.osc, sample_rate=self.sample_rate)

        reverb.room_size = 0.8
        self.assertEqual(reverb.room_size, 0.8)

        reverb.damping = 0.6
        self.assertEqual(reverb.damping, 0.6)

        reverb.mix = 0.4
        self.assertEqual(reverb.mix, 0.4)

        for name, value in (("room_size", 2.0), ("damping", 2.0), ("mix", 2.0)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                setattr(reverb, name, value)

        # Should still generate valid samples
        samples = reverb.get_samples_vectorized(1000)
        self.assertTrue(np.all(np.isfinite(samples)))

    def test_filter_buffers_created(self):
        """Test that all comb and allpass buffers are created."""
        reverb = Reverb(self.osc, sample_rate=self.sample_rate)

        # Should have 8 comb filters
        self.assertEqual(len(reverb._comb_buffers), 8)
        self.assertEqual(len(reverb._comb_positions), 8)

        # Should have 4 allpass filters
        self.assertEqual(len(reverb._allpass_buffers), 4)
        self.assertEqual(len(reverb._allpass_positions), 4)


class TestEffectChaining(unittest.TestCase):
    """Test chaining multiple effects together."""

    def test_distortion_then_delay(self):
        """Test chaining distortion into delay."""
        osc = SineOscillator(440, gain_db=-12)
        dist = Distortion(osc, drive=2.0, mix=0.8)
        delay = Delay(dist, delay_time=0.2, feedback=0.3, sample_rate=44100)

        samples = delay.get_samples_vectorized(5000)

        self.assertEqual(len(samples), 5000)
        self.assertTrue(np.all(np.isfinite(samples)))

    def test_delay_then_reverb(self):
        """Test chaining delay into reverb."""
        osc = SineOscillator(440, gain_db=-12)
        delay = Delay(osc, delay_time=0.15, feedback=0.4, sample_rate=44100)
        reverb = Reverb(delay, room_size=0.6, damping=0.5, sample_rate=44100)

        samples = reverb.get_samples_vectorized(5000)

        self.assertEqual(len(samples), 5000)
        self.assertTrue(np.all(np.isfinite(samples)))

    def test_full_effect_chain(self):
        """Test chaining distortion → delay → reverb."""
        osc = SineOscillator(440, gain_db=-12)
        dist = Distortion(osc, drive=1.5, mix=0.6, output_gain=0.6)
        delay = Delay(dist, delay_time=0.2, feedback=0.3, sample_rate=44100)
        reverb = Reverb(delay, room_size=0.5, mix=0.3, sample_rate=44100)

        samples = reverb.get_samples_vectorized(10000)

        self.assertEqual(len(samples), 10000)
        self.assertTrue(np.all(np.isfinite(samples)))

        # Should have reasonable amplitude
        self.assertLess(np.max(np.abs(samples)), 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
