"""Unit tests for modifiers (Panner, Volume, Clipper, etc.)."""

import unittest
import numpy as np
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.engine.modifier import (
    Modifier,
    Panner,
    ModulatedPanner,
    Volume,
    ModulatedVolume,
    Frequency,
    ModulatedFrequency,
    Clipper,
)
from src.engine.oscillator import SineOscillator
from src.engine.modulator import ADSREnvelope


class TestPanner(unittest.TestCase):
    """Test suite for Panner modifier."""

    def test_initialization(self) -> None:
        """Test panner initializes with correct pan position."""
        panner = Panner(0.0)
        self.assertEqual(panner.position, 0.0)

    def test_initialization_with_custom_position(self) -> None:
        """Test panner initializes with custom position."""
        panner = Panner(-0.5)
        self.assertEqual(panner.position, -0.5)

        panner = Panner(0.7)
        self.assertEqual(panner.position, 0.7)

    def test_initialization_clips_out_of_range(self) -> None:
        """Test panner clips position to valid range."""
        panner = Panner(-2.0)
        self.assertEqual(panner.position, -1.0)

        panner = Panner(2.0)
        self.assertEqual(panner.position, 1.0)

    def test_center_pan(self) -> None:
        """Test center panning produces equal left/right with constant power."""
        panner = Panner(0.0)
        result = panner(1.0)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        left, right = result

        # At center, both should be ~0.707 (1/sqrt(2)) for constant power
        self.assertAlmostEqual(left, np.sqrt(0.5), places=5)
        self.assertAlmostEqual(right, np.sqrt(0.5), places=5)

        # Total power should be preserved
        power = left**2 + right**2
        self.assertAlmostEqual(power, 1.0, places=5)

    def test_hard_left(self) -> None:
        """Test hard left panning."""
        panner = Panner(-1.0)
        left, right = panner(1.0)

        self.assertAlmostEqual(left, 1.0, places=5)
        self.assertAlmostEqual(right, 0.0, places=5)

    def test_hard_right(self) -> None:
        """Test hard right panning."""
        panner = Panner(1.0)
        left, right = panner(1.0)

        self.assertAlmostEqual(left, 0.0, places=5)
        self.assertAlmostEqual(right, 1.0, places=5)

    def test_partial_left(self) -> None:
        """Test partial left panning."""
        panner = Panner(-0.5)
        left, right = panner(1.0)

        # Left should be stronger than right
        self.assertGreater(left, right)

        # Power should be preserved
        power = left**2 + right**2
        self.assertAlmostEqual(power, 1.0, places=5)

    def test_partial_right(self) -> None:
        """Test partial right panning."""
        panner = Panner(0.5)
        left, right = panner(1.0)

        # Right should be stronger than left
        self.assertGreater(right, left)

        # Power should be preserved
        power = left**2 + right**2
        self.assertAlmostEqual(power, 1.0, places=5)

    def test_negative_values(self) -> None:
        """Test panner works with negative input."""
        panner = Panner(0.0)
        left, right = panner(-1.0)

        # Signs should be preserved
        self.assertLess(left, 0)
        self.assertLess(right, 0)

        # Magnitudes should match positive case
        self.assertAlmostEqual(abs(left), np.sqrt(0.5), places=5)
        self.assertAlmostEqual(abs(right), np.sqrt(0.5), places=5)

    def test_constant_power_law(self) -> None:
        """Test that constant power law is maintained across pan positions."""
        positions = np.linspace(-1.0, 1.0, 21)

        for pos in positions:
            panner = Panner(pos)
            left, right = panner(1.0)

            # Total power should always be 1.0
            power = left**2 + right**2
            self.assertAlmostEqual(
                power, 1.0, places=5,
                msg=f"Power not preserved at position {pos}: {power}"
            )

    def test_vectorized_panning(self) -> None:
        """Test vectorized panning with numpy arrays."""
        panner = Panner(0.0)
        samples = np.array([0.5, -0.5, 1.0, -1.0, 0.0])

        left, right = panner(samples)

        # Should return numpy arrays
        self.assertIsInstance(left, np.ndarray)
        self.assertIsInstance(right, np.ndarray)
        self.assertEqual(len(left), len(samples))
        self.assertEqual(len(right), len(samples))

        # Check each sample
        for i in range(len(samples)):
            expected_left = samples[i] * np.sqrt(0.5)
            expected_right = samples[i] * np.sqrt(0.5)
            self.assertAlmostEqual(left[i], expected_left, places=5)
            self.assertAlmostEqual(right[i], expected_right, places=5)

    def test_pan_vectorized_method(self) -> None:
        """Test the explicit pan_vectorized method."""
        panner = Panner(-0.5)
        samples = np.array([1.0, 0.5, -0.5, -1.0])

        left, right = panner.pan_vectorized(samples)

        # Should match calling panner directly
        left_direct, right_direct = panner(samples)
        np.testing.assert_array_almost_equal(left, left_direct)
        np.testing.assert_array_almost_equal(right, right_direct)

    def test_zero_input(self) -> None:
        """Test panning with zero input."""
        panner = Panner(0.5)
        left, right = panner(0.0)

        self.assertEqual(left, 0.0)
        self.assertEqual(right, 0.0)

    def test_large_input(self) -> None:
        """Test panning with large input values."""
        panner = Panner(0.0)
        left, right = panner(100.0)

        # Power should still be preserved relative to input
        expected_power = 100.0**2
        actual_power = left**2 + right**2
        self.assertAlmostEqual(actual_power, expected_power, places=3)


class TestModulatedPanner(unittest.TestCase):
    """Test suite for ModulatedPanner."""

    def test_initialization(self) -> None:
        """Test modulated panner initializes correctly."""
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1)
        panner = ModulatedPanner(env)
        self.assertIsNotNone(panner.modulator)
        self.assertEqual(panner.position, 0.0)

    def test_modulation_changes_pan(self) -> None:
        """Test that modulation affects pan position."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=100)
        panner = ModulatedPanner(env)

        # Get several values
        results = []
        for _ in range(10):
            result = panner(1.0)
            results.append(result)
            next(panner)

        # Pan should change over time due to modulation
        left_values = [r[0] for r in results]
        self.assertGreater(len(set(left_values)), 1,
                          "Pan values should change over time")

    def test_modulation_position_mapping(self) -> None:
        """Test that modulator values [-1, 1] map directly to pan [-1, 1]."""
        # Create a simple modulator that yields known values
        class SimpleModulator:
            def __init__(self):
                self.values = [-1.0, -0.5, 0.0, 0.5, 1.0]
                self.index = 0

            def __iter__(self):
                self.index = 0
                return self

            def __next__(self):
                if self.index >= len(self.values):
                    self.index = 0
                val = self.values[self.index]
                self.index += 1
                return val

        mod = SimpleModulator()
        panner = ModulatedPanner(mod)

        # Expected: direct mapping (no transformation)
        expected_positions = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for expected_pos in expected_positions:
            next(panner)
            self.assertAlmostEqual(panner.position, expected_pos, places=5)

    def test_iteration_resets_modulator(self) -> None:
        """Test that calling iter() resets the modulator."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=100)
        panner = ModulatedPanner(env)

        # Advance the modulator
        for _ in range(5):
            next(panner)

        # Reset
        iter(panner)

        # Position should be reset
        # (exact value depends on ADSR implementation)
        self.assertIsNotNone(panner.position)

    def test_constant_power_during_modulation(self) -> None:
        """Test that constant power is maintained during modulation."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=100)
        panner = ModulatedPanner(env)

        for _ in range(20):
            left, right = panner(1.0)
            power = left**2 + right**2
            self.assertAlmostEqual(
                power, 1.0, places=5,
                msg=f"Power not preserved during modulation at position {panner.position}"
            )
            next(panner)

    def test_vectorized_modulated_panning(self) -> None:
        """Test vectorized modulated panning."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=100)
        panner = ModulatedPanner(env)

        num_samples = 20
        samples = np.ones(num_samples)

        left, right = panner.pan_vectorized(samples, num_samples)

        # Should return numpy arrays
        self.assertIsInstance(left, np.ndarray)
        self.assertIsInstance(right, np.ndarray)
        self.assertEqual(len(left), num_samples)
        self.assertEqual(len(right), num_samples)

        # Values should change over time
        self.assertGreater(len(set(left)), 1,
                          "Left channel should vary over time")
        self.assertGreater(len(set(right)), 1,
                          "Right channel should vary over time")

    def test_vectorized_preserves_power(self) -> None:
        """Test that vectorized modulated panning preserves power."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=100)
        panner = ModulatedPanner(env)

        num_samples = 50
        samples = np.ones(num_samples)

        left, right = panner.pan_vectorized(samples, num_samples)

        # Check power preservation for each sample
        for i in range(num_samples):
            power = left[i]**2 + right[i]**2
            self.assertAlmostEqual(
                power, 1.0, places=5,
                msg=f"Power not preserved at sample {i}"
            )

    def test_negative_input_with_modulation(self) -> None:
        """Test modulated panner works with negative input."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=100)
        panner = ModulatedPanner(env)

        for _ in range(10):
            left, right = panner(-1.0)
            # Signs should be preserved
            self.assertLessEqual(left, 0)
            self.assertLessEqual(right, 0)
            next(panner)

    def test_varying_input_amplitude(self) -> None:
        """Test modulated panner with varying input amplitudes."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=100)
        panner = ModulatedPanner(env)

        test_values = [0.0, 0.25, 0.5, 0.75, 1.0, -0.5, -1.0]

        for val in test_values:
            left, right = panner(val)

            # Power should be proportional to input squared
            expected_power = val**2
            actual_power = left**2 + right**2
            self.assertAlmostEqual(
                actual_power, expected_power, places=5,
                msg=f"Power not correct for input {val}"
            )
            next(panner)

    def test_modulation_range_clamping(self) -> None:
        """Test that modulation values are clamped to valid range."""
        # Create a modulator that produces out-of-range values
        class OutOfRangeModulator:
            def __init__(self):
                self.values = [-1.5, 1.5, 0.0]  # Out of [-1, 1] range
                self.index = 0

            def __iter__(self):
                self.index = 0
                return self

            def __next__(self):
                val = self.values[self.index % len(self.values)]
                self.index += 1
                return val

        mod = OutOfRangeModulator()
        panner = ModulatedPanner(mod)

        # Position should be clamped to [-1, 1]
        next(panner)  # -1.5 -> clamped to -1.0
        self.assertEqual(panner.position, -1.0)

        next(panner)  # 1.5 -> clamped to 1.0
        self.assertEqual(panner.position, 1.0)

        next(panner)  # 0.0 -> stays 0.0
        self.assertEqual(panner.position, 0.0)


class TestVolume(unittest.TestCase):
    """Test suite for Volume modifier."""

    def test_initialization(self) -> None:
        """Test volume initializes with correct amplitude."""
        volume = Volume(0.5)
        self.assertEqual(volume.amplitude, 0.5)

    def test_volume_scaling(self) -> None:
        """Test volume correctly scales input."""
        volume = Volume(0.5)
        result = volume(1.0)
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_zero_volume(self) -> None:
        """Test zero volume produces silence."""
        volume = Volume(0.0)
        result = volume(1.0)
        self.assertEqual(result, 0.0)

    def test_amplification(self) -> None:
        """Test volume can amplify (>1.0)."""
        volume = Volume(2.0)
        result = volume(1.0)
        self.assertAlmostEqual(result, 2.0, places=5)

    def test_stereo_input(self) -> None:
        """Test volume works with stereo input."""
        volume = Volume(0.5)
        result = volume((1.0, 1.0))

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 0.5, places=5)
        self.assertAlmostEqual(result[1], 0.5, places=5)


class TestModulatedVolume(unittest.TestCase):
    """Test suite for ModulatedVolume."""

    def test_initialization(self) -> None:
        """Test modulated volume initializes correctly."""
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1)
        volume = ModulatedVolume(env)
        self.assertIsNotNone(volume.modulator)

    def test_modulation_changes_volume(self) -> None:
        """Test that modulation affects volume."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=100)
        volume = ModulatedVolume(env)
        iter(volume)

        # Get several values
        results = []
        for _ in range(10):
            result = volume(1.0)
            results.append(result)
            next(volume)

        # Volume should change over time
        self.assertGreater(len(set(results)), 1)


class TestFrequency(unittest.TestCase):
    """Test suite for Frequency modifier."""

    def test_initialization(self) -> None:
        """Test frequency modifier initializes correctly."""
        freq_mod = Frequency(2.0)
        self.assertEqual(freq_mod.frequency, 2.0)

    def test_frequency_scaling(self) -> None:
        """Test frequency modifier scales value."""
        freq_mod = Frequency(2.0)
        result = freq_mod(440.0)
        self.assertAlmostEqual(result, 880.0, places=5)


class TestClipper(unittest.TestCase):
    """Test suite for Clipper modifier."""

    def test_initialization(self) -> None:
        """Test clipper initializes with correct range."""
        clipper = Clipper((-0.5, 0.5))
        self.assertEqual(clipper.range, (-0.5, 0.5))

    def test_no_clipping_within_range(self) -> None:
        """Test clipper doesn't modify values within range."""
        clipper = Clipper((-1.0, 1.0))
        self.assertEqual(clipper(0.5), 0.5)
        self.assertEqual(clipper(-0.5), -0.5)

    def test_clips_above_max(self) -> None:
        """Test clipper clips values above maximum."""
        clipper = Clipper((-1.0, 1.0))
        self.assertEqual(clipper(2.0), 1.0)
        self.assertEqual(clipper(1.5), 1.0)

    def test_clips_below_min(self) -> None:
        """Test clipper clips values below minimum."""
        clipper = Clipper((-1.0, 1.0))
        self.assertEqual(clipper(-2.0), -1.0)
        self.assertEqual(clipper(-1.5), -1.0)

    def test_asymmetric_range(self) -> None:
        """Test clipper works with asymmetric range."""
        clipper = Clipper((-0.3, 0.7))
        self.assertEqual(clipper(-1.0), -0.3)
        self.assertEqual(clipper(1.0), 0.7)
        self.assertEqual(clipper(0.0), 0.0)

    def test_stereo_clipping(self) -> None:
        """Test clipper works with stereo input."""
        clipper = Clipper((-0.4, 0.6))

        result = clipper((1.0, -1.0))
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], 0.6)
        self.assertEqual(result[1], -0.4)

        result = clipper((-1.0, 1.0))
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], -0.4)
        self.assertEqual(result[1], 0.6)


class TestModifierIntegration(unittest.TestCase):
    """Test modifiers work together in chains."""

    def test_volume_then_pan(self) -> None:
        """Test volume followed by panning."""
        volume = Volume(0.5)
        panner = Panner(0.5)  # Right-biased pan

        # Apply volume, then pan
        after_volume = volume(1.0)
        after_pan = panner(after_volume)

        self.assertIsInstance(after_pan, tuple)
        left, right = after_pan

        # With constant-power panning at position 0.5 (right-biased)
        # and input 0.5 (after volume), right should be greater than left
        self.assertGreater(right, left)

        # Power should be preserved
        expected_power = after_volume**2
        actual_power = left**2 + right**2
        self.assertAlmostEqual(actual_power, expected_power, places=5)

    def test_clip_then_volume(self) -> None:
        """Test clipping followed by volume."""
        clipper = Clipper((-0.5, 0.5))
        volume = Volume(2.0)

        # Clip first, then amplify
        after_clip = clipper(1.0)
        after_volume = volume(after_clip)

        self.assertEqual(after_clip, 0.5)
        self.assertAlmostEqual(after_volume, 1.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

