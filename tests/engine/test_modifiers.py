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
        panner = Panner(0.5)
        self.assertEqual(panner.position, 0.5)

    def test_center_pan(self) -> None:
        """Test center panning produces equal left/right."""
        panner = Panner(0.5)
        result = panner(1.0)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        left, right = result
        self.assertAlmostEqual(left, 1.0, places=5)
        self.assertAlmostEqual(right, 1.0, places=5)

    def test_hard_left(self) -> None:
        """Test hard left panning."""
        panner = Panner(0.0)
        left, right = panner(1.0)

        self.assertAlmostEqual(left, 2.0, places=5)
        self.assertAlmostEqual(right, 0.0, places=5)

    def test_hard_right(self) -> None:
        """Test hard right panning."""
        panner = Panner(1.0)
        left, right = panner(1.0)

        self.assertAlmostEqual(left, 0.0, places=5)
        self.assertAlmostEqual(right, 2.0, places=5)

    def test_negative_values(self) -> None:
        """Test panner works with negative input."""
        panner = Panner(0.5)
        left, right = panner(-1.0)

        self.assertAlmostEqual(left, -1.0, places=5)
        self.assertAlmostEqual(right, -1.0, places=5)


class TestModulatedPanner(unittest.TestCase):
    """Test suite for ModulatedPanner."""

    def test_initialization(self) -> None:
        """Test modulated panner initializes correctly."""
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1)
        panner = ModulatedPanner(env)
        self.assertIsNotNone(panner.modulator)

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
        self.assertGreater(len(set(left_values)), 1)


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
        panner = Panner(0.5)

        # Apply volume, then pan
        after_volume = volume(1.0)
        after_pan = panner(after_volume)

        self.assertIsInstance(after_pan, tuple)
        left, right = after_pan
        self.assertAlmostEqual(left, 0.5, places=5)
        self.assertAlmostEqual(right, 0.5, places=5)

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

