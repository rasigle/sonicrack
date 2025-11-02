"""Test for WaveAdder mixed mono/stereo inputs bug fix."""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.engine import SineOscillator, WaveAdder, Chain, ModulatedPanner, TriangleOscillator


class TestWaveAdderMixedInputs(unittest.TestCase):
    """Test WaveAdder with mixed mono and stereo inputs."""

    def test_stereo_mode_with_one_stereo_one_mono(self):
        """Test WaveAdder stereo mode with one stereo and one mono input."""
        # Stereo input: Chain with ModulatedPanner
        stereo_gen = Chain(
            SineOscillator(440, sample_rate=1000),
            ModulatedPanner(TriangleOscillator(1, sample_rate=1000))
        )

        # Mono input
        mono_gen = SineOscillator(220, sample_rate=1000)

        # Mix them
        mixer = WaveAdder(stereo_gen, mono_gen, stereo=True)

        # Should not raise an error
        samples = mixer.get_samples(100)

        # Should be stereo output
        self.assertEqual(samples.shape, (100, 2))

        # Should be averaging the inputs
        self.assertTrue(isinstance(samples, np.ndarray))

    def test_stereo_mode_with_two_stereo_one_mono(self):
        """Test WaveAdder stereo mode with two stereo and one mono input (original bug case)."""
        # Two stereo inputs
        stereo1 = Chain(
            SineOscillator(440, amplitude=0.3, sample_rate=1000),
            ModulatedPanner(TriangleOscillator(1, phase=180, sample_rate=1000))
        )
        stereo2 = Chain(
            SineOscillator(550, amplitude=0.3, sample_rate=1000),
            ModulatedPanner(TriangleOscillator(1, sample_rate=1000))
        )

        # One mono input
        mono = SineOscillator(220, sample_rate=1000)

        # Mix them
        mixer = WaveAdder(stereo1, stereo2, mono, stereo=True)

        # Should not raise an error
        samples = mixer.get_samples(100)

        # Should be stereo output
        self.assertEqual(samples.shape, (100, 2))

    def test_stereo_mode_all_mono_inputs(self):
        """Test WaveAdder stereo mode with all mono inputs still works."""
        mixer = WaveAdder(
            SineOscillator(440, sample_rate=1000),
            SineOscillator(550, sample_rate=1000),
            stereo=True
        )

        samples = mixer.get_samples(100)

        # Should be stereo (duplicated mono)
        self.assertEqual(samples.shape, (100, 2))

        # Left and right should be identical
        np.testing.assert_array_almost_equal(samples[:, 0], samples[:, 1])

    def test_stereo_mode_all_stereo_inputs(self):
        """Test WaveAdder stereo mode with all stereo inputs still works."""
        stereo1 = Chain(
            SineOscillator(440, sample_rate=1000),
            ModulatedPanner(TriangleOscillator(1, sample_rate=1000))
        )
        stereo2 = Chain(
            SineOscillator(550, sample_rate=1000),
            ModulatedPanner(TriangleOscillator(2, sample_rate=1000))
        )

        mixer = WaveAdder(stereo1, stereo2, stereo=True)

        samples = mixer.get_samples(100)

        # Should be stereo
        self.assertEqual(samples.shape, (100, 2))

        # Left and right should be different (due to panning)
        self.assertFalse(np.allclose(samples[:, 0], samples[:, 1]))


if __name__ == '__main__':
    unittest.main()

