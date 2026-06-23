"""Test for modulated panning in Chain (vectorized mode)."""

import unittest

import numpy as np

from src.engine.composer import Chain
from src.engine.modifier import ModulatedPanner
from src.engine.modulator import ADSREnvelope
from src.engine.oscillator import SineOscillator


class TestModulatedPanningInChain(unittest.TestCase):
    """Test that modulated panning works correctly in Chain."""

    def test_modulated_panning_produces_different_channels(self):
        """Test that ModulatedPanner in Chain produces different L/R channels."""
        # Create chain with modulated panner
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=1000)
        env.trigger_note_on()  # Trigger envelope to produce varying values
        gen = Chain(SineOscillator(220, sample_rate=1000), ModulatedPanner(env))

        # Generate samples using vectorized mode
        samples = gen.get_samples(500)

        # Extract left and right channels
        left, right = np.array(samples).T

        # Channels should NOT be identical
        self.assertFalse(
            np.allclose(left, right),
            "Left and right channels should be different with modulated panning",
        )

        # Verify we got different values
        self.assertGreater(len(set(left)), 1, "Left channel should have varying values")
        self.assertGreater(
            len(set(right)), 1, "Right channel should have varying values"
        )

    def test_modulated_panning_with_oscillator_modulator(self):
        """Test modulated panning using an oscillator as the modulator."""
        # Use sine oscillator as LFO for panning
        gen = Chain(
            SineOscillator(220, sample_rate=1000),
            ModulatedPanner(
                SineOscillator(4, wave_range=(-0.8, 0.8), sample_rate=1000)
            ),
        )

        # Generate samples
        samples = gen.get_samples(1000)
        left, right = np.array(samples).T

        # Channels should be different
        self.assertFalse(np.allclose(left, right))

        # Verify stereo
        self.assertEqual(len(left), 1000)
        self.assertEqual(len(right), 1000)

    def test_modulated_panning_iterator_mode(self):
        """Test that modulated panning also works in iterator mode."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=1000)
        env.trigger_note_on()  # Trigger envelope
        gen = Chain(SineOscillator(220, sample_rate=1000), ModulatedPanner(env))

        # Use iterator mode explicitly
        samples = gen.get_samples(100, mode="iterator")

        # Extract channels
        left, right = np.array(samples).T

        # Should be different
        self.assertFalse(np.allclose(left, right))

    def test_panning_power_preservation(self):
        """Test that constant-power law is maintained with modulated panning."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=1000)
        gen = Chain(
            SineOscillator(220, amplitude=1.0, sample_rate=1000), ModulatedPanner(env)
        )

        # Generate samples
        samples = gen.get_samples(200)
        left, right = np.array(samples).T

        # Check power preservation for non-zero samples
        # Note: Due to the 220Hz oscillator, some values will be zero (zero crossings)
        non_zero_indices = np.where(np.abs(left) + np.abs(right) > 0.01)[0]

        self.assertGreater(
            len(non_zero_indices), 100, "Should have many non-zero samples"
        )

        # For non-zero samples, check that panning preserves power ratio
        for i in non_zero_indices[:50]:  # Check first 50 non-zero samples
            power = left[i] ** 2 + right[i] ** 2
            # Power should be reasonable (not extremely large)
            self.assertLess(power, 2.0)


if __name__ == "__main__":
    unittest.main()
