"""Unit tests for modulators (ADSR envelopes)."""

import unittest
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.modulator import ADSREnvelope, getadsr
from src.constants import DEFAULT_SAMPLE_RATE


class TestADSREnvelope(unittest.TestCase):
    """Test suite for ADSR envelope."""

    def setUp(self) -> None:
        """Create an ADSR envelope for each test."""
        self.env = ADSREnvelope(
            attack_duration=0.1,
            decay_duration=0.2,
            sustain_level=0.7,
            release_duration=0.3,
            sample_rate=44100,
        )

    def test_initialization(self) -> None:
        """Test envelope initializes with correct parameters."""
        self.assertEqual(self.env.attack_duration, 0.1)
        self.assertEqual(self.env.decay_duration, 0.2)
        self.assertEqual(self.env.sustain_level, 0.7)
        self.assertEqual(self.env.release_duration, 0.3)

    def test_attack_phase(self) -> None:
        """Test attack phase ramps from 0 to 1."""
        samples = self.env.get_samples_iterator(int(0.1 * 44100), reset=True)

        # Should start near 0
        self.assertLess(samples[0], 0.1)

        # Should increase
        self.assertGreater(samples[-1], samples[0])

        # Should reach close to 1.0
        self.assertGreater(samples[-1], 0.9)

    def test_decay_phase(self) -> None:
        """Test decay phase ramps down to sustain level."""
        # Skip attack phase
        attack_samples = int(0.1 * 44100)
        _ = self.env.get_samples_iterator(attack_samples, reset=True)

        # Get decay phase
        decay_samples = self.env.get_samples_iterator(int(0.2 * 44100))

        # Should decrease from ~1.0 to sustain level (0.7)
        self.assertGreater(decay_samples[0], 0.9)
        self.assertAlmostEqual(decay_samples[-1], 0.7, delta=0.1)

    def test_sustain_phase(self) -> None:
        """Test sustain phase holds constant level."""
        # Skip to sustain phase
        ads_duration = int((0.1 + 0.2) * 44100)
        _ = self.env.get_samples_iterator(ads_duration, reset=True)

        # Get sustain samples
        sustain_samples = self.env.get_samples_iterator(1000)

        # Should all be close to sustain level
        self.assertTrue(np.all(np.abs(np.array(sustain_samples) - 0.7) < 0.05))

    def test_trigger_release(self) -> None:
        """Test trigger_release initiates release phase."""
        # Get to sustain
        _ = self.env.get_samples_iterator(int(0.5 * 44100), reset=True)

        # Trigger release
        self.env.trigger_release()

        # Get release samples
        release_samples = self.env.get_samples_iterator(int(0.3 * 44100))

        # Should decrease toward 0
        self.assertGreater(release_samples[0], release_samples[-1])
        self.assertLess(release_samples[-1], 0.1)

    def test_ended_flag(self) -> None:
        """Test ended flag is set after release completes."""
        self.assertFalse(self.env.ended)

        # Go through envelope
        _ = self.env.get_samples_iterator(int(0.5 * 44100), reset=True)
        self.env.trigger_release()
        _ = self.env.get_samples_iterator(int(0.4 * 44100))

        # Should be marked as ended
        self.assertTrue(self.env.ended)

    def test_iterator_vs_vectorized(self) -> None:
        """Test iterator and vectorized produce same results."""
        env1 = ADSREnvelope(0.1, 0.2, 0.7, 0.3)
        env2 = ADSREnvelope(0.1, 0.2, 0.7, 0.3)

        samples_iter = env1.get_samples_iterator(100, reset=True)
        samples_vec = env2.get_samples_vectorized(100)

        np.testing.assert_allclose(samples_iter, samples_vec, rtol=1e-10)

    def test_auto_mode(self) -> None:
        """Test auto mode selection."""
        # Small buffer uses iterator
        small = self.env.get_samples(100, mode="auto", reset=True)
        self.assertIsInstance(small, list)

        # Large buffer uses vectorized
        large = self.env.get_samples(1000, mode="auto", reset=True)
        self.assertIsInstance(large, np.ndarray)

    def test_zero_durations(self) -> None:
        """Test envelope with zero-length phases."""
        env = ADSREnvelope(
            attack_duration=0.0,
            decay_duration=0.0,
            sustain_level=0.7,
            release_duration=0.0,
        )

        # Should jump directly to sustain
        samples = env.get_samples_iterator(10, reset=True)
        self.assertAlmostEqual(samples[0], 0.7, delta=0.1)


class TestGetADSR(unittest.TestCase):
    """Test the getadsr convenience function."""

    def test_basic_functionality(self) -> None:
        """Test getadsr generates complete envelope."""
        adsr_vals, down_len, up_len = getadsr(
            a=0.05,
            d=0.1,
            sl=0.7,
            r=0.1,
            sd=0.2,
            sample_rate=44100,
        )

        # Should return array and lengths
        self.assertIsInstance(adsr_vals, np.ndarray)
        self.assertGreater(down_len, 0)
        self.assertGreater(up_len, 0)

        # Total length should match
        self.assertEqual(len(adsr_vals), down_len + up_len)

    def test_envelope_shape(self) -> None:
        """Test getadsr produces correct envelope shape."""
        adsr_vals, down_len, up_len = getadsr(
            a=0.1,
            d=0.1,
            sl=0.5,
            r=0.1,
            sd=0.1,
        )

        # Down phase should peak around 1.0
        down_phase = adsr_vals[:down_len]
        self.assertGreater(np.max(down_phase), 0.9)

        # Up phase (release) should end near 0
        up_phase = adsr_vals[down_len:]
        self.assertLess(up_phase[-1], 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
