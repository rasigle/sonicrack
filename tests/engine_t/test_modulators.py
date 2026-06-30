"""Unit tests for modulators (ADSR envelopes)."""

import unittest

import numpy as np

from src.engine.modulator import ADSREnvelope, DecayEnvelope, getadsr


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
        self.env.trigger_note_on()  # Trigger envelope to start attack phase
        samples = self.env._get_samples_iterator(int(0.1 * 44100), reset=True)

        # Should start near 0
        self.assertLess(samples[0], 0.1)

        # Should increase
        self.assertGreater(samples[-1], samples[0])

        # Should reach close to 1.0
        self.assertGreater(samples[-1], 0.9)

    def test_decay_phase(self) -> None:
        """Test decay phase ramps down to sustain level."""
        self.env.trigger_note_on()  # Trigger envelope
        # Skip attack phase
        attack_samples = int(0.1 * 44100)
        _ = self.env._get_samples_iterator(attack_samples, reset=True)

        # Get decay phase
        decay_samples = self.env._get_samples_iterator(int(0.2 * 44100))

        # Should decrease from ~1.0 to sustain level (0.7)
        self.assertGreater(decay_samples[0], 0.9)
        self.assertAlmostEqual(decay_samples[-1], 0.7, delta=0.1)

    def test_sustain_phase(self) -> None:
        """Test sustain phase holds constant level."""
        self.env.trigger_note_on()  # Trigger envelope
        # Skip to sustain phase
        ads_duration = int((0.1 + 0.2) * 44100)
        _ = self.env._get_samples_iterator(ads_duration, reset=True)

        # Get sustain samples
        sustain_samples = self.env._get_samples_iterator(1000)

        # Should all be close to sustain level
        self.assertTrue(np.all(np.abs(np.array(sustain_samples) - 0.7) < 0.05))

    def test_trigger_release(self) -> None:
        """Test trigger_release initiates release phase."""
        self.env.trigger_note_on()  # Trigger envelope
        # Get to sustain
        _ = self.env._get_samples_iterator(int(0.5 * 44100), reset=True)

        # Trigger release
        self.env.trigger_release()

        # Get release samples
        release_samples = self.env._get_samples_iterator(int(0.3 * 44100))

        # Should decrease toward 0
        self.assertGreater(release_samples[0], release_samples[-1])
        self.assertLess(release_samples[-1], 0.1)

    def test_ended_flag(self) -> None:
        """Test ended flag is set after release completes."""
        self.env.trigger_note_on()  # Trigger envelope
        # Initially not ended (during attack/decay/sustain)
        self.assertFalse(self.env.ended)

        # Go through envelope
        _ = self.env._get_samples_iterator(int(0.5 * 44100), reset=True)
        self.env.trigger_release()
        _ = self.env._get_samples_iterator(int(0.4 * 44100))

        # Should be marked as ended
        self.assertTrue(self.env.ended)

    def test_iterator_vs_vectorized(self) -> None:
        """Test iterator and vectorized produce same results."""
        env1 = ADSREnvelope(0.1, 0.2, 0.7, 0.3)
        env2 = ADSREnvelope(0.1, 0.2, 0.7, 0.3)

        samples_iter = env1._get_samples_iterator(100, reset=True)
        # Need to reset env2 first since get_samples_vectorized doesn't auto-reset
        iter(env2)
        samples_vec = env2._get_samples_vectorized(100)

        np.testing.assert_allclose(samples_iter, samples_vec, rtol=1e-7)

    def test_auto_mode(self) -> None:
        """Test auto mode selection."""
        # Small buffer uses iterator internally but returns ndarray
        small = self.env.get_samples(100, mode="auto", reset=True)
        self.assertIsInstance(small, np.ndarray)

        # Large buffer uses vectorized and returns ndarray
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
        env.trigger_note_on()  # Trigger envelope

        # Should jump directly to sustain
        samples = env._get_samples_iterator(10, reset=True)
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
        adsr_vals, down_len, _ = getadsr(
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


class TestDecayEnvelope(unittest.TestCase):
    """Test suite for triggered decay envelopes."""

    def test_instant_attack_decay_shape(self) -> None:
        env = DecayEnvelope(
            attack_duration=0.0,
            decay_duration=4 / 44100,
            amount=1.0,
            sample_rate=44100,
        )

        env.trigger_note_on()
        samples = env.get_samples(6)

        np.testing.assert_allclose(
            samples,
            [1.0, 0.75, 0.5, 0.25, 0.0, 0.0],
            atol=1e-7,
        )

    def test_attack_phase_ramps_to_amount(self) -> None:
        env = DecayEnvelope(
            attack_duration=4 / 44100,
            decay_duration=4 / 44100,
            amount=0.8,
            sample_rate=44100,
        )

        env.trigger_note_on()
        samples = env.get_samples(5)

        np.testing.assert_allclose(
            samples,
            [0.0, 0.2, 0.4, 0.6, 0.8],
            atol=1e-7,
        )

    def test_retrigger_restarts_shape(self) -> None:
        env = DecayEnvelope(decay_duration=4 / 44100, amount=1.0, sample_rate=44100)

        env.trigger_note_on()
        first = env.get_samples(2)
        env.trigger_note_on()
        second = env.get_samples(2)

        np.testing.assert_allclose(first, [1.0, 0.75], atol=1e-7)
        np.testing.assert_allclose(second, [1.0, 0.75], atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
