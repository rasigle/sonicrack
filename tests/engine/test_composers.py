"""Unit tests for composers (Chain, WaveAdder)."""

import unittest
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.oscillator import SineOscillator, SquareOscillator
from src.engine.composer import Chain, WaveAdder
from src.engine.modifier import Volume, Panner, Clipper


class TestChain(unittest.TestCase):
    """Test suite for Chain composer."""

    def setUp(self) -> None:
        """Create oscillators and modifiers for testing."""
        self.osc = SineOscillator(freq=440, amp=1.0)
        self.volume = Volume(0.5)
        self.panner = Panner(0.5)
        self.clipper = Clipper((-0.5, 0.5))

    def test_initialization(self) -> None:
        """Test chain initializes correctly."""
        chain = Chain(self.osc, self.volume)
        self.assertIsNotNone(chain)
        self.assertEqual(chain.oscillator, self.osc)

    def test_single_modifier(self) -> None:
        """Test chain with single modifier."""
        chain = Chain(self.osc, self.volume)
        samples = chain.get_samples_vectorized(100)

        # Should be scaled by volume (0.5)
        self.assertLess(np.max(np.abs(samples)), 0.51)

    def test_multiple_modifiers(self) -> None:
        """Test chain with multiple modifiers."""
        chain = Chain(self.osc, self.volume, self.clipper)
        samples = chain.get_samples_vectorized(100)

        # Should be clipped to [-0.5, 0.5]
        self.assertLessEqual(np.max(samples), 0.51)
        self.assertGreaterEqual(np.min(samples), -0.51)

    def test_stereo_output(self) -> None:
        """Test chain with stereo panning."""
        chain = Chain(self.osc, self.panner)
        samples = chain.get_samples_iterator(10, reset=True)

        # First sample should be tuple (left, right)
        self.assertIsInstance(samples[0], tuple)
        self.assertEqual(len(samples[0]), 2)

    def test_trigger_release(self) -> None:
        """Test trigger_release propagates to oscillator."""
        from src.engine.modulator import ADSREnvelope
        from src.engine.modulated_oscillator import ModulatedOscillator

        env = ADSREnvelope(0.1, 0.2, 0.7, 0.3)
        mod_osc = ModulatedOscillator(
            SineOscillator(440), env, amp_mod=lambda a, e: a * e
        )
        chain = Chain(mod_osc, self.volume)

        # Should have trigger_release method
        self.assertTrue(hasattr(chain, "trigger_release"))
        chain.trigger_release()

    def test_ended_property(self) -> None:
        """Test ended property reflects component state."""
        from src.engine.modulator import ADSREnvelope
        from src.engine.modulated_oscillator import ModulatedOscillator

        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1)
        mod_osc = ModulatedOscillator(
            SineOscillator(440), env, amp_mod=lambda a, e: a * e
        )
        chain = Chain(mod_osc)

        # Initially not ended
        self.assertFalse(chain.ended)

        # After release and enough time, should be ended
        _ = chain.get_samples_iterator(5000, reset=True)
        chain.trigger_release()
        _ = chain.get_samples_iterator(10000)

        self.assertTrue(chain.ended)

    def test_iterator_vs_vectorized(self) -> None:
        """Test iterator and vectorized produce same results."""
        chain1 = Chain(SineOscillator(440), Volume(0.5))
        chain2 = Chain(SineOscillator(440), Volume(0.5))

        samples_iter = chain1.get_samples_iterator(100, reset=True)
        samples_vec = chain2.get_samples_vectorized(100)

        np.testing.assert_allclose(samples_iter, samples_vec, rtol=1e-5)

    def test_auto_mode(self) -> None:
        """Test auto mode selection."""
        chain = Chain(self.osc, self.volume)

        # Small buffer uses iterator
        small = chain.get_samples(100, mode="auto", reset=True)
        self.assertIsInstance(small, list)

        # Large buffer uses vectorized
        large = chain.get_samples(1000, mode="auto", reset=True)
        self.assertIsInstance(large, np.ndarray)


class TestWaveAdder(unittest.TestCase):
    """Test suite for WaveAdder composer."""

    def setUp(self) -> None:
        """Create oscillators for testing."""
        self.osc1 = SineOscillator(freq=440, amp=0.5)
        self.osc2 = SineOscillator(freq=880, amp=0.5)

    def test_initialization(self) -> None:
        """Test wave adder initializes correctly."""
        adder = WaveAdder(self.osc1, self.osc2)
        self.assertIsNotNone(adder)
        self.assertEqual(len(adder.generators), 2)

    def test_mixing(self) -> None:
        """Test wave adder mixes signals correctly."""
        adder = WaveAdder(self.osc1, self.osc2)
        samples = adder.get_samples_vectorized(100)

        # Mixed signal should be different from either source
        osc1_samples = SineOscillator(440, amp=0.5).get_samples_vectorized(100)
        osc2_samples = SineOscillator(880, amp=0.5).get_samples_vectorized(100)

        self.assertFalse(np.allclose(samples, osc1_samples))
        self.assertFalse(np.allclose(samples, osc2_samples))

    def test_mono_output(self) -> None:
        """Test wave adder produces mono output by default."""
        adder = WaveAdder(self.osc1, self.osc2, stereo=False)
        samples = adder.get_samples_iterator(10, reset=True)

        # Should be scalar values
        self.assertIsInstance(samples[0], (int, float, np.number))

    def test_stereo_output(self) -> None:
        """Test wave adder produces stereo output when enabled."""
        panner1 = Panner(0.3)
        panner2 = Panner(0.7)

        chain1 = Chain(self.osc1, panner1)
        chain2 = Chain(self.osc2, panner2)

        adder = WaveAdder(chain1, chain2, stereo=True)
        samples = adder.get_samples_iterator(10, reset=True)

        # Should be tuples (left, right)
        self.assertIsInstance(samples[0], tuple)
        self.assertEqual(len(samples[0]), 2)

    def test_multiple_generators(self) -> None:
        """Test wave adder with more than 2 generators."""
        osc3 = SineOscillator(freq=1320, amp=0.33)

        adder = WaveAdder(self.osc1, self.osc2, osc3)
        samples = adder.get_samples_vectorized(100)

        # Should produce valid samples
        self.assertEqual(len(samples), 100)
        self.assertTrue(np.all(np.isfinite(samples)))

    def test_trigger_release(self) -> None:
        """Test trigger_release propagates to all generators."""
        from src.engine.modulator import ADSREnvelope
        from src.engine.modulated_oscillator import ModulatedOscillator

        env1 = ADSREnvelope(0.1, 0.1, 0.7, 0.1)
        env2 = ADSREnvelope(0.1, 0.1, 0.7, 0.1)

        mod_osc1 = ModulatedOscillator(
            SineOscillator(440), env1, amp_mod=lambda a, e: a * e
        )
        mod_osc2 = ModulatedOscillator(
            SineOscillator(880), env2, amp_mod=lambda a, e: a * e
        )

        adder = WaveAdder(mod_osc1, mod_osc2)

        # Should have trigger_release method
        self.assertTrue(hasattr(adder, "trigger_release"))
        adder.trigger_release()

    def test_iterator_vs_vectorized(self) -> None:
        """Test iterator and vectorized produce same results."""
        adder1 = WaveAdder(SineOscillator(440), SineOscillator(880))
        adder2 = WaveAdder(SineOscillator(440), SineOscillator(880))

        samples_iter = adder1.get_samples_iterator(100, reset=True)
        samples_vec = adder2.get_samples_vectorized(100)

        np.testing.assert_allclose(samples_iter, samples_vec, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
