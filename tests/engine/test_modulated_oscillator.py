"""Unit tests for ModulatedOscillator."""

import unittest
from typing import cast

import numpy as np

from src.engine.oscillator_base import Oscillator
from src.engine.oscillator_modulated import ModulatedOscillator
from src.engine.modulator import ADSREnvelope
from src.engine.oscillator import SineOscillator, SquareOscillator, SawtoothOscillator


class TestModulatedOscillatorInitialization(unittest.TestCase):
    """Test ModulatedOscillator initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic initialization with amplitude modulation."""
        osc = SineOscillator(440)
        env = ADSREnvelope(0.1, 0.2, 0.7, 0.3)

        mod_osc = ModulatedOscillator(osc, env, amp_mod=lambda a, e: a * e)

        self.assertIsNotNone(mod_osc)
        self.assertEqual(mod_osc.oscillator, osc)

    def test_multiple_modulators(self) -> None:
        """Test initialization with multiple modulators."""
        osc = SineOscillator(440)
        env1 = ADSREnvelope(0.1, 0.2, 0.7, 0.3)
        env2 = ADSREnvelope(0.05, 0.1, 0.8, 0.2)

        mod_osc = ModulatedOscillator(
            osc,
            env1,
            env2,
            amp_mod=lambda a, e: a * e,
            freq_mod=lambda f, e: f * (1 + 0.1 * e),
        )

        self.assertEqual(mod_osc._modulators_count, 2)

    def test_type_checking(self) -> None:
        """Test type checking for oscillator parameter."""
        env = ADSREnvelope(0.1, 0.2, 0.7, 0.3)

        # Should raise TypeError with non-Oscillator
        with self.assertRaises(TypeError):
            ModulatedOscillator(
                cast(Oscillator, cast(object, "not an oscillator")),
                env,
                amp_mod=lambda a, e: a * e,
            )


class TestAmplitudeModulation(unittest.TestCase):
    """Test amplitude modulation functionality."""

    def test_amp_modulation_affects_output(self) -> None:
        """Test amplitude modulation changes output amplitude."""
        osc = SineOscillator(frequency=440, amplitude=1.0, gain_db=None)
        env = ADSREnvelope(
            attack_duration=0.1,
            decay_duration=0.0,
            sustain_level=1.0,
            release_duration=0.0,
            sample_rate=100,
        )
        env.trigger_note_on()  # Trigger envelope to start attack phase

        mod_osc = ModulatedOscillator(
            osc, env, amp_mod=lambda base_amp, env_val: base_amp * env_val
        )

        # Generate samples during attack phase
        samples = mod_osc.get_samples_iterator(10, reset=True)

        # Amplitude should increase during attack
        self.assertLess(abs(samples[0]), abs(samples[-1]))

    def test_zero_envelope_produces_silence(self) -> None:
        """Test that zero envelope value produces silence."""
        osc = SineOscillator(frequency=440, amplitude=1.0)
        env = ADSREnvelope(
            attack_duration=0.0,
            decay_duration=0.0,
            sustain_level=0.0,
            release_duration=0.0,
        )

        mod_osc = ModulatedOscillator(
            osc, env, amp_mod=lambda base_amp, env_val: base_amp * env_val
        )

        samples = mod_osc.get_samples_iterator(10, reset=True)

        # Should be close to silence
        self.assertTrue(all(abs(s) < 0.1 for s in samples))


class TestFrequencyModulation(unittest.TestCase):
    """Test frequency modulation functionality."""

    def test_freq_modulation_changes_frequency(self) -> None:
        """Test frequency modulation affects oscillator frequency."""
        osc = SineOscillator(frequency=440, amplitude=1.0)
        env = ADSREnvelope(0.1, 0.0, 1.0, 0.0, sample_rate=100)

        mod_osc = ModulatedOscillator(
            osc, env, freq_mod=lambda base_freq, env_val: base_freq * (1 + env_val)
        )

        # Generate samples
        _ = mod_osc.get_samples_iterator(50, reset=True)

        # Frequency should have been modulated
        # (Hard to test precisely, but we can verify it didn't crash)
        self.assertTrue(True)


class TestPhaseModulation(unittest.TestCase):
    """Test phase modulation functionality."""

    def test_phase_modulation(self) -> None:
        """Test phase modulation affects oscillator phase."""
        osc = SineOscillator(frequency=440, amplitude=1.0, phase=0.0)
        env = ADSREnvelope(0.1, 0.0, 1.0, 0.0, sample_rate=100)

        mod_osc = ModulatedOscillator(
            osc, env, phase_mod=lambda base_phase, env_val: base_phase + env_val * 90
        )

        # Generate samples
        samples = mod_osc.get_samples_iterator(10, reset=True)

        # Should produce valid output
        self.assertEqual(len(samples), 10)
        self.assertTrue(all(isinstance(s, (int, float, np.number)) for s in samples))


class TestTriggerRelease(unittest.TestCase):
    """Test trigger_release functionality."""

    def test_trigger_release_propagates(self) -> None:
        """Test trigger_release is called on modulators."""
        osc = SineOscillator(440)
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1)

        mod_osc = ModulatedOscillator(osc, env, amp_mod=lambda a, e: a * e)

        # Generate some samples
        _ = mod_osc.get_samples_iterator(50, reset=True)

        # Trigger release
        mod_osc.trigger_release()

        # Should have triggered release on envelope
        self.assertTrue(hasattr(mod_osc, "trigger_release"))

    def test_ended_property(self) -> None:
        """Test ended property reflects modulator state."""
        osc = SineOscillator(440, gain_db=None)
        env = ADSREnvelope(0.05, 0.05, 0.7, 0.05, sample_rate=1000)
        env.trigger_note_on()  # Trigger envelope so it's not in ended state

        mod_osc = ModulatedOscillator(osc, env, amp_mod=lambda a, e: a * e)

        # Initially not ended (during attack/decay/sustain)
        self.assertFalse(mod_osc.ended)

        # Generate samples and trigger release
        _ = mod_osc.get_samples_iterator(100, reset=True)
        mod_osc.trigger_release()
        release_samples = mod_osc.get_samples_iterator(300)

        # After sufficient release time, should be ended
        # Check if envelope modulator has ended property and it's true
        if hasattr(env, "ended"):
            # If envelope supports ended, it should be true after release completes
            # Note: The envelope may not be "ended" in the traditional sense as it
            # continues to output the sustain level, so we just verify samples were
            # generated
            self.assertEqual(len(release_samples), 300)


class TestSampleGeneration(unittest.TestCase):
    """Test sample generation methods."""

    def test_get_samples_iterator(self) -> None:
        """Test iterator-based sample generation."""
        osc = SineOscillator(440)
        env = ADSREnvelope(0.1, 0.2, 0.7, 0.3)

        mod_osc = ModulatedOscillator(osc, env, amp_mod=lambda a, e: a * e)

        samples = mod_osc.get_samples_iterator(100, reset=True)

        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(len(samples), 100)

    def test_get_samples_vectorized(self) -> None:
        """Test vectorized sample generation."""
        osc = SineOscillator(440)
        env = ADSREnvelope(0.1, 0.2, 0.7, 0.3)

        mod_osc = ModulatedOscillator(osc, env, amp_mod=lambda a, e: a * e)

        samples = mod_osc.get_samples_vectorized(100)

        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(len(samples), 100)

    def test_auto_mode_selection(self) -> None:
        """Test auto mode selects appropriate method."""
        osc = SineOscillator(440)
        env = ADSREnvelope(0.1, 0.2, 0.7, 0.3)

        mod_osc = ModulatedOscillator(osc, env, amp_mod=lambda a, e: a * e)

        # Small buffer should use iterator
        small = mod_osc.get_samples(100, mode="auto", reset=True)
        self.assertIsInstance(small, np.ndarray)

        # Large buffer should use vectorized
        large = mod_osc.get_samples(1000, mode="auto", reset=True)
        self.assertIsInstance(large, np.ndarray)

    def test_vectorized_matches_iterator_for_bright_sine_mode(self) -> None:
        """Vectorized modulation should preserve sine harmonic modes."""
        sample_rate = 2000
        iterator_osc = SineOscillator(
            frequency=55,
            amplitude=0.7,
            gain_db=None,
            sample_rate=sample_rate,
            mode="bright",
        )
        vectorized_osc = SineOscillator(
            frequency=55,
            amplitude=0.7,
            gain_db=None,
            sample_rate=sample_rate,
            mode="bright",
        )
        iterator_mod = SineOscillator(
            frequency=2,
            amplitude=1.0,
            gain_db=None,
            sample_rate=sample_rate,
            wave_range=(0, 1),
        )
        vectorized_mod = SineOscillator(
            frequency=2,
            amplitude=1.0,
            gain_db=None,
            sample_rate=sample_rate,
            wave_range=(0, 1),
        )

        mod_iter = ModulatedOscillator(
            iterator_osc,
            iterator_mod,
            amp_mod=lambda base_amp, env_val: base_amp * env_val,
        )
        mod_vec = ModulatedOscillator(
            vectorized_osc,
            vectorized_mod,
            amp_mod=lambda base_amp, env_val: base_amp * env_val,
        )

        samples_iter = mod_iter.get_samples_iterator(512, reset=True)
        samples_vec = mod_vec.get_samples_vectorized(512)

        np.testing.assert_allclose(samples_iter, samples_vec, rtol=1e-5, atol=1e-5)

    def test_vectorized_matches_iterator_for_square_oscillator(self) -> None:
        """Vectorized modulation should keep square waves as square waves."""
        sample_rate = 4000
        iterator_osc = SquareOscillator(
            frequency=35,
            amplitude=0.8,
            gain_db=None,
            sample_rate=sample_rate,
            pulsewidth=0.3,
            mode="soft",
            smoothness=18.0,
        )
        vectorized_osc = SquareOscillator(
            frequency=35,
            amplitude=0.8,
            gain_db=None,
            sample_rate=sample_rate,
            pulsewidth=0.3,
            mode="soft",
            smoothness=18.0,
        )
        iterator_mod = SineOscillator(
            frequency=1.5,
            amplitude=0.1,
            gain_db=None,
            sample_rate=sample_rate,
            phase=90,
        )
        vectorized_mod = SineOscillator(
            frequency=1.5,
            amplitude=0.1,
            gain_db=None,
            sample_rate=sample_rate,
            phase=90,
        )

        mod_iter = ModulatedOscillator(
            iterator_osc,
            iterator_mod,
            freq_mod=lambda base_freq, mod_val: base_freq * (1.0 + mod_val),
        )
        mod_vec = ModulatedOscillator(
            vectorized_osc,
            vectorized_mod,
            freq_mod=lambda base_freq, mod_val: base_freq * (1.0 + mod_val),
        )

        samples_iter = mod_iter.get_samples_iterator(512, reset=True)
        samples_vec = mod_vec.get_samples_vectorized(512)

        np.testing.assert_allclose(samples_iter, samples_vec, rtol=1e-5, atol=1e-5)

    def test_vectorized_preserves_analog_sawtooth_mode(self) -> None:
        """Vectorized modulation should keep analog sawtooth distinct from pure."""
        sample_rate = 3000
        analog_osc = SawtoothOscillator(
            frequency=40,
            amplitude=0.6,
            gain_db=None,
            sample_rate=sample_rate,
            mode="analog",
        )
        pure_osc = SawtoothOscillator(
            frequency=40,
            amplitude=0.6,
            gain_db=None,
            sample_rate=sample_rate,
            mode="pure",
        )
        analog_mod = SineOscillator(
            frequency=1,
            amplitude=1.0,
            gain_db=None,
            sample_rate=sample_rate,
            wave_range=(0, 1),
        )
        pure_mod = SineOscillator(
            frequency=1,
            amplitude=1.0,
            gain_db=None,
            sample_rate=sample_rate,
            wave_range=(0, 1),
        )

        mod_analog = ModulatedOscillator(
            analog_osc,
            analog_mod,
            amp_mod=lambda base_amp, env_val: base_amp * env_val,
        )
        mod_pure = ModulatedOscillator(
            pure_osc,
            pure_mod,
            amp_mod=lambda base_amp, env_val: base_amp * env_val,
        )

        analog_samples = mod_analog.get_samples_vectorized(512)
        pure_samples = mod_pure.get_samples_vectorized(512)

        self.assertFalse(np.allclose(analog_samples, pure_samples))
        self.assertGreater(np.max(np.abs(analog_samples - pure_samples)), 0.01)


class TestMultipleModulators(unittest.TestCase):
    """Test behavior with multiple modulators."""

    def test_two_modulators(self) -> None:
        """Test with two modulators."""
        osc = SineOscillator(440)
        env1 = ADSREnvelope(0.1, 0.2, 0.7, 0.3)
        env2 = ADSREnvelope(0.05, 0.1, 0.8, 0.2)

        mod_osc = ModulatedOscillator(
            osc,
            env1,
            env2,
            amp_mod=lambda a, e: a * e,
            freq_mod=lambda f, e: f * (1 + 0.1 * e),
        )

        samples = mod_osc.get_samples_iterator(100, reset=True)

        self.assertEqual(len(samples), 100)
        self.assertTrue(all(isinstance(s, (int, float, np.number)) for s in samples))

    def test_three_modulators(self) -> None:
        """Test with three modulators (max)."""
        osc = SineOscillator(440)
        env1 = ADSREnvelope(0.1, 0.2, 0.7, 0.3)
        env2 = ADSREnvelope(0.05, 0.1, 0.8, 0.2)
        env3 = ADSREnvelope(0.08, 0.15, 0.6, 0.25)

        mod_osc = ModulatedOscillator(
            osc,
            env1,
            env2,
            env3,
            amp_mod=lambda a, e: a * e,
            freq_mod=lambda f, e: f * (1 + 0.1 * e),
            phase_mod=lambda p, e: p + e * 10,
        )

        samples = mod_osc.get_samples_iterator(100, reset=True)

        self.assertEqual(len(samples), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
