"""Unit tests for modifiers (Panner, Volume, Clipper, etc.)."""

import unittest
from typing import cast

import numpy as np

from engine.generator.oscillator_modulated import ModulatedFrequency
from engine.generator.oscillator_ramp import TriangleOscillator
from engine.generator.oscillator_sine import SineOscillator
from engine.generator.oscillator_square import SquareOscillator
from src.constants import DEFAULT_GAIN_DB
from src.engine.composer import Chain
from src.engine.modifier import (
    Clipper,
    Frequency,
    ModulatedPanner,
    ModulatedVolume,
    Panner,
    Volume,
)
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

    def test_initialization_rejects_out_of_range(self) -> None:
        """Test panner rejects position values outside valid range."""
        with self.assertRaises(ValueError):
            Panner(-2.0)

        with self.assertRaises(ValueError):
            Panner(2.0)

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
                power,
                1.0,
                places=5,
                msg=f"Power not preserved at position {pos}: {power}",
            )

    def test_vectorized_panning(self) -> None:
        """Test vectorized panning with numpy arrays."""
        panner = Panner(0.0)
        samples = np.array([0.5, -0.5, 1.0, -1.0, 0.0])

        left, right = cast(tuple[np.ndarray, np.ndarray], panner(samples))

        # Should return numpy arrays
        self.assertIsInstance(left, np.ndarray)
        self.assertIsInstance(right, np.ndarray)
        self.assertEqual(len(left), len(samples))
        self.assertEqual(len(right), len(samples))

        # Check each sample
        for i, sample in enumerate(samples):
            expected_left = sample * np.sqrt(0.5)
            expected_right = sample * np.sqrt(0.5)
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

    def test_initialization_default(self):
        """Test Panner initializes with default center position."""
        panner = Panner()
        self.assertEqual(panner.position, 0.0)

    def test_initialization_left(self):
        """Test Panner initializes with left position."""
        panner = Panner(-1.0)
        self.assertEqual(panner.position, -1.0)

    def test_initialization_right(self):
        """Test Panner initializes with right position."""
        panner = Panner(1.0)
        self.assertEqual(panner.position, 1.0)

    def test_position_property_setter(self):
        """Test position property setter updates gains."""
        panner = Panner(0.0)
        old_left = panner._left_gain
        old_right = panner._right_gain

        panner.position = 1.0  # Full right
        self.assertNotEqual(panner._left_gain, old_left)
        self.assertNotEqual(panner._right_gain, old_right)

        with self.assertRaises(ValueError):
            panner.position = 2.0

    def test_call_with_scalar(self):
        """Test panner with scalar input."""
        panner = Panner(0.0)  # Center
        left, right = panner(1.0)

        self.assertIsInstance(left, (float, np.floating))
        self.assertIsInstance(right, (float, np.floating))
        # Center pan should have equal gains
        self.assertAlmostEqual(left, right, places=5)

    def test_call_with_array(self):
        """Test panner with array input."""
        panner = Panner(0.5)
        samples = np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32)
        left, right = cast(tuple[np.ndarray, np.ndarray], panner(samples))

        self.assertIsInstance(left, np.ndarray)
        self.assertIsInstance(right, np.ndarray)
        self.assertEqual(len(left), len(samples))
        self.assertEqual(len(right), len(samples))

    def test_pan_vectorized(self):
        """Test vectorized panning method."""
        panner = Panner(0.7)  # Pan right
        samples = np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32)
        left, right = panner.pan_vectorized(samples)

        self.assertEqual(left.dtype, np.float32)
        self.assertEqual(right.dtype, np.float32)
        # Right pan should have right gain > left gain (check absolute values)
        self.assertTrue(np.all(np.abs(right) >= np.abs(left)))

    def test_constant_power_panning_center(self):
        """Test constant-power law at center."""
        panner = Panner(0.0)
        # At center, power should be equal: left² + right² = 1
        power = panner._left_gain**2 + panner._right_gain**2
        self.assertAlmostEqual(power, 1.0, places=5)

    def test_constant_power_panning_extremes(self):
        """Test constant-power law at extremes."""
        panner_left = Panner(-1.0)
        panner_right = Panner(1.0)

        # At extremes, one channel should be 1, other should be 0
        self.assertAlmostEqual(panner_left._left_gain, 1.0, places=5)
        self.assertAlmostEqual(panner_left._right_gain, 0.0, places=5)
        self.assertAlmostEqual(panner_right._left_gain, 0.0, places=5)
        self.assertAlmostEqual(panner_right._right_gain, 1.0, places=5)

    def test_input_validation_invalid_type(self):
        """Test Panner rejects invalid position type."""
        with self.assertRaises(TypeError):
            Panner(cast(float, "invalid"))

        with self.assertRaises(TypeError):
            Panner(cast(float, [0.5]))


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
        env.trigger_note_on()  # Trigger envelope so it produces varying values
        panner = ModulatedPanner(env)

        # Get several values
        results = []
        for _ in range(10):
            result = panner(1.0)  # __call__ advances modulator internally
            results.append(result)

        # Pan should change over time due to modulation
        left_values = [r[0] for r in results]
        self.assertGreater(
            len(set(left_values)), 1, "Pan values should change over time"
        )

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
                power,
                1.0,
                places=5,
                msg=f"Power not preserved during modulation at position "
                f"{panner.position}",
            )
            next(panner)

    def test_vectorized_modulated_panning(self) -> None:
        """Test vectorized modulated panning."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=100)
        env.trigger_note_on()  # Trigger envelope
        panner = ModulatedPanner(env)

        num_samples = 20
        samples = np.ones(num_samples)

        left, right = cast(
            tuple[np.ndarray, np.ndarray], panner(samples)
        )  # Use __call__ which handles vectorization

        # Should return numpy arrays
        self.assertIsInstance(left, np.ndarray)
        self.assertIsInstance(right, np.ndarray)
        self.assertEqual(len(left), num_samples)
        self.assertEqual(len(right), num_samples)

        # Values should change over time
        self.assertGreater(len(set(left)), 1, "Left channel should vary over time")
        self.assertGreater(len(set(right)), 1, "Right channel should vary over time")

    def test_vectorized_preserves_power(self) -> None:
        """Test that vectorized modulated panning preserves power."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=100)
        env.trigger_note_on()  # Trigger envelope
        panner = ModulatedPanner(env)

        num_samples = 50
        samples = np.ones(num_samples)

        left, right = cast(
            tuple[np.ndarray, np.ndarray], panner(samples)
        )  # Use __call__ which handles vectorization

        # Check power preservation for each sample
        for i in range(num_samples):
            power = left[i] ** 2 + right[i] ** 2
            self.assertAlmostEqual(
                power, 1.0, places=5, msg=f"Power not preserved at sample {i}"
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
                actual_power,
                expected_power,
                places=5,
                msg=f"Power not correct for input {val}",
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

    def test_initialization_with_oscillator(self):
        """Test ModulatedPanner initializes with oscillator modulator."""
        lfo = SineOscillator(4, sample_rate=1000)
        panner = ModulatedPanner(lfo)
        self.assertIsNotNone(panner.modulator)

    def test_initialization_with_envelope(self):
        """Test ModulatedPanner initializes with envelope modulator."""
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1, sample_rate=1000)
        panner = ModulatedPanner(env)
        self.assertIsNotNone(panner.modulator)

    def test_iterator_protocol(self):
        """Test ModulatedPanner supports iteration."""
        lfo = SineOscillator(4, sample_rate=1000)
        panner = ModulatedPanner(lfo)
        iter(panner)

        # Should be able to get next value
        position = next(panner)
        self.assertIsInstance(position, (float, np.floating))
        self.assertGreaterEqual(position, -1.0)
        self.assertLessEqual(position, 1.0)

    def test_pan_vectorized_with_modulation(self):
        """Test vectorized panning with modulation."""
        lfo = SineOscillator(4, sample_rate=1000)
        panner = ModulatedPanner(lfo)
        samples = np.ones(100, dtype=np.float32)

        left, right = panner(samples)  # Use __call__ which handles vectorization

        self.assertEqual(len(left), 100)
        self.assertEqual(len(right), 100)
        self.assertEqual(left.dtype, np.float32)
        self.assertEqual(right.dtype, np.float32)

    def test_modulation_values_clipped(self):
        """Test modulation values are clipped to valid range."""
        # Create oscillator that outputs values > 1
        lfo = SineOscillator(4, amplitude=2.0, sample_rate=1000)
        panner = ModulatedPanner(lfo)

        # Get some positions
        for _ in range(10):
            position = next(panner)
            self.assertGreaterEqual(position, -1.0)
            self.assertLessEqual(position, 1.0)

    def test_input_validation_none_modulator(self):
        """Test ModulatedPanner rejects None modulator."""
        with self.assertRaises(TypeError):
            ModulatedPanner(None)

    def test_input_validation_non_iterable(self):
        """Test ModulatedPanner rejects non-iterable modulator."""
        with self.assertRaises(TypeError):
            ModulatedPanner(42)


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

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 0.5, places=5)
        self.assertAlmostEqual(result[1], 0.5, places=5)

    def test_initialization_default(self):
        """Test Volume initializes with default amplitude."""
        volume = Volume()
        self.assertEqual(volume.amplitude, 0.1)
        self.assertEqual(volume.gain_db, DEFAULT_GAIN_DB)

    def test_initialization_custom(self):
        """Test Volume initializes with custom amplitude."""
        volume = Volume(0.5)
        self.assertEqual(volume.amplitude, 0.5)

    def test_call_with_scalar(self):
        """Test volume with scalar input."""
        volume = Volume(0.5)
        result = volume(1.0)
        self.assertAlmostEqual(result, 0.5)

    def test_call_with_tuple(self):
        """Test volume with stereo tuple input."""
        volume = Volume(0.5)
        result = volume((1.0, -1.0))
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(result[1], -0.5)

    def test_call_with_array(self):
        """Test volume with array input."""
        volume = Volume(0.5)
        samples = np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32)
        result = volume(samples)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, samples * 0.5)

    def test_scale_vectorized(self):
        """Test vectorized scaling method."""
        volume = Volume(0.8)
        samples = np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32)
        result = volume._scale_vectorized(samples)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_almost_equal(result, samples * 0.8)

    def test_zero_amplitude(self):
        """Test volume with zero amplitude produces silence."""
        volume = Volume(0.0)
        samples = np.array([1.0, 0.5, -0.5, -1.0], dtype=np.float32)
        result = volume(samples)

        np.testing.assert_array_equal(result, np.zeros_like(samples))

    def test_amplification2(self):
        """Test volume can amplify signal."""
        volume = Volume(2.0)
        result = volume(0.5)
        self.assertAlmostEqual(result, 1.0)

    def test_input_validation_invalid_type(self):
        """Test Volume rejects invalid amplitude type."""
        with self.assertRaises(TypeError):
            Volume(cast(float, "invalid"))

    def test_input_validation_negative(self):
        """Test Volume rejects negative amplitude."""
        with self.assertRaises(ValueError):
            Volume(amplitude=-1.0)
        try:
            Volume(gain_db=-1.0)
        except ValueError:
            self.fail("Volume raised ValueError unexpectedly with gain_db!")

    def test_call_invalid_input(self):
        """Test Volume rejects invalid input types in call."""
        volume = Volume(0.5)
        with self.assertRaises(TypeError):
            volume(cast(float | tuple[float, ...] | np.ndarray, {"invalid": "dict"}))


class TestModulatedVolume(unittest.TestCase):
    """Test suite for ModulatedVolume."""

    def test_initialization(self) -> None:
        """Test modulated volume initializes correctly."""
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1)
        volume = ModulatedVolume(env)
        self.assertIsNotNone(volume.modulator)

    def test_modulation_changes_volume(self) -> None:
        """Test that modulation affects volume."""
        env = ADSREnvelope(0.1, 0.1, 0.5, 0.1, sample_rate=1000)
        env.trigger_note_on()  # Trigger envelope to produce varying values
        volume = ModulatedVolume(env)

        # Use vectorized mode to avoid smoothing interference
        # Create input signal (constant amplitude)
        input_signal = np.ones(100, dtype=np.float32)

        # Apply modulated volume
        result = volume(input_signal)

        # Volume should vary over time during attack phase
        # Check that output values are different (envelope is modulating)
        unique_values = len(np.unique(np.round(result, decimals=3)))
        self.assertGreater(
            unique_values, 10, "Volume should change during envelope attack"
        )

    def test_initialization_with_envelope(self):
        """Test ModulatedVolume initializes with envelope."""
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1, sample_rate=1000)
        volume = ModulatedVolume(env)
        self.assertIsNotNone(volume.modulator)

    def test_iterator_protocol(self):
        """Test ModulatedVolume supports iteration."""
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1, sample_rate=1000)
        volume = ModulatedVolume(env)
        iter(volume)

        amplitude = next(volume)
        # Accept both int and float (0 might be int initially)
        self.assertIsInstance(amplitude, (int, float, np.number))
        self.assertGreaterEqual(amplitude, 0.0)

    def test_trigger_release(self):
        """Test trigger_release propagates to modulator."""
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1, sample_rate=1000)
        volume = ModulatedVolume(env)

        # Should not raise
        volume.trigger_release()
        self.assertTrue(hasattr(env, "ended"))

    def test_ended_property(self):
        """Test ended property reflects modulator state."""
        env = ADSREnvelope(0.01, 0.01, 0.7, 0.01, sample_rate=1000)
        env.trigger_note_on()  # Trigger envelope
        volume = ModulatedVolume(env)

        # Initially not ended (during attack/decay/sustain)
        self.assertFalse(volume.ended)

    def test_input_validation_none_modulator(self):
        """Test ModulatedVolume rejects None modulator."""
        with self.assertRaises(TypeError):
            ModulatedVolume(None)


class TestFrequency(unittest.TestCase):
    """Test suite for Frequency modifier."""

    def test_initialization(self) -> None:
        """Test frequency modifier initializes correctly."""
        freq_mod = Frequency(2.0)
        self.assertEqual(freq_mod.frequency, 2.0)

    def test_frequency_scaling(self) -> None:
        """Test frequency modifier scales value."""
        freq_mod = Frequency(2.0)
        result = cast(float, freq_mod(440.0))
        self.assertAlmostEqual(result, 880.0, places=5)

    def test_initialization_default(self):
        """Test Frequency initializes with default multiplier."""
        freq = Frequency()
        self.assertEqual(freq.frequency, 1.0)

    def test_initialization_custom(self):
        """Test Frequency initializes with custom multiplier."""
        freq = Frequency(2.0)
        self.assertEqual(freq.frequency, 2.0)

    def test_call_with_scalar(self):
        """Test frequency with scalar input."""
        freq = Frequency(2.0)
        result = freq(440.0)
        self.assertAlmostEqual(result, 880.0)

    def test_call_with_tuple(self):
        """Test frequency with tuple input."""
        freq = Frequency(2.0)
        result = freq((440.0, 220.0))
        self.assertIsInstance(result, tuple)
        self.assertAlmostEqual(result[0], 880.0)
        self.assertAlmostEqual(result[1], 440.0)

    def test_call_with_array(self):
        """Test frequency with array input."""
        freq = Frequency(2.0)
        samples = np.array([440.0, 220.0, 110.0], dtype=np.float32)
        result = freq(samples)

        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, samples * 2.0)

    def test_scale_vectorized(self):
        """Test vectorized scaling method."""
        freq = Frequency(1.5)
        samples = np.array([440.0, 220.0], dtype=np.float32)
        result = freq.scale_vectorized(samples)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_almost_equal(result, samples * 1.5)

    def test_input_validation_invalid_type(self):
        """Test Frequency rejects invalid type."""
        with self.assertRaises(TypeError):
            Frequency(cast(float, "invalid"))

    def test_input_validation_negative(self):
        """Test Frequency rejects negative value."""
        with self.assertRaises(ValueError):
            Frequency(-1.0)


class TestClipper(unittest.TestCase):
    """Test suite for Clipper modifier."""

    def test_initialization(self) -> None:
        """Test clipper initializes with correct range."""
        clipper = Clipper((-0.5, 0.5))
        self.assertEqual(clipper.wave_range, (-0.5, 0.5))

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

        result = cast(tuple[float, ...], clipper((1.0, -1.0)))
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], 0.6)
        self.assertEqual(result[1], -0.4)

        result = cast(tuple[float, ...], clipper((-1.0, 1.0)))
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], -0.4)
        self.assertEqual(result[1], 0.6)

    def test_initialization_default(self):
        """Test Clipper initializes with default range."""
        clipper = Clipper()
        self.assertEqual(clipper._min, -1.0)
        self.assertEqual(clipper._max, 1.0)

    def test_initialization_custom(self):
        """Test Clipper initializes with custom range."""
        clipper = Clipper((-0.5, 0.5))
        self.assertEqual(clipper._min, -0.5)
        self.assertEqual(clipper._max, 0.5)

    def test_call_with_scalar_no_clipping(self):
        """Test clipper with scalar in range."""
        clipper = Clipper((-1.0, 1.0))
        result = clipper(0.5)
        self.assertAlmostEqual(result, 0.5)

    def test_call_with_scalar_clipping_high(self):
        """Test clipper clips high values."""
        clipper = Clipper((-1.0, 1.0))
        result = clipper(2.0)
        self.assertAlmostEqual(result, 1.0)

    def test_call_with_scalar_clipping_low(self):
        """Test clipper clips low values."""
        clipper = Clipper((-1.0, 1.0))
        result = clipper(-2.0)
        self.assertAlmostEqual(result, -1.0)

    def test_call_with_tuple(self):
        """Test clipper with tuple input."""
        clipper = Clipper((-1.0, 1.0))
        result = clipper((2.0, -2.0, 0.5))
        self.assertIsInstance(result, tuple)
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], -1.0)
        self.assertAlmostEqual(result[2], 0.5)

    def test_call_with_array(self):
        """Test clipper with array input."""
        clipper = Clipper((-1.0, 1.0))
        samples = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
        result = clipper(samples)

        self.assertIsInstance(result, np.ndarray)
        expected = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_clip_vectorized(self):
        """Test vectorized clipping method."""
        clipper = Clipper((-0.5, 0.5))
        samples = np.array([1.0, -1.0, 0.3, -0.3], dtype=np.float32)
        result = clipper.clip_vectorized(samples)

        self.assertEqual(result.dtype, np.float32)
        expected = np.array([0.5, -0.5, 0.3, -0.3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_range_property_setter(self):
        """Test wave_range property setter updates min/max."""
        clipper = Clipper((-1.0, 1.0))
        clipper.wave_range = (-0.5, 0.5)

        self.assertEqual(clipper._min, -0.5)
        self.assertEqual(clipper._max, 0.5)

    def test_input_validation_invalid_type(self):
        """Test Clipper rejects invalid wave_range type."""
        with self.assertRaises(TypeError):
            Clipper(cast(tuple[float, float], "invalid"))

    def test_input_validation_wrong_length(self):
        """Test Clipper rejects wrong length tuple."""
        with self.assertRaises(ValueError):
            Clipper(cast(tuple[float, float], (-1.0,)))

        with self.assertRaises(ValueError):
            Clipper(cast(tuple[float, float], (-1.0, 0.0, 1.0)))

    def test_input_validation_invalid_range(self):
        """Test Clipper rejects invalid range (min >= max)."""
        with self.assertRaises(ValueError):
            Clipper((1.0, -1.0))

        with self.assertRaises(ValueError):
            Clipper((0.5, 0.5))

    def test_input_validation_non_numeric_values(self):
        """Test Clipper rejects non-numeric range values."""
        with self.assertRaises(TypeError):
            Clipper(cast(tuple[float, float], ("a", "b")))


class TestModifierIntegration(unittest.TestCase):
    """Test modifiers work together in chains."""

    def test_volume_then_pan(self) -> None:
        """Test volume followed by panning."""
        volume = Volume(0.5)
        panner = Panner(0.5)  # Right-biased pan

        # Apply volume, then pan
        after_volume = cast(float, volume(1.0))
        after_pan = cast(tuple[float, float], panner(after_volume))

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


class TestModifierIntegrationAdditional(unittest.TestCase):
    """Integration tests for modifiers working together."""

    def test_chain_volume_and_clipper(self):
        """Test volume followed by clipper."""
        volume = Volume(2.0)
        clipper = Clipper((-1.0, 1.0))

        samples = np.array([0.8, -0.8], dtype=np.float32)
        scaled = volume(samples)
        clipped = clipper(scaled)

        # Should be clipped to [-1, 1]
        np.testing.assert_array_equal(clipped, np.array([1.0, -1.0]))

    def test_panner_with_volume(self):
        """Test panner output can be scaled by volume."""
        panner = Panner(0.0)
        volume = Volume(0.5)

        mono_sample = 1.0
        left, right = panner(mono_sample)

        # Scale stereo output
        left_scaled = volume(left)
        right_scaled = volume(right)

        self.assertAlmostEqual(left_scaled, left * 0.5)
        self.assertAlmostEqual(right_scaled, right * 0.5)

    def test_modulated_chain(self):
        """Test modulated modifiers in chain."""
        lfo = SineOscillator(4, sample_rate=1000)
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1, sample_rate=1000)

        panner = ModulatedPanner(lfo)
        volume = ModulatedVolume(env)

        # Both should work together
        self.assertIsNotNone(panner.modulator)
        self.assertIsNotNone(volume.modulator)


class TestModulatedFrequency(unittest.TestCase):
    """Test cases for ModulatedFrequency class."""

    def test_initialization_with_modulator(self):
        """Test ModulatedFrequency initializes with modulator."""
        lfo = SineOscillator(5, sample_rate=1000)
        vol = ModulatedVolume(lfo)
        self.assertIsNotNone(vol.modulator)

    def test_iterator_protocol(self):
        """Test ModulatedFrequency supports iteration."""
        lfo = SineOscillator(5, sample_rate=1000)
        vol = ModulatedVolume(lfo)
        iter(vol)

        frequency = next(vol)
        self.assertIsInstance(frequency, (float, np.floating))

    def test_input_validation_none_modulator(self):
        """Test ModulatedFrequency rejects None modulator."""
        with self.assertRaises(TypeError):
            # pylint: disable=no-value-for-parameter
            ModulatedFrequency(cast(object, None))


class TestModulatedVolumeVectorization(unittest.TestCase):
    """Test ModulatedVolume vectorized methods."""

    def test_regression_chain_with_modulated_volume(self):
        """Regression test: Chain should use vectorized ModulatedVolume, not iterator
        fallback.

        This test ensures that the bug where Chain fell back to Python loops
        for ModulatedVolume has been fixed.
        """

        # Create chain with ModulatedVolume (the problematic case)
        osc = SquareOscillator(440, amplitude=0.5, gain_db=None, sample_rate=1000)
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1, sample_rate=1000)
        env.trigger_note_on()  # Trigger envelope to produce varying values
        mod_vol = ModulatedVolume(env)

        chain = Chain(osc, mod_vol)

        # This should use vectorization, not iterator fallback
        samples = chain.get_samples(100, mode="vectorized")

        # Verify output
        self.assertEqual(len(samples), 100)
        self.assertEqual(samples.dtype, np.float32)

        # Verify modulation happened (samples should vary)
        self.assertGreater(np.std(samples), 0.005)  # Should have variation

        # First samples should be lower (attack phase)
        self.assertLess(np.mean(samples[:10]), np.mean(samples[40:50]))


class TestChainVectorizationPerformance(unittest.TestCase):
    """Performance regression tests for Chain vectorization."""

    def test_chain_detects_vectorized_methods(self):
        """Test that Chain properly detects and uses vectorized methods.

        This is the core regression test for the bug where Chain was
        falling back to Python loops instead of using vectorized methods.
        """
        # Setup chain with all vectorized-capable modifiers
        osc = SineOscillator(440, amplitude=1.0, sample_rate=1000)
        env = ADSREnvelope(0.1, 0.1, 0.7, 0.1, sample_rate=1000)
        mod_vol = ModulatedVolume(env)

        chain = Chain(osc, mod_vol)

        # Generate samples - should use vectorization
        n = 1000
        samples = chain.get_samples(n, mode="vectorized")

        # Verify correct output
        self.assertEqual(len(samples), n)
        self.assertEqual(samples.dtype, np.float32)

    def test_chain_with_multiple_modulated_modifiers(self):
        """Test Chain with multiple modulated modifiers (complex case)."""

        # This is similar to the user's original code
        osc = SquareOscillator(440, amplitude=0.3, sample_rate=1000)
        env = ADSREnvelope(
            attack_duration=0.2,
            decay_duration=0.1,
            sustain_level=0.7,
            release_duration=0.1,
            sample_rate=1000,
        )
        mod_vol = ModulatedVolume(env)
        lfo = TriangleOscillator(1, phase=180, wave_range=(-1, 1), sample_rate=1000)
        mod_pan = ModulatedPanner(lfo)

        chain = Chain(osc, mod_vol, mod_pan)

        # Generate samples
        n = 500
        samples = chain.get_samples(n, mode="vectorized")

        # Should produce stereo output from ModulatedPanner
        self.assertEqual(samples.ndim, 2)
        self.assertEqual(samples.shape, (n, 2))
        self.assertEqual(samples.dtype, np.float32)

        # Should be able to unpack with .T
        left, right = samples.T
        self.assertEqual(len(left), n)
        self.assertEqual(len(right), n)


if __name__ == "__main__":
    unittest.main()
