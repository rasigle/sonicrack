"""Tests for CV (Control Voltage) scaling utilities.

Tests the CVScaler component and helper functions to ensure correct
signal range conversion for modular synthesis.
"""

import numpy as np
import pytest

from src.engine import (
    ADSREnvelope,
    CVScaler,
    PITCH_CV_REFERENCE_FREQUENCY,
    SineOscillator,
    bipolar_to_unipolar,
    frequency_to_pitch_cv,
    midi_note_to_pitch_cv,
    pitch_cv_to_frequency,
    scale_cv,
    unipolar_to_bipolar,
)


class TestPitchCVConversion:
    """Tests for global 1V/oct pitch CV helpers."""

    def test_midi_note_to_pitch_cv_uses_c4_as_zero_volts(self):
        assert midi_note_to_pitch_cv(60) == pytest.approx(0.0)
        assert midi_note_to_pitch_cv(72) == pytest.approx(1.0)
        assert midi_note_to_pitch_cv(48) == pytest.approx(-1.0)

    def test_pitch_cv_to_frequency_doubles_per_volt(self):
        assert pitch_cv_to_frequency(0.0) == pytest.approx(PITCH_CV_REFERENCE_FREQUENCY)
        assert pitch_cv_to_frequency(1.0) == pytest.approx(
            PITCH_CV_REFERENCE_FREQUENCY * 2.0
        )
        assert pitch_cv_to_frequency(-1.0) == pytest.approx(
            PITCH_CV_REFERENCE_FREQUENCY / 2.0
        )

    def test_frequency_to_pitch_cv_round_trips_arrays(self):
        pitch_cv = np.array([-1.0, 0.0, 0.75, 1.0], dtype=np.float32)
        frequencies = pitch_cv_to_frequency(pitch_cv)

        np.testing.assert_allclose(
            frequency_to_pitch_cv(frequencies),
            pitch_cv,
            atol=1e-6,
        )


class TestCVScaler:
    """Tests for CVScaler component."""

    def test_bipolar_to_unipolar_scaling(self):
        """CVScaler correctly scales bipolar [-1, 1] to unipolar [0, 1]."""
        # Create bipolar source (oscillator)
        source = SineOscillator(1, amplitude=1.0, sample_rate=44100)

        # Scale to unipolar
        scaler = CVScaler(source, input_range=(-1, 1), output_range=(0, 1))

        # Get samples
        samples = scaler.get_samples(1000)

        # Check range
        assert np.min(samples) >= 0.0
        assert np.max(samples) <= 1.0

        # Check that -1 maps to 0
        source_at_min = -1.0
        expected = 0.0
        actual = source_at_min * scaler._scale + scaler._offset
        assert abs(actual - expected) < 0.001

        # Check that 1 maps to 1
        source_at_max = 1.0
        expected = 1.0
        actual = source_at_max * scaler._scale + scaler._offset
        assert abs(actual - expected) < 0.001

    def test_unipolar_to_bipolar_scaling(self):
        """CVScaler correctly scales unipolar [0, 1] to bipolar [-1, 1]."""
        # Create unipolar source (envelope) - starts automatically
        source = ADSREnvelope(
            attack_duration=0.1,
            decay_duration=0.1,
            sustain_level=0.7,
            release_duration=0.1,
        )

        # Scale to bipolar
        scaler = CVScaler(source, input_range=(0, 1), output_range=(-1, 1))

        # Get samples (envelope starts producing values automatically)
        samples = scaler.get_samples(1000)

        # Check range
        assert np.min(samples) >= -1.0
        assert np.max(samples) <= 1.0

        # Check that 0 maps to -1
        source_at_min = 0.0
        expected = -1.0
        actual = source_at_min * scaler._scale + scaler._offset
        assert abs(actual - expected) < 0.001

        # Check that 1 maps to 1
        source_at_max = 1.0
        expected = 1.0
        actual = source_at_max * scaler._scale + scaler._offset
        assert abs(actual - expected) < 0.001

    def test_custom_range_scaling(self):
        """CVScaler works with arbitrary custom ranges."""
        source = SineOscillator(1, amplitude=0.5, sample_rate=44100)  # [-0.5, 0.5]

        # Scale to [0.2, 0.8]
        scaler = CVScaler(source, input_range=(-0.5, 0.5), output_range=(0.2, 0.8))

        samples = scaler.get_samples(1000)

        # Check range (with small tolerance for floating point)
        assert np.min(samples) >= 0.19
        assert np.max(samples) <= 0.81

    def test_clamping_works(self):
        """CVScaler clamps values outside the expected range."""
        source = SineOscillator(
            1, amplitude=2.0, sample_rate=44100
        )  # [-2, 2] (exceeds expected)

        # Scale expecting [-1, 1] input, but with clamping
        scaler = CVScaler(source, input_range=(-1, 1), output_range=(0, 1), clamp=True)

        samples = scaler.get_samples(1000)

        # Even though source exceeds [-1, 1], output should be clamped to [0, 1]
        assert np.min(samples) >= 0.0
        assert np.max(samples) <= 1.0

    def test_iterator_protocol(self):
        """CVScaler implements iterator protocol correctly."""
        source = SineOscillator(1, amplitude=1.0, sample_rate=44100)
        scaler = CVScaler(source, input_range=(-1, 1), output_range=(0, 1))

        # Should be iterable
        assert hasattr(scaler, "__iter__")
        assert hasattr(scaler, "__next__")

        # Get individual values
        val1 = next(scaler)
        val2 = next(scaler)

        # Should be in correct range
        assert 0.0 <= val1 <= 1.0
        assert 0.0 <= val2 <= 1.0

        # Should be able to reset
        iter(scaler)
        val3 = next(scaler)
        assert 0.0 <= val3 <= 1.0

    def test_invalid_source_raises_error(self):
        """CVScaler raises TypeError for non-iterable source."""
        with pytest.raises(TypeError, match="iterator protocol"):
            CVScaler(42, input_range=(-1, 1), output_range=(0, 1))

    def test_invalid_input_range_raises_error(self):
        """CVScaler raises ValueError for invalid input range."""
        source = SineOscillator(1, amplitude=1.0, sample_rate=44100)

        # Min >= max
        with pytest.raises(ValueError, match="input_range min"):
            CVScaler(source, input_range=(1, -1), output_range=(0, 1))

    def test_invalid_output_range_raises_error(self):
        """CVScaler raises ValueError for invalid output range."""
        source = SineOscillator(1, amplitude=1.0, sample_rate=44100)

        # Min >= max
        with pytest.raises(ValueError, match="output_range min"):
            CVScaler(source, input_range=(-1, 1), output_range=(1, 0))

    def test_scaling_factors_precalculated(self):
        """CVScaler pre-calculates scaling factors for efficiency."""
        source = SineOscillator(1, amplitude=1.0, sample_rate=44100)
        scaler = CVScaler(source, input_range=(-1, 1), output_range=(0, 1))

        # Check scale and offset are calculated
        assert hasattr(scaler, "_scale")
        assert hasattr(scaler, "_offset")

        # For [-1, 1] to [0, 1]: scale = 0.5, offset = 0.5
        assert abs(scaler._scale - 0.5) < 0.001
        assert abs(scaler._offset - 0.5) < 0.001


class TestBipolarToUnipolar:
    """Tests for bipolar_to_unipolar helper function."""

    def test_converts_bipolar_to_unipolar(self):
        """bipolar_to_unipolar creates correct CVScaler."""
        source = SineOscillator(1, amplitude=1.0, sample_rate=44100)
        scaler = bipolar_to_unipolar(source)

        # Should be a CVScaler
        assert isinstance(scaler, CVScaler)

        # Check ranges
        assert scaler._in_min == -1.0
        assert scaler._in_max == 1.0
        assert scaler._out_min == 0.0
        assert scaler._out_max == 1.0

        # Verify output
        samples = scaler.get_samples(1000)
        assert np.min(samples) >= 0.0
        assert np.max(samples) <= 1.0

    def test_clamping_enabled_by_default(self):
        """bipolar_to_unipolar enables clamping by default."""
        source = SineOscillator(1, amplitude=2.0, sample_rate=44100)  # Exceeds [-1, 1]
        scaler = bipolar_to_unipolar(source)

        assert scaler._clamp is True

        samples = scaler.get_samples(1000)
        assert np.min(samples) >= 0.0
        assert np.max(samples) <= 1.0


class TestUnipolarToBipolar:
    """Tests for unipolar_to_bipolar helper function."""

    def test_converts_unipolar_to_bipolar(self):
        """unipolar_to_bipolar creates correct CVScaler."""
        source = ADSREnvelope(
            attack_duration=0.1,
            decay_duration=0.1,
            sustain_level=0.7,
            release_duration=0.1,
        )

        scaler = unipolar_to_bipolar(source)

        # Should be a CVScaler
        assert isinstance(scaler, CVScaler)

        # Check ranges
        assert scaler._in_min == 0.0
        assert scaler._in_max == 1.0
        assert scaler._out_min == -1.0
        assert scaler._out_max == 1.0

        # Verify output
        samples = scaler.get_samples(1000)
        assert np.min(samples) >= -1.0
        assert np.max(samples) <= 1.0


class TestScaleCV:
    """Tests for scale_cv general purpose helper function."""

    def test_scales_arbitrary_ranges(self):
        """scale_cv works with arbitrary input and output ranges."""
        source = SineOscillator(1, amplitude=0.3, sample_rate=44100)  # [-0.3, 0.3]

        scaler = scale_cv(source, from_range=(-0.3, 0.3), to_range=(0.4, 0.6))

        # Should be a CVScaler
        assert isinstance(scaler, CVScaler)

        # Check configuration
        assert scaler._in_min == -0.3
        assert scaler._in_max == 0.3
        assert scaler._out_min == 0.4
        assert scaler._out_max == 0.6

        # Verify output
        samples = scaler.get_samples(1000)
        assert np.min(samples) >= 0.39
        assert np.max(samples) <= 0.61


class TestCVScalerIntegration:
    """Integration tests with real modular synth scenarios."""

    def test_lfo_to_volume_modulation(self):
        """LFO [-1, 1] scaled to Volume [0, 1] produces correct modulation."""
        from src.engine import ModulatedVolume

        # LFO outputs [-1, 1]
        lfo = SineOscillator(2, amplitude=1.0, sample_rate=44100)

        # Scale for volume (needs [0, 1])
        scaled_lfo = bipolar_to_unipolar(lfo)

        # Create modulated volume
        volume = ModulatedVolume(scaled_lfo)

        # Generate samples
        samples = [next(volume) for _ in range(1000)]

        # All samples should be in valid amplitude range
        assert all(0.0 <= s <= 1.0 for s in samples)

    def test_envelope_to_panner_modulation(self):
        """Envelope [0, 1] scaled to Panner [-1, 1] produces correct modulation."""
        from src.engine import ModulatedPanner

        # Envelope outputs [0, 1] - starts automatically
        env = ADSREnvelope(
            attack_duration=0.1,
            decay_duration=0.1,
            sustain_level=0.5,
            release_duration=0.1,
        )

        # Scale for panner (needs [-1, 1])
        scaled_env = unipolar_to_bipolar(env)

        # Create modulated panner
        panner = ModulatedPanner(scaled_env)

        # Generate samples (mono input)
        mono_signal = 0.5
        for _ in range(1000):
            left, right = panner(mono_signal)
            # Panning values should be valid
            assert 0.0 <= left <= 0.5
            assert 0.0 <= right <= 0.5

    def test_chained_scalers(self):
        """Multiple scalers can be chained together."""
        source = SineOscillator(1, amplitude=1.0, sample_rate=44100)  # [-1, 1]

        # First scale to [0, 1]
        scaler1 = CVScaler(source, input_range=(-1, 1), output_range=(0, 1))

        # Then scale to [0.25, 0.75]
        scaler2 = CVScaler(scaler1, input_range=(0, 1), output_range=(0.25, 0.75))

        samples = scaler2.get_samples(1000)

        # Final output should be in [0.25, 0.75]
        assert np.min(samples) >= 0.24
        assert np.max(samples) <= 0.76
