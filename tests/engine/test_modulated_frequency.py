"""
Comprehensive tests for ModulatedFrequency class.

Tests cover:
- Basic instantiation
- Default additive frequency modulation
- Custom frequency modulation functions
- Different oscillator types
- Iterator and vectorized modes
- Edge cases
"""

import pytest
import numpy as np
from src.engine.oscillator import (
    SineOscillator,
    SquareOscillator,
    SawtoothOscillator,
    TriangleOscillator,
)
from src.engine.oscillator_modulated import ModulatedFrequency, ModulatedOscillator


class TestModulatedFrequencyBasics:
    """Test basic instantiation and configuration."""

    def test_instantiation_with_sine_oscillators(self):
        """Test creating ModulatedFrequency with sine oscillators."""
        carrier = SineOscillator(frequency=440, amplitude=0.5)
        lfo = SineOscillator(frequency=5.0, amplitude=50.0)

        fm = ModulatedFrequency(carrier, lfo)

        assert isinstance(fm, ModulatedFrequency)
        assert isinstance(fm, ModulatedOscillator)
        assert fm.oscillator is carrier
        assert fm.modulators == (lfo,)

    def test_instantiation_with_different_oscillator_types(self):
        """Test with different oscillator combinations."""
        # Square carrier, sine LFO
        carrier = SquareOscillator(frequency=220)
        lfo = SineOscillator(frequency=3.0, amplitude=30.0)
        fm = ModulatedFrequency(carrier, lfo)
        assert isinstance(fm, ModulatedFrequency)

        # Sawtooth carrier, triangle LFO
        carrier = SawtoothOscillator(frequency=110)
        lfo = TriangleOscillator(frequency=2.0, amplitude=20.0)
        fm = ModulatedFrequency(carrier, lfo)
        assert isinstance(fm, ModulatedFrequency)

    def test_default_freq_mod_function(self):
        """Test that default frequency modulation is additive."""
        carrier = SineOscillator(frequency=440)
        lfo = SineOscillator(frequency=1.0, amplitude=1.0)

        fm = ModulatedFrequency(carrier, lfo)

        # The default should be: new_freq = base_freq + mod_val
        assert fm.freq_mod is not None
        assert fm.amp_mod is None
        assert fm.phase_mod is None

    def test_custom_freq_mod_function(self):
        """Test using a custom frequency modulation function."""
        carrier = SineOscillator(frequency=440)
        lfo = SineOscillator(frequency=1.0)

        # Custom multiplicative modulation
        def custom_mod(base_freq, mod_val):
            return base_freq * (1.0 + 0.1 * mod_val)

        fm = ModulatedFrequency(carrier, lfo, freq_mod_func=custom_mod)

        assert fm.freq_mod is custom_mod

    def test_repr(self):
        """Test string representation."""
        carrier = SineOscillator(frequency=440)
        lfo = SquareOscillator(frequency=5.0)
        fm = ModulatedFrequency(carrier, lfo)

        repr_str = repr(fm)
        assert "ModulatedFrequency" in repr_str
        assert "SineOscillator" in repr_str
        assert "SquareOscillator" in repr_str


class TestModulatedFrequencySampleGeneration:
    """Test sample generation in different modes."""

    def test_generate_samples_iterator(self):
        """Test generating samples using iterator mode."""
        carrier = SineOscillator(frequency=440, amplitude=0.5)
        lfo = SineOscillator(frequency=5.0, amplitude=50.0)
        fm = ModulatedFrequency(carrier, lfo)

        samples = fm.get_samples_iterator(100)

        assert len(samples) == 100
        assert isinstance(samples, np.ndarray)
        assert samples.dtype in [np.float32, np.float64]  # Accept both

    def test_generate_samples_vectorized(self):
        """Test generating samples using vectorized mode."""
        carrier = SineOscillator(frequency=440, amplitude=0.5)
        lfo = SineOscillator(frequency=5.0, amplitude=50.0)
        fm = ModulatedFrequency(carrier, lfo)

        samples = fm.get_samples_vectorized(1000)

        assert len(samples) == 1000
        assert isinstance(samples, np.ndarray)

    def test_generate_samples_auto_mode(self):
        """Test auto mode sample generation."""
        carrier = SineOscillator(frequency=440, amplitude=0.5)
        lfo = SineOscillator(frequency=5.0, amplitude=50.0)
        fm = ModulatedFrequency(carrier, lfo)

        # Small buffer should use iterator
        samples_small = fm.get_samples(100, mode="auto")
        assert len(samples_small) == 100

        # Large buffer should use vectorized
        samples_large = fm.get_samples(2000, mode="auto")
        assert len(samples_large) == 2000

    def test_samples_are_not_constant(self):
        """Verify that frequency modulation creates varying signal."""
        carrier = SineOscillator(frequency=440, amplitude=0.5)
        lfo = SineOscillator(frequency=5.0, amplitude=50.0)
        fm = ModulatedFrequency(carrier, lfo)

        samples = fm.get_samples(1000)

        # Signal should vary (not be constant)
        assert np.std(samples) > 0.01

        # Should have reasonable amplitude
        assert np.max(np.abs(samples)) <= 1.0


class TestModulatedFrequencyModulation:
    """Test frequency modulation behavior."""

    def test_additive_frequency_modulation(self):
        """Test default additive frequency modulation."""
        sample_rate = 44100
        carrier = SineOscillator(frequency=440, amplitude=1.0, sample_rate=sample_rate)
        # LFO at DC (0 Hz) with constant value of 100
        lfo = SineOscillator(frequency=0.0, amplitude=100.0, sample_rate=sample_rate)

        fm = ModulatedFrequency(carrier, lfo)

        # Generate enough samples to see effect
        samples = fm.get_samples(1000)

        # With additive modulation: freq = 440 + lfo_value
        # The frequency should be modulated
        assert len(samples) == 1000
        assert np.max(np.abs(samples)) <= 1.0

    def test_multiplicative_frequency_modulation(self):
        """Test custom multiplicative frequency modulation."""

        def freq_mod_multiply(base_freq, mod_val):
            # Scale -1..+1 to 0.5..1.5
            factor = 0.5 + (mod_val + 1) / 2
            return base_freq * factor

        carrier = SineOscillator(frequency=440, amplitude=0.5)
        lfo = SineOscillator(frequency=3.0, amplitude=1.0)

        fm = ModulatedFrequency(carrier, lfo, freq_mod_func=freq_mod_multiply)

        samples = fm.get_samples(1000)

        assert len(samples) == 1000
        # Verify modulation is happening
        assert np.std(samples) > 0.01

    def test_frequency_modulation_affects_pitch(self):
        """Verify that frequency modulation actually changes the pitch."""
        # Create two signals: one unmodulated, one modulated
        carrier1 = SineOscillator(frequency=440, amplitude=0.5, sample_rate=44100)
        samples_unmod = carrier1.get_samples(4410)  # 0.1 seconds

        carrier2 = SineOscillator(frequency=440, amplitude=0.5, sample_rate=44100)
        lfo = SineOscillator(frequency=10.0, amplitude=100.0, sample_rate=44100)
        fm = ModulatedFrequency(carrier2, lfo)
        samples_mod = fm.get_samples(4410)

        # Modulated signal should have different frequency content
        # Compare FFT magnitudes
        fft_unmod = np.abs(np.fft.rfft(samples_unmod))
        fft_mod = np.abs(np.fft.rfft(samples_mod))

        # They should be different
        assert not np.allclose(fft_unmod, fft_mod, rtol=0.1)

    def test_vibrato_effect(self):
        """Test creating a vibrato effect with slow LFO."""
        # Vibrato: slow, subtle pitch variation
        carrier = SineOscillator(frequency=440, amplitude=0.5, sample_rate=44100)
        lfo = SineOscillator(frequency=5.0, amplitude=10.0, sample_rate=44100)  # ±10 Hz

        fm = ModulatedFrequency(carrier, lfo)
        samples = fm.get_samples(44100)  # 1 second

        assert len(samples) == 44100
        # Should have smooth modulation
        assert np.max(np.abs(samples)) <= 0.5  # Respects carrier amplitude

    def test_wide_pitch_sweep(self):
        """Test creating a wide pitch sweep with multiplicative modulation."""

        def freq_mod_sweep(base_freq, mod_val):
            # Map -1..+1 to 0.5..2.0 (one octave down to one octave up)
            factor = 0.5 + (mod_val + 1) * 0.75
            return base_freq * factor

        carrier = SineOscillator(frequency=440, amplitude=0.3, sample_rate=44100)
        lfo = SawtoothOscillator(frequency=0.5, amplitude=1.0, sample_rate=44100)

        fm = ModulatedFrequency(carrier, lfo, freq_mod_func=freq_mod_sweep)
        samples = fm.get_samples(44100)

        assert len(samples) == 44100
        assert np.std(samples) > 0.01  # Signal varies


class TestModulatedFrequencyEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_frequency_lfo(self):
        """Test with LFO at zero frequency (constant modulation)."""
        carrier = SineOscillator(frequency=440)
        lfo = SineOscillator(frequency=0.0, amplitude=50.0)

        fm = ModulatedFrequency(carrier, lfo)
        samples = fm.get_samples(100)

        assert len(samples) == 100

    def test_very_high_modulation_depth(self):
        """Test with very high modulation depth."""
        carrier = SineOscillator(frequency=440, amplitude=0.5)
        lfo = SineOscillator(frequency=5.0, amplitude=500.0)  # Very high

        fm = ModulatedFrequency(carrier, lfo)
        samples = fm.get_samples(1000)

        assert len(samples) == 1000
        # Should not clip or cause errors
        assert np.all(np.isfinite(samples))

    def test_negative_frequency_handling(self):
        """Test how negative frequencies are handled."""
        carrier = SineOscillator(frequency=100, amplitude=0.5)
        lfo = SineOscillator(frequency=5.0, amplitude=200.0)  # Can make freq negative

        fm = ModulatedFrequency(carrier, lfo)
        samples = fm.get_samples(1000)

        # Should handle gracefully
        assert len(samples) == 1000
        assert np.all(np.isfinite(samples))

    def test_multiple_sample_generations(self):
        """Test generating samples multiple times."""
        carrier = SineOscillator(frequency=440)
        lfo = SineOscillator(frequency=5.0, amplitude=50.0)
        fm = ModulatedFrequency(carrier, lfo)

        samples1 = fm.get_samples(100)
        samples2 = fm.get_samples(100)
        samples3 = fm.get_samples(100)

        assert len(samples1) == 100
        assert len(samples2) == 100
        assert len(samples3) == 100

        # Samples should continue the waveform (not restart)
        # So they should be different
        assert not np.array_equal(samples1, samples2)


class TestModulatedFrequencyIntegration:
    """Integration tests with real-world scenarios."""

    def test_audio_rate_modulation(self):
        """Test audio-rate FM synthesis (classic FM synthesis)."""
        # Classic FM: both carrier and modulator at audio rate
        carrier = SineOscillator(frequency=440, amplitude=0.5, sample_rate=44100)
        modulator = SineOscillator(frequency=220, amplitude=100.0, sample_rate=44100)

        fm = ModulatedFrequency(carrier, modulator)
        samples = fm.get_samples(4410)  # 0.1 seconds

        assert len(samples) == 4410
        # FM synthesis should create complex spectrum
        fft = np.abs(np.fft.rfft(samples))
        # Should have energy at multiple frequencies (sidebands)
        peaks = np.where(fft > np.max(fft) * 0.1)[0]
        assert len(peaks) > 1  # Multiple frequency components

    def test_lfo_modulation_different_waveforms(self):
        """Test LFO with different waveform combinations."""
        test_configs = [
            (SineOscillator, SineOscillator, "Sine + Sine"),
            (SquareOscillator, SineOscillator, "Square + Sine"),
            (SawtoothOscillator, TriangleOscillator, "Saw + Triangle"),
            (TriangleOscillator, SquareOscillator, "Triangle + Square"),
        ]

        for CarrierClass, LFOClass, desc in test_configs:
            carrier = CarrierClass(frequency=220, amplitude=0.5)
            lfo = LFOClass(frequency=3.0, amplitude=30.0)

            fm = ModulatedFrequency(carrier, lfo)
            samples = fm.get_samples(1000)

            assert len(samples) == 1000, f"Failed for {desc}"
            assert np.std(samples) > 0.01, f"Signal constant for {desc}"

    def test_stereo_compatibility(self):
        """Test that ModulatedFrequency works in stereo context."""
        # Create two separate FM oscillators for left/right
        carrier_l = SineOscillator(frequency=440, amplitude=0.5)
        lfo_l = SineOscillator(frequency=5.0, amplitude=50.0)
        fm_l = ModulatedFrequency(carrier_l, lfo_l)

        carrier_r = SineOscillator(frequency=442, amplitude=0.5)  # Slightly detuned
        lfo_r = SineOscillator(frequency=5.1, amplitude=51.0)
        fm_r = ModulatedFrequency(carrier_r, lfo_r)

        samples_l = fm_l.get_samples(1000)
        samples_r = fm_r.get_samples(1000)

        assert len(samples_l) == 1000
        assert len(samples_r) == 1000

        # Stereo image should be slightly different
        assert not np.array_equal(samples_l, samples_r)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
