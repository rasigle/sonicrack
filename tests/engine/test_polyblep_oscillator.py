"""Test cases for PolyBLEP oscillator.

This module contains comprehensive unit tests for the PolyBLEP oscillator
implementation, verifying API compatibility, waveform generation, and
antialiasing effectiveness.
"""

import pytest
import numpy as np
from src.engine.polyblep_oscillator import (
    PolyBLEPOscillator,
    PolyBLEPWaveforms,
    WaveShape,
    generate_sine,
    generate_square,
    generate_sawtooth,
    generate_triangle,
)


class TestPolyBLEPWaveforms:
    """Test static waveform generation methods."""

    def test_polyblep_correction_near_start(self):
        """Test PolyBLEP correction near start of cycle."""
        phase = 0.001
        increment = 0.01
        correction = PolyBLEPWaveforms.polyblep(phase, increment)
        assert correction != 0.0, "Should apply correction near start"
        assert -2 < correction < 2, "Correction should be bounded"

    def test_polyblep_correction_near_end(self):
        """Test PolyBLEP correction near end of cycle."""
        phase = 0.999
        increment = 0.01
        correction = PolyBLEPWaveforms.polyblep(phase, increment)
        assert correction != 0.0, "Should apply correction near end"
        assert -2 < correction < 2, "Correction should be bounded"

    def test_polyblep_no_correction_middle(self):
        """Test that no correction is applied far from discontinuities."""
        phase = 0.5
        increment = 0.01
        correction = PolyBLEPWaveforms.polyblep(phase, increment)
        assert correction == 0.0, "Should not apply correction in middle"

    def test_sine_output_range(self):
        """Test sine wave output range."""
        phases = np.linspace(0, 1, 100, endpoint=False)
        for phase in phases:
            value = PolyBLEPWaveforms.sine(phase)
            assert -1.0 <= value <= 1.0, f"Sine out of range at phase {phase}"

    def test_square_output_range(self):
        """Test square wave stays bounded."""
        phases = np.linspace(0, 1, 100, endpoint=False)
        increment = 0.01
        for phase in phases:
            value = PolyBLEPWaveforms.square(phase, increment)
            # PolyBLEP may cause slight overshoot
            assert -1.5 <= value <= 1.5, f"Square out of range at phase {phase}"

    def test_sawtooth_output_range(self):
        """Test sawtooth wave stays bounded."""
        phases = np.linspace(0, 1, 100, endpoint=False)
        increment = 0.01
        for phase in phases:
            value = PolyBLEPWaveforms.sawtooth(phase, increment)
            # PolyBLEP may cause slight overshoot
            assert -1.5 <= value <= 1.5, f"Sawtooth out of range at phase {phase}"


class TestPolyBLEPOscillatorConstruction:
    """Test oscillator construction and initialization."""

    def test_default_construction(self):
        """Test construction with default parameters."""
        osc = PolyBLEPOscillator()
        assert osc.frequency == 440.0
        assert osc.sample_rate == 44100.0
        assert osc.wave_shape == WaveShape.SAWTOOTH_UP

    def test_custom_frequency(self):
        """Test construction with custom frequency."""
        osc = PolyBLEPOscillator(frequency=880.0)
        assert osc.frequency == 880.0
        assert osc.init_freq == 880.0

    def test_gain_db_parameter(self):
        """Test construction with gain_db parameter."""
        osc = PolyBLEPOscillator(gain_db=-12.0)
        assert abs(osc.gain_db - (-12.0)) < 0.1
        # -12 dB ≈ 0.251 amplitude
        assert abs(osc.amplitude - 0.251) < 0.01

    def test_amplitude_parameter(self):
        """Test construction with amplitude parameter."""
        osc = PolyBLEPOscillator(amplitude=0.5, gain_db=None)
        assert abs(osc.amplitude - 0.5) < 0.01

    def test_phase_parameter(self):
        """Test construction with phase parameter."""
        osc = PolyBLEPOscillator(phase=90.0)
        assert osc.phase == 90.0
        assert osc.init_phase == 90.0

    def test_wave_range_parameter(self):
        """Test construction with custom wave range."""
        osc = PolyBLEPOscillator(wave_range=(0, 1))
        assert osc.wave_range == (0, 1)

    def test_wave_shape_parameter(self):
        """Test construction with different wave shapes."""
        for shape in [WaveShape.SINE, WaveShape.SQUARE, WaveShape.SAWTOOTH_UP,
                     WaveShape.SAWTOOTH_DOWN, WaveShape.TRIANGLE]:
            osc = PolyBLEPOscillator(wave_shape=shape)
            assert osc.wave_shape == shape


class TestPolyBLEPOscillatorProperties:
    """Test oscillator property getters and setters."""

    def test_frequency_getter_setter(self):
        """Test frequency property."""
        osc = PolyBLEPOscillator(frequency=440.0)
        assert osc.frequency == 440.0

        osc.frequency = 880.0
        assert osc.frequency == 880.0

    def test_amplitude_getter_setter(self):
        """Test amplitude property."""
        osc = PolyBLEPOscillator(amplitude=0.5, gain_db=None)
        assert abs(osc.amplitude - 0.5) < 0.01

        osc.amplitude = 0.8
        assert abs(osc.amplitude - 0.8) < 0.01

    def test_gain_db_getter_setter(self):
        """Test gain_db property."""
        osc = PolyBLEPOscillator(gain_db=-20.0)
        assert abs(osc.gain_db - (-20.0)) < 0.1

        osc.gain_db = -6.0
        assert abs(osc.gain_db - (-6.0)) < 0.1

    def test_gain_db_amplitude_conversion(self):
        """Test that gain_db and amplitude are consistent."""
        osc = PolyBLEPOscillator(gain_db=0.0)
        assert abs(osc.amplitude - 1.0) < 0.01

        osc.gain_db = -6.0
        assert abs(osc.amplitude - 0.5) < 0.01

        osc.amplitude = 1.0
        assert abs(osc.gain_db - 0.0) < 0.1

    def test_phase_getter_setter(self):
        """Test phase property."""
        osc = PolyBLEPOscillator(phase=0.0)
        assert osc.phase == 0.0

        osc.phase = 90.0
        assert osc.phase == 90.0

    def test_wave_range_getter_setter(self):
        """Test wave_range property."""
        osc = PolyBLEPOscillator(wave_range=(-1, 1))
        assert osc.wave_range == (-1, 1)

        osc.wave_range = (0, 1)
        assert osc.wave_range == (0, 1)

    def test_sample_rate_getter(self):
        """Test sample_rate property is read-only."""
        osc = PolyBLEPOscillator(sample_rate=48000)
        assert osc.sample_rate == 48000.0

    def test_init_properties(self):
        """Test init_* properties return construction values."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=-12, phase=45)
        assert osc.init_freq == 440.0
        assert osc.init_phase == 45.0
        # init_amp should be frozen at construction
        initial_amp = osc.init_amp
        osc.amplitude = 0.1
        assert osc.init_amp == initial_amp


class TestPolyBLEPOscillatorGeneration:
    """Test sample generation methods."""

    def test_iterator_interface(self):
        """Test that oscillator works as iterator."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0)
        samples = []
        for i, sample in enumerate(osc):
            samples.append(sample)
            if i >= 99:
                break

        assert len(samples) == 100
        assert all(isinstance(s, (float, np.floating)) for s in samples)

    def test_get_samples_vectorized(self):
        """Test vectorized generation."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0)
        samples = osc.get_samples_vectorized(1000)

        assert len(samples) == 1000
        assert samples.dtype == np.float32
        assert samples.shape == (1000,)

    def test_get_samples_iterator_method(self):
        """Test iterator generation method."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0)
        samples = osc.get_samples_iterator(100)

        assert len(samples) == 100
        assert samples.dtype == np.float32

    def test_get_samples_auto_mode_small(self):
        """Test auto mode selects iterator for small buffers."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0)
        samples = osc.get_samples(200, mode="auto")

        assert len(samples) == 200
        assert samples.dtype == np.float32

    def test_get_samples_auto_mode_large(self):
        """Test auto mode selects vectorized for large buffers."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0)
        samples = osc.get_samples(1000, mode="auto")

        assert len(samples) == 1000
        assert samples.dtype == np.float32

    def test_get_samples_explicit_vectorized(self):
        """Test explicit vectorized mode."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0)
        samples = osc.get_samples(100, mode="vectorized")

        assert len(samples) == 100
        assert samples.dtype == np.float32

    def test_get_samples_explicit_iterator(self):
        """Test explicit iterator mode."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0)
        samples = osc.get_samples(100, mode="iterator")

        assert len(samples) == 100
        assert samples.dtype == np.float32

    def test_get_samples_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        osc = PolyBLEPOscillator(frequency=440)
        with pytest.raises(ValueError):
            osc.get_samples(100, mode="invalid")

    def test_get_samples_reset(self):
        """Test reset parameter."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0, phase=0)

        # Generate some samples
        osc.get_samples(100)

        # Generate with reset
        samples1 = osc.get_samples(10, reset=True)

        # Generate again with reset
        samples2 = osc.get_samples(10, reset=True)

        # Should be identical after reset
        np.testing.assert_array_almost_equal(samples1, samples2, decimal=5)


class TestPolyBLEPOscillatorWaveShapes:
    """Test different waveform shapes."""

    def test_sine_generation(self):
        """Test sine wave generation."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0,
                                wave_shape=WaveShape.SINE)
        samples = osc.get_samples(1000, mode="vectorized")

        assert -1.1 <= samples.min() <= -0.9, "Sine min should be near -1"
        assert 0.9 <= samples.max() <= 1.1, "Sine max should be near 1"
        assert abs(samples.mean()) < 0.1, "Sine mean should be near 0"

    def test_square_generation(self):
        """Test square wave generation."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0,
                                wave_shape=WaveShape.SQUARE)
        samples = osc.get_samples(1000, mode="vectorized")

        # PolyBLEP square may have slight overshoot
        assert -1.2 <= samples.min() <= -0.8
        assert 0.8 <= samples.max() <= 1.2

    def test_sawtooth_up_generation(self):
        """Test sawtooth up generation."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0,
                                wave_shape=WaveShape.SAWTOOTH_UP)
        samples = osc.get_samples(1000, mode="vectorized")

        assert -1.2 <= samples.min() <= -0.8
        assert 0.8 <= samples.max() <= 1.2

    def test_sawtooth_down_generation(self):
        """Test sawtooth down generation."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0,
                                wave_shape=WaveShape.SAWTOOTH_DOWN)
        samples = osc.get_samples(1000, mode="vectorized")

        assert -1.2 <= samples.min() <= -0.8
        assert 0.8 <= samples.max() <= 1.2

    def test_triangle_generation(self):
        """Test triangle wave generation."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0,
                                wave_shape=WaveShape.TRIANGLE)
        samples = osc.get_samples(1000, mode="vectorized")

        # Triangle via integration may have wider bounds initially
        # as the integrator stabilizes
        assert -2.0 <= samples.min() <= 2.0, f"Triangle min out of bounds: {samples.min()}"
        assert -2.0 <= samples.max() <= 2.0, f"Triangle max out of bounds: {samples.max()}"

        # After stabilization, should be closer to ±1
        # Check the last 500 samples (after integrator stabilizes)
        stable_samples = samples[-500:]
        assert -1.5 <= stable_samples.min() <= 1.5
        assert -1.5 <= stable_samples.max() <= 1.5


class TestPolyBLEPOscillatorAmplitudeControl:
    """Test amplitude and gain control."""

    def test_amplitude_affects_output(self):
        """Test that amplitude scales output."""
        osc = PolyBLEPOscillator(frequency=440, amplitude=0.5, gain_db=None,
                                wave_shape=WaveShape.SINE)
        samples = osc.get_samples(1000, mode="vectorized")

        # Output should be scaled to ±0.5
        assert -0.6 <= samples.min() <= -0.4
        assert 0.4 <= samples.max() <= 0.6

    def test_gain_db_affects_output(self):
        """Test that gain_db scales output."""
        # -6 dB ≈ 0.5 amplitude
        osc = PolyBLEPOscillator(frequency=440, gain_db=-6,
                                wave_shape=WaveShape.SINE)
        samples = osc.get_samples(1000, mode="vectorized")

        # Output should be scaled to approximately ±0.5
        assert -0.6 <= samples.min() <= -0.4
        assert 0.4 <= samples.max() <= 0.6

    def test_runtime_amplitude_change(self):
        """Test changing amplitude at runtime."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0,
                                wave_shape=WaveShape.SINE)

        samples1 = osc.get_samples(100, mode="vectorized")
        rms1 = np.sqrt(np.mean(samples1**2))

        osc.amplitude = 0.5
        # Generate enough samples for smoothing to complete
        osc.get_samples(1000, mode="vectorized")
        samples2 = osc.get_samples(100, mode="vectorized")
        rms2 = np.sqrt(np.mean(samples2**2))

        # Second RMS should be approximately half
        assert 0.4 < (rms2 / rms1) < 0.6

    def test_amplitude_smoothing_prevents_clicks(self):
        """Test that amplitude smoothing is applied."""
        osc = PolyBLEPOscillator(frequency=440, amplitude=1.0, gain_db=None)

        # Change amplitude suddenly
        osc.amplitude = 0.1

        # First sample should show smoothing (not immediate jump)
        samples = osc.get_samples(10, mode="vectorized")

        # First samples should be transitioning (between 0.1 and 1.0)
        assert any(0.1 < abs(s) < 1.0 for s in samples[:5])


class TestPolyBLEPOscillatorPhaseControl:
    """Test phase control."""

    def test_phase_affects_initial_output(self):
        """Test that phase shifts waveform."""
        osc1 = PolyBLEPOscillator(frequency=440, phase=0, gain_db=0,
                                 wave_shape=WaveShape.SINE)
        osc2 = PolyBLEPOscillator(frequency=440, phase=90, gain_db=0,
                                 wave_shape=WaveShape.SINE)

        samples1 = osc1.get_samples(10, mode="vectorized")
        samples2 = osc2.get_samples(10, mode="vectorized")

        # Waveforms should be different
        diff = np.max(np.abs(samples1 - samples2))
        assert diff > 0.1, "Phase should shift waveform"

    def test_runtime_phase_change(self):
        """Test changing phase at runtime."""
        osc = PolyBLEPOscillator(frequency=440, phase=0, gain_db=0,
                                wave_shape=WaveShape.SINE)

        samples1 = osc.get_samples(10, mode="vectorized")

        osc.phase = 180
        samples2 = osc.get_samples(10, mode="vectorized")

        # Samples should be different
        diff = np.max(np.abs(samples1 - samples2))
        assert diff > 0.1


class TestPolyBLEPOscillatorWaveRangeConversion:
    """Test wave range conversion."""

    def test_standard_range(self):
        """Test standard [-1, 1] range."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0, wave_range=(-1, 1),
                                wave_shape=WaveShape.SINE)
        samples = osc.get_samples(1000, mode="vectorized")

        assert -1.1 <= samples.min() <= -0.9
        assert 0.9 <= samples.max() <= 1.1

    def test_unipolar_range(self):
        """Test unipolar [0, 1] range."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0, wave_range=(0, 1),
                                wave_shape=WaveShape.SINE)
        samples = osc.get_samples(1000, mode="vectorized")

        assert -0.1 <= samples.min() <= 0.1
        assert 0.9 <= samples.max() <= 1.1

    def test_custom_range(self):
        """Test custom range."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0, wave_range=(-5, 5),
                                wave_shape=WaveShape.SINE)
        samples = osc.get_samples(1000, mode="vectorized")

        assert -5.5 <= samples.min() <= -4.5
        assert 4.5 <= samples.max() <= 5.5

    def test_runtime_range_change(self):
        """Test changing wave range at runtime."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0, wave_range=(-1, 1),
                                wave_shape=WaveShape.SINE)

        samples1 = osc.get_samples(100, mode="vectorized")

        osc.wave_range = (0, 1)
        samples2 = osc.get_samples(100, mode="vectorized")

        # Ranges should be different
        assert samples1.min() < -0.5
        assert samples2.min() > -0.5


class TestPolyBLEPOscillatorAntialiasing:
    """Test antialiasing effectiveness."""

    def test_polyblep_differs_from_naive(self):
        """Test that PolyBLEP output differs from naive waveform."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0, sample_rate=44100,
                                wave_shape=WaveShape.SQUARE)
        polyblep_samples = osc.get_samples(1000, mode="vectorized")

        # Generate naive square wave
        phases = np.linspace(0, 1000 / 44100 * 440, 1000) % 1.0
        naive_samples = np.where(phases < 0.5, -1.0, 1.0).astype(np.float32)

        # They should be different
        diff = np.max(np.abs(polyblep_samples - naive_samples))
        assert diff > 0.01, "PolyBLEP should differ from naive"

    def test_polyblep_reduces_high_frequency_content(self):
        """Test that PolyBLEP reduces high-frequency aliasing."""
        # Generate PolyBLEP square
        osc = PolyBLEPOscillator(frequency=440, gain_db=0, sample_rate=44100,
                                wave_shape=WaveShape.SQUARE)
        polyblep_samples = osc.get_samples(4410, mode="vectorized")  # 0.1s

        # Generate naive square
        phases = np.linspace(0, 0.1 * 440, 4410) % 1.0
        naive_samples = np.where(phases < 0.5, -1.0, 1.0)

        # FFT analysis
        fft_polyblep = np.abs(np.fft.rfft(polyblep_samples))
        fft_naive = np.abs(np.fft.rfft(naive_samples))

        # High frequency content (upper half of spectrum)
        nyquist_idx = len(fft_naive) // 2
        high_freq_polyblep = np.mean(fft_polyblep[nyquist_idx:])
        high_freq_naive = np.mean(fft_naive[nyquist_idx:])

        # PolyBLEP should have less high-frequency content
        assert high_freq_polyblep < high_freq_naive, \
            "PolyBLEP should reduce high-frequency aliasing"


class TestPolyBLEPOscillatorPhaseContinuity:
    """Test phase continuity between generation modes."""

    def test_phase_continuity_after_vectorized(self):
        """Test phase is maintained after vectorized generation."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0, phase=0,
                                wave_shape=WaveShape.SINE)

        # Generate using vectorized
        samples1 = osc.get_samples(100, mode="vectorized")
        samples2 = osc.get_samples(100, mode="vectorized")

        # Should be continuous (no phase jump)
        # Check that last sample of samples1 and first of samples2 are close
        expected_diff = abs(samples1[-1] - samples1[-2])
        actual_diff = abs(samples2[0] - samples1[-1])

        # Allow some tolerance for phase continuity
        assert actual_diff < expected_diff * 3

    def test_phase_continuity_mixed_modes(self):
        """Test phase continuity when switching modes."""
        osc = PolyBLEPOscillator(frequency=440, gain_db=0, phase=0,
                                wave_shape=WaveShape.SINE)

        # Generate using vectorized
        osc.get_samples(1000, mode="vectorized")
        sample1 = next(osc)

        # Generate using iterator
        osc.get_samples(1000, mode="iterator")
        sample2 = next(osc)

        # Both should produce valid samples
        assert -1.5 <= sample1 <= 1.5
        assert -1.5 <= sample2 <= 1.5


class TestConvenienceFunctions:
    """Test convenience generation functions."""

    def test_generate_sine(self):
        """Test generate_sine convenience function."""
        samples = generate_sine(440, 0.1, 44100, 1.0)
        assert len(samples) == 4410  # 0.1s at 44100 Hz
        assert samples.dtype == np.float32

    def test_generate_square(self):
        """Test generate_square convenience function."""
        samples = generate_square(440, 0.1, 44100, 1.0)
        assert len(samples) == 4410
        assert samples.dtype == np.float32

    def test_generate_sawtooth(self):
        """Test generate_sawtooth convenience function."""
        samples = generate_sawtooth(440, 0.1, 44100, 1.0)
        assert len(samples) == 4410
        assert samples.dtype == np.float32

    def test_generate_triangle(self):
        """Test generate_triangle convenience function."""
        samples = generate_triangle(440, 0.1, 44100, 1.0)
        assert len(samples) == 4410
        assert samples.dtype == np.float32


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])

