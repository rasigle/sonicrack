"""
Tests for dB gain control in Oscillator and Volume components.

Verifies that:
1. gain_db and amplitude work correctly
2. Priority rules are followed (gain_db > amplitude)
3. Conversion functions work correctly
4. Warnings are logged when both are specified with mismatched values
"""

import pytest
import numpy as np
import logging
from src.engine.oscillator import SineOscillator, Oscillator
from src.engine.modifier import Volume, ModulatedVolume


class TestOscillatorDBControl:
    """Test dB control in Oscillator class."""

    def test_db_to_linear_conversion(self):
        """Test dB to linear amplitude conversion."""
        assert Oscillator.db_to_linear(0) == pytest.approx(1.0)
        assert Oscillator.db_to_linear(-6) == pytest.approx(0.5, rel=0.01)
        assert Oscillator.db_to_linear(-20) == pytest.approx(0.1)
        assert Oscillator.db_to_linear(6) == pytest.approx(2.0, rel=0.01)
        assert Oscillator.db_to_linear(-40) == pytest.approx(0.01)

    def test_linear_to_db_conversion(self):
        """Test linear to dB conversion."""
        assert Oscillator.linear_to_db(1.0) == pytest.approx(0.0)
        assert Oscillator.linear_to_db(0.5) == pytest.approx(-6.0, rel=0.1)
        assert Oscillator.linear_to_db(0.1) == pytest.approx(-20.0)
        assert Oscillator.linear_to_db(2.0) == pytest.approx(6.0, rel=0.1)
        assert Oscillator.linear_to_db(0.0) == float("-inf")

    def test_gain_db_default(self):
        """Test default gain_db value of -20 dB."""
        osc = SineOscillator()
        assert osc.gain_db == pytest.approx(-20.0)
        assert osc.amplitude == pytest.approx(0.1)

    def test_gain_db_priority_over_amplitude(self):
        """Test that gain_db takes priority over amplitude."""
        osc = SineOscillator(gain_db=-6, amplitude=0.3)
        assert osc.gain_db == pytest.approx(-6.0)
        assert osc.amplitude == pytest.approx(0.5, rel=0.01)

    def test_gain_db_none_uses_amplitude(self):
        """Test that setting gain_db=None uses amplitude instead."""
        osc = SineOscillator(amplitude=0.7, gain_db=None)
        assert osc.amplitude == pytest.approx(0.7)
        assert osc.gain_db == pytest.approx(-3.1, rel=0.1)

    def test_gain_db_property_setter(self):
        """Test setting gain_db property after initialization."""
        osc = SineOscillator(gain_db=-20)
        osc.gain_db = -12
        assert osc.gain_db == pytest.approx(-12.0)
        assert osc.amplitude == pytest.approx(0.25, rel=0.01)

        osc.gain_db = 0
        assert osc.amplitude == pytest.approx(1.0)

    def test_amplitude_property_updates_gain_db(self):
        """Test that setting amplitude updates gain_db."""
        osc = SineOscillator(gain_db=None, amplitude=1.0)
        osc.amplitude = 0.5
        assert osc.gain_db == pytest.approx(-6.0, rel=0.1)

    def test_warning_on_mismatched_gain_and_amplitude(self, caplog):
        """Test that warning is logged when gain_db and amplitude don't match."""
        with caplog.at_level(logging.WARNING):
            SineOscillator(gain_db=-6, amplitude=0.3)
            assert "Both gain_db" in caplog.text
            assert "Using gain_db" in caplog.text

    def test_no_warning_when_amplitude_is_default(self, caplog):
        """Test no warning when amplitude is at default value."""
        with caplog.at_level(logging.WARNING):
            SineOscillator(gain_db=-6)  # amplitude defaults to 1.0
            # Should not warn because amplitude wasn't explicitly set
            assert "Both gain_db" not in caplog.text

    def test_output_amplitude_with_gain_db(self):
        """Test that output amplitude matches gain_db setting."""
        osc = SineOscillator(frequency=440, gain_db=-20)
        samples = osc.get_samples_vectorized(1000)

        # Check that max amplitude is approximately 0.1 (=-20 dB)
        max_amp = np.max(np.abs(samples))
        assert max_amp == pytest.approx(0.1, rel=0.01)

    def test_output_amplitude_with_linear(self):
        """Test output with linear amplitude (gain_db=None)."""
        osc = SineOscillator(frequency=440, amplitude=0.5, gain_db=None)
        samples = osc.get_samples_vectorized(1000)

        max_amp = np.max(np.abs(samples))
        assert max_amp == pytest.approx(0.5, rel=0.01)


class TestVolumeDBControl:
    """Test dB control in Volume class."""

    def test_volume_db_to_linear(self):
        """Test Volume's dB conversion."""
        assert Volume.db_to_linear(0) == pytest.approx(1.0)
        assert Volume.db_to_linear(-6) == pytest.approx(0.5, rel=0.01)

    def test_volume_default_amplitude(self):
        """Test Volume default is 1.0 (unity gain)."""
        vol = Volume()
        assert vol.amplitude == 1.0
        assert vol.gain_db == pytest.approx(0.0)

    def test_volume_with_gain_db(self):
        """Test Volume initialization with gain_db."""
        vol = Volume(gain_db=-6)
        assert vol.gain_db == pytest.approx(-6.0)
        assert vol.amplitude == pytest.approx(0.5, rel=0.01)

    def test_volume_gain_db_priority(self):
        """Test that gain_db takes priority in Volume."""
        vol = Volume(gain_db=-12, amplitude=0.7)
        assert vol.gain_db == pytest.approx(-12.0)
        assert vol.amplitude == pytest.approx(0.25, rel=0.01)

    def test_volume_gain_db_property(self):
        """Test Volume gain_db property setter."""
        vol = Volume(amplitude=1.0)
        vol.gain_db = -20
        assert vol.amplitude == pytest.approx(0.1)

    def test_volume_amplitude_property(self):
        """Test Volume amplitude property setter."""
        vol = Volume(gain_db=-6)
        vol.amplitude = 0.8
        assert vol.gain_db == pytest.approx(-1.94, rel=0.1)

    def test_volume_call_with_gain_db(self):
        """Test Volume scaling with dB control."""
        vol = Volume(gain_db=-6)  # Half amplitude
        result = vol(1.0)
        assert result == pytest.approx(0.5, rel=0.01)

    def test_volume_vectorized_with_gain_db(self):
        """Test Volume vectorized scaling with dB."""
        vol = Volume(gain_db=-20)  # 0.1 amplitude
        samples = np.ones(100)
        result = vol.scale_vectorized(samples)
        assert np.all(result == pytest.approx(0.1))

    def test_volume_warning_on_mismatch(self, caplog):
        """Test Volume logs warning on gain_db/amplitude mismatch."""
        with caplog.at_level(logging.WARNING):
            Volume(gain_db=-6, amplitude=0.3)
            assert "Both gain_db" in caplog.text


class TestModulatedVolumeInheritance:
    """Test that ModulatedVolume inherits dB functionality."""

    def test_modulated_volume_has_gain_db(self):
        """Test ModulatedVolume has gain_db property from parent."""
        from src.engine.oscillator import SineOscillator

        lfo = SineOscillator(frequency=5, amplitude=0.5, gain_db=None)
        mod_vol = ModulatedVolume(lfo)

        # Should have gain_db property
        assert hasattr(mod_vol, "gain_db")
        assert hasattr(mod_vol, "db_to_linear")
        assert hasattr(mod_vol, "linear_to_db")


class TestWaveRangeWithGainDB:
    """Test interaction between wave_range, amplitude, and gain_db."""

    def test_wave_range_with_gain_db(self):
        """Test that wave_range and gain_db work together correctly."""
        # wave_range affects raw waveform output range
        # gain_db affects final amplitude scaling
        osc = SineOscillator(
            frequency=440,
            gain_db=-6,  # Final output halved
            wave_range=(-1, 1),  # Standard range
        )
        samples = osc.get_samples_vectorized(1000)

        # Should be scaled by -6 dB (approx 0.5)
        max_amp = np.max(np.abs(samples))
        assert max_amp == pytest.approx(0.5, rel=0.05)

    def test_wave_range_default_with_db(self):
        """Test default wave_range (-1, 1) with dB control."""
        osc = SineOscillator(frequency=440, gain_db=-20)
        samples = osc.get_samples_vectorized(1000)

        max_amp = np.max(np.abs(samples))
        assert max_amp == pytest.approx(0.1, rel=0.01)


class TestBackwardsCompatibility:
    """Test that existing code without gain_db still works."""

    def test_oscillator_amplitude_only(self):
        """Test Oscillator with only amplitude (old style)."""
        osc = SineOscillator(frequency=440, amplitude=0.5, gain_db=None)
        assert osc.amplitude == 0.5

        samples = osc.get_samples_vectorized(1000)
        assert np.max(np.abs(samples)) == pytest.approx(0.5, rel=0.01)

    def test_volume_amplitude_only(self):
        """Test Volume with only amplitude (old style)."""
        vol = Volume(amplitude=0.7)
        assert vol.amplitude == 0.7

        result = vol(1.0)
        assert result == pytest.approx(0.7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
