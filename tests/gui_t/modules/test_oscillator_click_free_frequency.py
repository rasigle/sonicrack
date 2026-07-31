"""Test cases for click-free frequency transitions in oscillator modules.

This test verifies that oscillators use proper frequency slewing to prevent
clicking/popping artifacts when frequency changes, and that the LFO (backed by
soniclab.LFO) keeps phase-continuous output across rate changes.
"""

import numpy as np
import pytest

from sonicrack.gui.modules.source.lfo import LFOModule
from sonicrack.gui.modules.source.oscillator import OscillatorModule


@pytest.fixture
def lfo_module(app):
    """Create an LFO module for testing."""
    return LFOModule()


@pytest.fixture
def osc_module(app):
    """Create an Oscillator module for testing."""
    return OscillatorModule()


class TestLFOPhaseContinuousFrequency:
    """Test that LFO rate changes stay phase-continuous (no hard jumps)."""

    def test_lfo_uses_soniclab_lfo(self, lfo_module):
        """LFO module is backed by soniclab.LFO instances."""
        from soniclab.dsp.modulators import LFO

        assert len(lfo_module.lfos) == 4
        assert all(isinstance(lfo, LFO) for lfo in lfo_module.lfos)

    def test_lfo_frequency_changes_are_smooth(self, lfo_module):
        """Test that LFO frequency changes produce smooth output transitions."""
        # Start at 1 Hz
        lfo_module.process_runtime(256, {"frequency": 1.0, "pulsewidth": 0.5})
        first_buffer = np.asarray(lfo_module.sine_port.value)

        # Change to 5 Hz
        lfo_module.process_runtime(256, {"frequency": 5.0, "pulsewidth": 0.5})
        second_buffer = np.asarray(lfo_module.sine_port.value)

        # Phase-continuous rate change: sample values stay continuous.
        transition_jump = abs(second_buffer[0] - first_buffer[-1])
        assert transition_jump < 0.3  # Reasonable continuity threshold

    def test_lfo_tracks_rate_on_all_instances(self, lfo_module):
        """All shape instances receive the same free-running rate."""
        lfo_module.process_runtime(64, {"frequency": 3.5, "pulsewidth": 0.5})
        for lfo in lfo_module.lfos:
            assert lfo.rate_hz == pytest.approx(3.5)


class TestOscillatorClickFreeFrequency:
    """Test that regular Oscillator uses frequency slewing for click-free
    transitions."""

    def test_oscillator_uses_frequency_slewing(self, osc_module):
        """Test that Oscillator uses the render_with_frequency_ramp function."""
        from sonicrack.gui.modules.source._oscillator_runtime import (
            DEFAULT_FREQUENCY_SLEW_TIME_MS,
        )

        # Should have a substantial slew time to prevent clicks
        assert DEFAULT_FREQUENCY_SLEW_TIME_MS > 50.0  # At least 50ms
        assert isinstance(DEFAULT_FREQUENCY_SLEW_TIME_MS, float)

    def test_oscillator_frequency_changes_are_smooth(self, osc_module):
        """Test that Oscillator frequency changes produce smooth output transitions."""
        # Start at 200 Hz
        osc_module.process_runtime(256, {"frequency": 200.0, "pulsewidth": 0.5})
        first_buffer = np.asarray(osc_module.sine_port.value)

        # Change to 400 Hz (one octave up)
        osc_module.process_runtime(256, {"frequency": 400.0, "pulsewidth": 0.5})
        second_buffer = np.asarray(osc_module.sine_port.value)

        # The transition between buffers should be smooth
        # For audio oscillators, we expect continuity
        transition_jump = abs(second_buffer[0] - first_buffer[-1])
        assert transition_jump < 0.3  # Reasonable continuity threshold

    def test_oscillator_tracks_last_frequency(self, osc_module):
        """Test that Oscillator tracks the last rendered frequency for continuity."""
        assert hasattr(osc_module, "_last_runtime_frequency")
        assert isinstance(osc_module._last_runtime_frequency, (int, float))


class TestFrequencySlewingMechanism:
    """Test the underlying frequency slewing mechanism."""

    def test_frequency_slewing_in_log_space(self):
        """Test that frequency slewing happens in logarithmic (pitch) space."""
        from sonicrack.gui.modules.source._oscillator_runtime import (
            _frequency_slew_values,
        )

        # Slew from 100 Hz to 200 Hz (one octave)
        # Use enough samples to reach the target (50ms at 44100 Hz = 2205 samples)
        frequencies = _frequency_slew_values(
            previous_frequency=100.0,
            target_frequency=200.0,
            num_samples=2500,  # About 57ms at 44100 Hz
            sample_rate=44100.0,
            slew_time_ms=50.0,
        )

        assert len(frequencies) == 2500
        # First value should be close to start
        assert abs(frequencies[0] - 100.0) < 5.0
        # Should be moving toward target
        assert frequencies[-1] > 150.0  # Should be more than halfway
        # Should be monotonically increasing
        assert np.all(np.diff(frequencies) >= -1e-6)  # Allow tiny floating point errors

    def test_zero_slew_time_changes_immediately(self):
        """Test that zero slew time changes frequency immediately."""
        from sonicrack.gui.modules.source._oscillator_runtime import (
            _frequency_slew_values,
        )

        frequencies = _frequency_slew_values(
            previous_frequency=100.0,
            target_frequency=200.0,
            num_samples=100,
            sample_rate=44100.0,
            slew_time_ms=0.0,  # No slewing
        )

        # All values should be at target
        assert np.allclose(frequencies, 200.0)

    def test_equal_frequencies_no_slewing(self):
        """Test that equal start and target frequencies produce constant output."""
        from sonicrack.gui.modules.source._oscillator_runtime import (
            _frequency_slew_values,
        )

        frequencies = _frequency_slew_values(
            previous_frequency=440.0,
            target_frequency=440.0,
            num_samples=100,
            sample_rate=44100.0,
            slew_time_ms=50.0,
        )

        # All values should be at target (no slewing needed)
        assert np.allclose(frequencies, 440.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
