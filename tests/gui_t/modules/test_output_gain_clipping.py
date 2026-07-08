"""Test cases for OutputModule click-free gain transitions.

This test verifies that the output module uses the Volume component
for smooth gain transitions that prevent clicking/popping artifacts.
"""

import numpy as np
import pytest

from sonicrack.gui.modules.output.output import OutputModule
from sonicrack.patching.port import Port


@pytest.fixture
def output_module(app):
    """Create an OutputModule instance for testing."""
    return OutputModule()


class TestOutputClickFreeGain:
    """Test that output module uses Volume component for click-free gain."""

    def test_uses_volume_component(self, output_module):
        """Test that output module has Volume component for gain control."""
        assert hasattr(output_module, "volume_component")
        assert output_module.volume_component is not None

        # Volume component should have smoothing
        assert hasattr(output_module.volume_component, "smoothing_time_ms")
        assert output_module.volume_component.smoothing_time_ms > 0

    def test_gain_updates_volume_component(self, output_module):
        """Test that changing gain updates the Volume component."""
        # Set gain to +6 dB
        output_module.master_gain_knob.set_value(6.0)

        # Volume component should be updated
        assert output_module.volume_component.gain_db == pytest.approx(6.0)

        # Set gain to -6 dB
        output_module.master_gain_knob.set_value(-6.0)
        assert output_module.volume_component.gain_db == pytest.approx(-6.0)

    def test_volume_component_processes_audio(self, output_module):
        """Test that Volume component is used to process audio samples."""

        class FakeEngine:
            def render_ports(self, ports, num_samples):
                # Return a test signal
                return [np.full(num_samples, 0.5, dtype=np.float32)]

        output_module.audio_engine = FakeEngine()
        source_port = Port("output", "Out")
        output_module.inp_port_l.connect(source_port)

        # Set gain to +6 dB
        output_module.master_gain_knob.set_value(6.0)

        # Generate samples
        samples = output_module._generate_samples(8)

        # Verify samples are processed (not equal to raw 0.5)
        # Volume component applies gain smoothly
        assert samples.shape == (8, 2)
        assert not np.allclose(samples, 0.5)  # Should be modified by gain

    def test_volume_component_smoothing_prevents_clicks(self, output_module):
        """Test that Volume component smoothing is active for click prevention."""
        # Volume component should use smoothing to prevent clicks
        assert output_module.volume_component.smoothing_time_ms == pytest.approx(10.0)

        # The Volume component has internal smoothing parameters
        assert hasattr(output_module.volume_component, "_amplitude_param")
        assert hasattr(output_module.volume_component, "_gain_db_param")

    def test_sample_rate_updates_volume_component(self, output_module):
        """Test that sample rate changes update Volume component."""
        # Simulate sample rate change
        new_sample_rate = 96000
        output_module._on_global_sample_rate_changed(new_sample_rate)

        # Volume component should be updated
        assert output_module.volume_component.sample_rate == new_sample_rate

    def test_mute_at_minus_80db(self, output_module):
        """Test that -80 dB effectively mutes output."""

        class FakeEngine:
            def render_ports(self, ports, num_samples):
                return [np.full(num_samples, 0.5, dtype=np.float32)]

        output_module.audio_engine = FakeEngine()
        source_port = Port("output", "Out")
        output_module.inp_port_l.connect(source_port)

        # Set gain to -80 dB (effectively mute)
        output_module.master_gain_knob.set_value(-80.0)

        # Give Volume component time to settle (generate enough samples for smoothing)
        samples = output_module._generate_samples(1000)

        # Check the last samples where smoothing has settled
        # Volume component uses ~10ms smoothing, so by the end it should be very quiet
        assert (
            np.max(np.abs(samples[-100:])) < 0.001
        )  # Last 100 samples should be very quiet


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
