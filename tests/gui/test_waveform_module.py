"""Test for the Waveform visualization module."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.gui.modules.visualization.waveform import WaveformModule
from src.gui.audio_module_interface import ModuleCategory


class TestWaveformModule:
    """Tests for WaveformModule."""

    def test_module_metadata(self):
        """Test that module has correct metadata."""
        assert WaveformModule.metadata.title == "Waveform"
        assert WaveformModule.metadata.category == ModuleCategory.VISUALIZATION
        assert "visualization" in WaveformModule.metadata.description.lower()

    def test_module_creation(self):
        """Test creating a waveform module."""
        module = WaveformModule()
        assert module is not None
        assert hasattr(module, "in_port")
        assert hasattr(module, "waveform_display")
        assert hasattr(module, "min_label")
        assert hasattr(module, "max_label")
        assert hasattr(module, "update_timer")

    def test_timer_auto_starts(self):
        """Test that timer auto-starts on creation."""
        module = WaveformModule()
        assert module.update_timer.isActive()
        assert module.update_timer.interval() == 50  # 50ms

    def test_required_inputs(self):
        """Test that module requires In port."""
        module = WaveformModule()
        required = module.get_required_inputs()
        assert "In" in required

    def test_pass_through_behavior(self):
        """Test that module passes through the input component."""
        module = WaveformModule()

        # Create a mock audio component
        mock_component = MagicMock()
        mock_component.get_samples.return_value = np.zeros(1024)

        # Create engine component with mock input
        result = module.create_engine_component(input_components=[mock_component])

        # Should return the input component (pass-through)
        assert result is mock_component
        assert module.input_component is mock_component

    def test_get_output_component(self):
        """Test getting output component."""
        module = WaveformModule()

        # Create a mock audio component
        mock_component = MagicMock()

        # Set up the module
        module.create_engine_component(input_components=[mock_component])

        # Get output should return the same component
        output = module.get_output_component("Out")
        assert output is mock_component

    def test_cleanup_stops_timer(self):
        """Test cleanup stops timer."""
        module = WaveformModule()

        # Timer should be active
        assert module.update_timer.isActive()

        # Cleanup should stop it
        module.cleanup()
        assert not module.update_timer.isActive()

    def test_update_display_with_no_component(self):
        """Test that update doesn't crash with no input component."""
        module = WaveformModule()

        # Should not crash
        module._update_display()

    def test_update_display_with_component(self):
        """Test updating display with audio component."""
        module = WaveformModule()

        # Create a mock audio component
        mock_component = MagicMock()
        test_samples = np.random.randn(2048) * 0.5
        mock_component.get_samples.return_value = test_samples

        # Set up the module
        module.create_engine_component(input_components=[mock_component])

        # Update display
        module._update_display()

        # Should have called get_samples
        mock_component.get_samples.assert_called_once_with(2048)

    def test_min_max_display(self):
        """Test that min/max values are displayed correctly."""
        module = WaveformModule()

        # Create a mock audio component with known values
        mock_component = MagicMock()
        test_samples = np.array([0.5, -0.3, 0.8, -0.9, 0.2])
        mock_component.get_samples.return_value = test_samples

        # Set up the module
        module.create_engine_component(input_components=[mock_component])

        # Update display
        module._update_display()

        # Check min/max labels
        assert "-0.900" in module.min_label.text()
        assert "+0.800" in module.max_label.text()

    def test_stereo_to_mono_conversion(self):
        """Test that stereo input is converted to mono for display."""
        module = WaveformModule()

        # Create stereo samples (2 channels)
        mock_component = MagicMock()
        left = np.array([1.0, 0.5, 0.0])
        right = np.array([0.0, 0.5, 1.0])
        stereo_samples = np.column_stack((left, right))
        mock_component.get_samples.return_value = stereo_samples

        # Set up the module
        module.create_engine_component(input_components=[mock_component])

        # Should not crash when processing stereo
        module._update_display()

        # Verify it was called
        mock_component.get_samples.assert_called()

    def test_no_component_returns_none(self):
        """Test that create_engine_component returns None with no inputs."""
        module = WaveformModule()

        # Call with no inputs
        result = module.create_engine_component(input_components=None)
        assert result is None
        assert module.input_component is None

        # Call with empty list
        result = module.create_engine_component(input_components=[])
        assert result is None
        assert module.input_component is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

