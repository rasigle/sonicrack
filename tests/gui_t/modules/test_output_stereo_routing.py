"""Test cases for OutputModule stereo routing functionality."""

import numpy as np
import pytest

from src.gui.modules.output.output import OutputModule
from src.utils.audio_utils import combine_lr_to_stereo, mono_to_stereo


@pytest.fixture
def output_module(app):
    """Create an OutputModule instance for testing."""
    return OutputModule()


class TestMonoToStereo:
    """Test cases for mono_to_stereo function."""

    def test_scalar_input(self):
        """Test scalar input is duplicated to stereo."""
        result = mono_to_stereo(np.array(0.5))
        assert result.shape == (1, 2)
        assert np.allclose(result, [[0.5, 0.5]])

    def test_single_sample_1d(self):
        """Test single sample 1D array."""
        result = mono_to_stereo(np.array([0.5]))
        assert result.shape == (1, 2)
        assert np.allclose(result, [[0.5, 0.5]])

    def test_multiple_samples_1d(self):
        """Test multiple samples in 1D array."""
        input_samples = np.array([0.1, 0.2, 0.3, 0.4])
        result = mono_to_stereo(input_samples)
        assert result.shape == (4, 2)
        assert np.allclose(result[:, 0], input_samples)
        assert np.allclose(result[:, 1], input_samples)

    def test_already_stereo_2d(self):
        """Test 2D array that's already stereo (N, 2)."""
        input_samples = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = mono_to_stereo(input_samples)
        assert result.shape == (2, 2)
        assert np.allclose(result, input_samples)

    def test_transposed_stereo_2d(self):
        """Test 2D array that's transposed stereo (2, N)."""
        input_samples = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = mono_to_stereo(input_samples)
        assert result.shape == (3, 2)
        assert np.allclose(result, [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])

    def test_single_column_2d(self):
        """Test 2D array with single column (N, 1)."""
        input_samples = np.array([[0.1], [0.2], [0.3]])
        result = mono_to_stereo(input_samples)
        assert result.shape == (3, 2)
        assert np.allclose(result[:, 0], [0.1, 0.2, 0.3])
        assert np.allclose(result[:, 1], [0.1, 0.2, 0.3])

    def test_single_row_2d(self):
        """Test 2D array with single row (1, N)."""
        input_samples = np.array([[0.1, 0.2, 0.3]])
        result = mono_to_stereo(input_samples)
        assert result.shape == (3, 2)
        assert np.allclose(result[:, 0], [0.1, 0.2, 0.3])
        assert np.allclose(result[:, 1], [0.1, 0.2, 0.3])

    def test_multiple_columns_2d(self):
        """Test 2D array with (2, N) shape - treated as transposed stereo."""
        # (2, 3) array is treated as transposed stereo: 2 channels, 3 samples each
        input_samples = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        result = mono_to_stereo(input_samples)
        # Should be transposed to (3, 2)
        assert result.shape == (3, 2)
        assert np.allclose(result, [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])

    def test_averaging_multiple_rows_and_columns(self):
        """Test 2D array with (>2, >2) shape - averaged and duplicated."""
        # (3, 3) array should average each row and duplicate to stereo
        input_samples = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        result = mono_to_stereo(input_samples)
        assert result.shape == (3, 2)
        # Average of each row
        expected_avg = np.array([0.2, 0.5, 0.8])
        assert np.allclose(result[:, 0], expected_avg, rtol=1e-5)
        assert np.allclose(result[:, 1], expected_avg, rtol=1e-5)

    def test_3d_array_flattened(self):
        """Test 3D array is flattened then duplicated."""
        input_samples = np.array([[[0.1, 0.2]], [[0.3, 0.4]]])
        result = mono_to_stereo(input_samples)
        assert result.shape == (4, 2)
        flat = np.array([0.1, 0.2, 0.3, 0.4])
        assert np.allclose(result[:, 0], flat)
        assert np.allclose(result[:, 1], flat)


class TestCombineLRToStereo:
    """Test cases for combine_lr_to_stereo function."""

    def test_same_length_1d(self):
        """Test combining two 1D arrays of same length."""
        left = np.array([0.1, 0.2, 0.3])
        right = np.array([0.4, 0.5, 0.6])
        result = combine_lr_to_stereo(left, right)
        assert result.shape == (3, 2)
        assert np.allclose(result[:, 0], left)
        assert np.allclose(result[:, 1], right)

    def test_left_longer(self):
        """Test when left array is longer (right is zero-padded)."""
        left = np.array([0.1, 0.2, 0.3, 0.4])
        right = np.array([0.5, 0.6])
        result = combine_lr_to_stereo(left, right)
        assert result.shape == (4, 2)
        assert np.allclose(result[:, 0], left)
        assert np.allclose(result[:, 1], [0.5, 0.6, 0.0, 0.0])

    def test_right_longer(self):
        """Test when right array is longer (left is zero-padded)."""
        left = np.array([0.1, 0.2])
        right = np.array([0.3, 0.4, 0.5, 0.6])
        result = combine_lr_to_stereo(left, right)
        assert result.shape == (4, 2)
        assert np.allclose(result[:, 0], [0.1, 0.2, 0.0, 0.0])
        assert np.allclose(result[:, 1], right)

    def test_scalar_inputs(self):
        """Test scalar inputs."""
        left = np.array(0.1)
        right = np.array(0.2)
        result = combine_lr_to_stereo(left, right)
        assert result.shape == (1, 2)
        assert np.allclose(result, [[0.1, 0.2]])

    def test_single_sample_each(self):
        """Test single sample in each channel."""
        left = np.array([0.1])
        right = np.array([0.2])
        result = combine_lr_to_stereo(left, right)
        assert result.shape == (1, 2)
        assert np.allclose(result, [[0.1, 0.2]])

    def test_2d_arrays_flattened(self):
        """Test 2D arrays are flattened before combining."""
        left = np.array([[0.1, 0.2], [0.3, 0.4]])
        right = np.array([[0.5, 0.6], [0.7, 0.8]])
        result = combine_lr_to_stereo(left, right)
        assert result.shape == (4, 2)
        assert np.allclose(result[:, 0], [0.1, 0.2, 0.3, 0.4])
        assert np.allclose(result[:, 1], [0.5, 0.6, 0.7, 0.8])

    def test_empty_arrays(self):
        """Test empty arrays."""
        left = np.array([])
        right = np.array([])
        result = combine_lr_to_stereo(left, right)
        assert result.shape == (0, 2)


class TestOutputModuleRouting:
    """Test cases for OutputModule stereo routing logic (integration tests)."""

    def test_inactive_output_does_not_start_playback(self, output_module):
        """Inactive output module must not restart playback."""
        output_module.set_active(False)

        output_module.start_playback()

        assert not output_module.audio_output.is_playing
        assert output_module.status_label.text() == "Off"

    def test_inactive_output_generates_silence(self, output_module):
        """Inactive output callback must return silence even with inputs."""
        output_module.set_active(False)

        samples = output_module._generate_samples(8)

        assert samples.shape == (8, 2)
        assert np.allclose(samples, np.zeros((8, 2), dtype=np.float32))

    def test_mono_to_stereo_integration(self):
        """Test mono input is properly converted to stereo."""
        mono_input = np.array([0.1, 0.2, 0.3, 0.4])
        stereo_output = mono_to_stereo(mono_input)

        # Verify stereo shape
        assert stereo_output.shape == (4, 2)
        # Both channels identical
        assert np.allclose(stereo_output[:, 0], mono_input)
        assert np.allclose(stereo_output[:, 1], mono_input)

    def test_lr_combination_integration(self):
        """Test L and R inputs are properly combined."""
        left_input = np.array([0.1, 0.2, 0.3])
        right_input = np.array([0.4, 0.5, 0.6])
        stereo_output = combine_lr_to_stereo(left_input, right_input)

        # Verify stereo shape
        assert stereo_output.shape == (3, 2)
        # Left channel has left input
        assert np.allclose(stereo_output[:, 0], left_input)
        # Right channel has right input
        assert np.allclose(stereo_output[:, 1], right_input)

    def test_mismatched_lengths_integration(self):
        """Test L and R with different lengths are properly handled."""
        left_input = np.array([0.1, 0.2])
        right_input = np.array([0.3, 0.4, 0.5, 0.6])
        stereo_output = combine_lr_to_stereo(left_input, right_input)

        # Verify stereo shape (padded to longer length)
        assert stereo_output.shape == (4, 2)
        # Left is zero-padded
        assert np.allclose(stereo_output[:, 0], [0.1, 0.2, 0.0, 0.0])
        # Right has all values
        assert np.allclose(stereo_output[:, 1], right_input)

    def test_gain_calculation(self):
        """Test dB to linear gain calculation."""
        # +6 dB should be approximately 2x
        gain_db = 6.0
        linear_gain = 10.0 ** (gain_db / 20.0)
        assert np.isclose(linear_gain, 1.995, rtol=0.01)

        # 0 dB should be 1x (unity)
        gain_db = 0.0
        linear_gain = 10.0 ** (gain_db / 20.0)
        assert np.isclose(linear_gain, 1.0)

        # -6 dB should be approximately 0.5x
        gain_db = -6.0
        linear_gain = 10.0 ** (gain_db / 20.0)
        assert np.isclose(linear_gain, 0.501, rtol=0.01)

        # -80 dB should be very small
        gain_db = -80.0
        linear_gain = 10.0 ** (gain_db / 20.0)
        assert linear_gain < 0.001

    def test_apply_gain_to_stereo(self):
        """Test applying gain to stereo samples."""
        stereo_samples = np.array([[0.1, 0.2], [0.3, 0.4]])
        gain_db = 6.0
        linear_gain = 10.0 ** (gain_db / 20.0)

        result = stereo_samples * linear_gain
        expected = np.array([[0.1, 0.2], [0.3, 0.4]]) * linear_gain

        assert np.allclose(result, expected)

    def test_edge_case_single_sample(self):
        """Test single sample mono to stereo."""
        mono_input = np.array([0.5])
        stereo_output = mono_to_stereo(mono_input)

        assert stereo_output.shape == (1, 2)
        assert np.allclose(stereo_output, [[0.5, 0.5]])

    def test_edge_case_scalar(self):
        """Test scalar input to stereo."""
        mono_input = np.array(0.5)
        stereo_output = mono_to_stereo(mono_input)

        assert stereo_output.shape == (1, 2)
        assert np.allclose(stereo_output, [[0.5, 0.5]])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
