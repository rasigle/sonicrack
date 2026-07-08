"""Test LFO to Clipper modulation with automatic CV scaling.

This test suite validates that the PortModulatorAdapter correctly scales
CV signals between incompatible ranges (e.g., bipolar LFO to unipolar Clipper).
"""

from typing import Any

import numpy as np
import pytest

from sonicrack.gui.modules._modulated_base import PortModulatorAdapter


class MockPort:
    """Mock port for testing CV signal flow."""

    def __init__(self, module, data):
        self.parent_module = module
        self.cables = []
        self._data = data
        self._read_count = 0

    def read(self, num_samples):
        """Simulate reading samples from port."""
        return self._data[:num_samples]


class MockCable:
    """Mock cable for testing port connections."""

    def __init__(self, start_port):
        self.start_port = start_port


class MockLFOModule:
    """Mock LFO module that outputs bipolar CV in range [-1, 1]."""

    @staticmethod
    def get_cv_output_range():
        """Return the CV output range for this module."""
        return (-1.0, 1.0)


class MockClipperModule:
    """Mock Clipper module that expects unipolar CV in range [0, 1]."""

    def get_cv_range(self, port_name="Mod"):
        """Return the expected CV input range for this module."""
        return (0.0, 1.0)


@pytest.fixture
def qapp(qapp: Any):
    """Qt application fixture."""
    return qapp


@pytest.fixture
def mock_lfo_bipolar():
    """Create a mock LFO module with bipolar output [-1, 1]."""
    return MockLFOModule()


@pytest.fixture
def mock_clipper_unipolar():
    """Create a mock clipper module expecting unipolar input [0, 1]."""
    return MockClipperModule()


@pytest.fixture
def mock_read_samples(monkeypatch):
    """Patch runtime_helpers.read_samples to use mock data."""
    import sonicrack.gui.core.runtime_helpers

    def _mock_read_samples(port, num_samples):
        return port._data[:num_samples]

    monkeypatch.setattr(
        sonicrack.gui.core.runtime_helpers, "read_samples", _mock_read_samples
    )


def test_lfo_to_clipper_cv_scaling_full_range(
    qapp: Any, mock_lfo_bipolar, mock_clipper_unipolar, mock_read_samples
):
    """Test that LFO bipolar output [-1, 1] is correctly scaled to unipolar [0, 1]."""
    del qapp  # Unused but required by fixture

    # Create mock LFO output port with bipolar data [-1, 1]
    lfo_output_data = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    lfo_port = MockPort(mock_lfo_bipolar, lfo_output_data)

    # Create mock cable and clipper input port
    cable = MockCable(lfo_port)
    clipper_mod_port = MockPort(mock_clipper_unipolar, lfo_output_data)
    clipper_mod_port.cables = [cable]

    # Create PortModulatorAdapter with expected range [0, 1]
    adapter = PortModulatorAdapter(
        clipper_mod_port, num_samples=5, expected_range=(0.0, 1.0)
    )

    # Verify scaling parameters are correct
    assert adapter._needs_scaling, "CV scaling should be enabled for range mismatch"
    assert adapter._scale == 0.5, f"Scale should be 0.5, got {adapter._scale}"
    assert adapter._offset == 0.5, f"Offset should be 0.5, got {adapter._offset}"

    # Get scaled samples
    samples = adapter.get_samples(5)

    # Expected: [-1, -0.5, 0, 0.5, 1] -> [0, 0.25, 0.5, 0.75, 1]
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    assert np.allclose(samples, expected, rtol=1e-5), (
        f"CV scaling failed! Expected {expected}, got {samples}"
    )


def test_lfo_at_minimum_value(
    qapp: Any, mock_lfo_bipolar, mock_clipper_unipolar, mock_read_samples
):
    """Test that LFO at minimum value (-1.0) scales to 0.0."""
    del qapp

    # Create mock data with LFO at minimum (-1.0)
    lfo_at_min = np.full(10, -1.0, dtype=np.float32)
    lfo_port = MockPort(mock_lfo_bipolar, lfo_at_min)

    cable = MockCable(lfo_port)
    clipper_mod_port = MockPort(mock_clipper_unipolar, lfo_at_min)
    clipper_mod_port.cables = [cable]

    adapter = PortModulatorAdapter(
        clipper_mod_port, num_samples=10, expected_range=(0.0, 1.0)
    )

    samples = adapter.get_samples(10)

    # LFO at -1.0 should scale to 0.0 (minimum clipping threshold)
    assert np.allclose(samples, 0.0, rtol=1e-5), (
        f"LFO at minimum (-1.0) should scale to 0.0, got {samples[0]:.3f}"
    )


def test_lfo_at_maximum_value(
    qapp: Any, mock_lfo_bipolar, mock_clipper_unipolar, mock_read_samples
):
    """Test that LFO at maximum value (1.0) scales to 1.0."""
    del qapp

    # Create mock data with LFO at maximum (1.0)
    lfo_at_max = np.full(10, 1.0, dtype=np.float32)
    lfo_port = MockPort(mock_lfo_bipolar, lfo_at_max)

    cable = MockCable(lfo_port)
    clipper_mod_port = MockPort(mock_clipper_unipolar, lfo_at_max)
    clipper_mod_port.cables = [cable]

    adapter = PortModulatorAdapter(
        clipper_mod_port, num_samples=10, expected_range=(0.0, 1.0)
    )

    samples = adapter.get_samples(10)

    # LFO at 1.0 should scale to 1.0 (maximum clipping threshold)
    assert np.allclose(samples, 1.0, rtol=1e-5), (
        f"LFO at maximum (1.0) should scale to 1.0, got {samples[0]:.3f}"
    )


def test_lfo_midpoint_value(
    qapp: Any, mock_lfo_bipolar, mock_clipper_unipolar, mock_read_samples
):
    """Test that LFO at midpoint (0.0) scales to 0.5."""
    del qapp

    # Create mock data with LFO at midpoint (0.0)
    lfo_at_mid = np.full(10, 0.0, dtype=np.float32)
    lfo_port = MockPort(mock_lfo_bipolar, lfo_at_mid)

    cable = MockCable(lfo_port)
    clipper_mod_port = MockPort(mock_clipper_unipolar, lfo_at_mid)
    clipper_mod_port.cables = [cable]

    adapter = PortModulatorAdapter(
        clipper_mod_port, num_samples=10, expected_range=(0.0, 1.0)
    )

    samples = adapter.get_samples(10)

    # LFO at 0.0 should scale to 0.5 (middle clipping threshold)
    assert np.allclose(samples, 0.5, rtol=1e-5), (
        f"LFO at midpoint (0.0) should scale to 0.5, got {samples[0]:.3f}"
    )


def test_no_scaling_when_ranges_match(qapp: Any, mock_read_samples):
    """Test that no scaling is applied when CV ranges already match."""
    del qapp

    # Create mock module with matching range
    class MockMatchingModule:
        @staticmethod
        def get_cv_output_range():
            return (0.0, 1.0)

    module = MockMatchingModule()

    # Create mock data already in correct range [0, 1]
    unipolar_data = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    port = MockPort(module, unipolar_data)

    # Add cable from matching source
    cable = MockCable(port)
    port.cables = [cable]

    adapter = PortModulatorAdapter(port, num_samples=5, expected_range=(0.0, 1.0))

    # Verify no scaling is needed
    assert not adapter._needs_scaling, "Scaling should be disabled for matching ranges"

    samples = adapter.get_samples(5)

    # Data should pass through unchanged
    assert np.allclose(samples, unipolar_data, rtol=1e-5), (
        f"Data should be unchanged when ranges match, got {samples}"
    )


def test_adapter_with_varying_cv_signal(
    qapp: Any, mock_lfo_bipolar, mock_clipper_unipolar, mock_read_samples
):
    """Test scaling with a realistic varying CV signal."""
    del qapp

    # Create a more realistic LFO curve (sine-like)
    lfo_signal = np.array(
        [0.0, 0.5, 0.866, 1.0, 0.866, 0.5, 0.0, -0.5, -0.866, -1.0],
        dtype=np.float32,
    )
    lfo_port = MockPort(mock_lfo_bipolar, lfo_signal)

    cable = MockCable(lfo_port)
    clipper_mod_port = MockPort(mock_clipper_unipolar, lfo_signal)
    clipper_mod_port.cables = [cable]

    adapter = PortModulatorAdapter(
        clipper_mod_port, num_samples=10, expected_range=(0.0, 1.0)
    )

    samples = adapter.get_samples(10)

    # All samples should be in range [0, 1]
    assert np.all(samples >= 0.0), "All samples should be >= 0.0"
    assert np.all(samples <= 1.0), "All samples should be <= 1.0"

    # Verify specific points are scaled correctly
    # 0.0 -> 0.5, 1.0 -> 1.0, -1.0 -> 0.0
    assert np.isclose(samples[0], 0.5, rtol=1e-5), "0.0 should scale to 0.5"
    assert np.isclose(samples[3], 1.0, rtol=1e-5), "1.0 should scale to 1.0"
    assert np.isclose(samples[9], 0.0, rtol=1e-5), "-1.0 should scale to 0.0"


def test_adapter_multiple_get_samples_calls(
    qapp: Any, mock_lfo_bipolar, mock_clipper_unipolar, mock_read_samples
):
    """Test that adapter maintains correct scaling across multiple buffer reads."""
    del qapp

    # Create test data
    lfo_output_data = np.array(
        [-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.0], dtype=np.float32
    )
    lfo_port = MockPort(mock_lfo_bipolar, lfo_output_data)

    cable = MockCable(lfo_port)
    clipper_mod_port = MockPort(mock_clipper_unipolar, lfo_output_data)
    clipper_mod_port.cables = [cable]

    adapter = PortModulatorAdapter(
        clipper_mod_port, num_samples=10, expected_range=(0.0, 1.0)
    )

    # First call: read 5 samples
    samples1 = adapter.get_samples(5)
    expected1 = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    assert np.allclose(samples1, expected1, rtol=1e-5), (
        f"First read failed: expected {expected1}, got {samples1}"
    )

    # Second call: reads the same data again (mock always reads from start)
    # In real usage, the port would have new data written each buffer
    samples2 = adapter.get_samples(5)
    # Should get the same scaled values (first 5 samples)
    assert np.allclose(samples2, expected1, rtol=1e-5), (
        f"Second read failed: expected {expected1}, got {samples2}"
    )

    # Verify scaling is consistent across multiple calls
    assert adapter._needs_scaling, "Scaling should remain enabled"
    assert adapter._scale == 0.5, "Scale should remain 0.5"
    assert adapter._offset == 0.5, "Offset should remain 0.5"


def test_modulation_amount_control(
    qapp: Any, mock_lfo_bipolar, mock_clipper_unipolar, mock_read_samples
):
    """Test that modulation_amount correctly scales the modulation depth."""
    del qapp

    # Create mock LFO output with full range [-1, 1]
    lfo_output_data = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    lfo_port = MockPort(mock_lfo_bipolar, lfo_output_data)

    cable = MockCable(lfo_port)
    clipper_mod_port = MockPort(mock_clipper_unipolar, lfo_output_data)
    clipper_mod_port.cables = [cable]

    # Test with full modulation (amount = 1.0)
    adapter = PortModulatorAdapter(
        clipper_mod_port,
        num_samples=5,
        expected_range=(0.0, 1.0),
        modulation_amount=1.0,
    )
    samples_full = adapter.get_samples(5)
    expected_full = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    assert np.allclose(samples_full, expected_full, rtol=1e-5), (
        f"Full modulation failed: expected {expected_full}, got {samples_full}"
    )

    # Test with 50% modulation (amount = 0.5)
    # Should scale around center (0.5): output = 0.5 + (scaled_value - 0.5) * 0.5
    adapter.modulation_amount = 0.5
    samples_half = adapter.get_samples(5)
    # Expected: [0.0, 0.25, 0.5, 0.75, 1.0] scaled around 0.5 by 0.5
    # = [0.5 + (0.0-0.5)*0.5, 0.5 + (0.25-0.5)*0.5, 0.5, 0.5 + (0.75-0.5)*0.5, 0.5 +
    # (1.0-0.5)*0.5]
    # = [0.25, 0.375, 0.5, 0.625, 0.75]
    expected_half = np.array([0.25, 0.375, 0.5, 0.625, 0.75], dtype=np.float32)
    assert np.allclose(samples_half, expected_half, rtol=1e-5), (
        f"Half modulation failed: expected {expected_half}, got {samples_half}"
    )

    # Test with no modulation (amount = 0.0)
    # Should output center value (0.5) for all samples
    adapter.modulation_amount = 0.0
    samples_none = adapter.get_samples(5)
    expected_none = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    assert np.allclose(samples_none, expected_none, rtol=1e-5), (
        f"No modulation failed: expected {expected_none}, got {samples_none}"
    )
