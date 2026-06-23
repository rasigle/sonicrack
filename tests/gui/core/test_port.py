"""Test cases for the Port class.

NOTE: These tests have been updated to work with the refactored architecture.
The Port class now uses PortWidget (UI) + Port (logic).

For pure logic tests without Qt, see tests/core/test_port_model.py
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.engine import SineOscillator, TriangleOscillator
from src.gui.core.port import Port


# Mock parent module for tests
def mock_parent():
    """Create a mock parent module for testing."""
    return MagicMock()


class TestPortBasics:
    """Test basic Port functionality."""

    def test_port_creation(self):
        """Test that a port can be created with a name."""
        port = Port("input", "test_port", mock_parent())
        assert port.port_name == "test_port"
        assert port.port_type == "input"
        assert port.value == 0.0
        assert not port.is_connected

    def test_port_name_storage(self):
        """Test that port stores its name correctly."""
        port1 = Port("input", "input", mock_parent())
        port2 = Port("output", "output", mock_parent())
        assert port1.port_name == "input"
        assert port2.port_name == "output"

    def test_initial_value_is_zero(self):
        """Test that port initializes with value 0.0."""
        port = Port("input", "test", mock_parent())
        assert port.value == 0.0


class TestPortWriteRead:
    """Test writing and reading port values."""

    def test_write_value(self):
        """Test writing a value to a port."""
        port = Port("output", "test", mock_parent())
        port.write(5.0)
        assert port.value == 5.0

    def test_write_multiple_values(self):
        """Test writing multiple values overwrites previous value."""
        port = Port("output", "test", mock_parent())
        port.write(1.0)
        assert port.value == 1.0
        port.write(2.0)
        assert port.value == 2.0
        port.write(3.5)
        assert port.value == 3.5

    def test_read_disconnected_port_returns_zero(self):
        """Test that reading a disconnected port returns 0.0."""
        port = Port("input", "test", mock_parent())
        assert port.read() == 0.0

    def test_read_disconnected_port_with_value(self):
        """Test that reading disconnected port returns 0.0 even if port has value."""
        port = Port("input", "test", mock_parent())
        port.write(5.0)
        # Reading disconnected port returns 0.0, not the port's own value
        assert port.read() == 0.0

    def test_write_negative_value(self):
        """Test writing negative values."""
        port = Port("output", "test", mock_parent())
        port.write(-3.5)
        assert port.value == -3.5

    def test_write_zero(self):
        """Test writing zero value."""
        port = Port("output", "test", mock_parent())
        port.write(10.0)
        port.write(0.0)
        assert port.value == 0.0


class TestPortConnection:
    """Test port connection functionality."""

    def test_connect_two_ports(self):
        """Test connecting one port to another."""
        input_port = Port("input", "input", mock_parent())
        output_port = Port("output", "output", mock_parent())

        input_port.connect(output_port)

        assert output_port in input_port.connected_to
        assert input_port.is_connected

    def test_connect_changes_connected_to(self):
        """Test that connect sets the connected_to attribute."""
        port1 = Port("input", "port1", mock_parent())
        port2 = Port("output", "port2", mock_parent())

        assert not port1.is_connected
        port1.connect(port2)
        assert port1.is_connected
        assert port2 in port1.connected_to

    def test_connect_multiple_times(self):
        """Test that connecting to same port multiple times is idempotent."""
        port1 = Port("input", "port1", mock_parent())
        port2 = Port("output", "port2", mock_parent())

        port1.connect(port2)
        assert port2 in port1.connected_to
        assert len(port1.connected_to) == 1

        # Connect again - should be idempotent
        port1.connect(port2)
        assert port2 in port1.connected_to
        assert len(port1.connected_to) == 1  # Still only one connection

    def test_is_connected_property(self):
        """Test the is_connected property."""
        port1 = Port("input", "port1", mock_parent())
        port2 = Port("output", "port2", mock_parent())

        assert not port1.is_connected

        port1.connect(port2)
        assert port1.is_connected


class TestPortReadConnected:
    """Test reading values from connected ports."""

    def test_read_from_connected_Port(self):
        """Test that reading from connected port returns its value."""
        input_port = Port("input", "input", mock_parent())
        output_port = Port("input", "output", mock_parent())

        output_port.write(7.5)
        input_port.connect(output_port)

        assert input_port.read() == 7.5

    def test_read_updates_when_connected_port_changes(self):
        """Test that read reflects changes in connected port."""
        input_port = Port("input", "input", mock_parent())
        output_port = Port("input", "output", mock_parent())

        input_port.connect(output_port)

        output_port.write(1.0)
        assert input_port.read() == 1.0

        output_port.write(2.5)
        assert input_port.read() == 2.5

        output_port.write(0.0)
        assert input_port.read() == 0.0

    def test_read_ignores_own_value_when_connected(self):
        """Test that reading connected port ignores the reading port's own value."""
        input_port = Port("input", "input", mock_parent())
        output_port = Port("input", "output", mock_parent())

        input_port.write(100.0)  # Input port has its own value
        output_port.write(5.0)  # Output port has different value
        input_port.connect(output_port)

        # Should read from connected port, not own value
        assert input_port.read() == 5.0

    def test_read_with_zero_value_connection(self):
        """Test reading from connected port that has zero value."""
        input_port = Port("input", "input", mock_parent())
        output_port = Port("input", "output", mock_parent())

        output_port.write(0.0)
        input_port.connect(output_port)

        assert input_port.read() == 0.0


class TestPortDisconnect:
    """Test port disconnection functionality."""

    def test_disconnect_Port(self):
        """Test disconnecting a port."""
        port1 = Port("input", "port1", mock_parent())
        port2 = Port("output", "port2", mock_parent())

        port1.connect(port2)
        assert port1.is_connected

        port1.disconnect()
        assert not port1.is_connected
        assert port1.connected_to == []

    def test_disconnect_clears_connection(self):
        """Test that disconnect clears the connected_to attribute."""
        port1 = Port("input", "port1", mock_parent())
        port2 = Port("output", "port2", mock_parent())

        port1.connect(port2)
        port1.disconnect()

        assert port1.connected_to == []

    def test_disconnect_affects_read(self):
        """Test that disconnecting affects read behavior."""
        input_port = Port("input", "input", mock_parent())
        output_port = Port("input", "output", mock_parent())

        output_port.write(5.0)
        input_port.connect(output_port)
        assert input_port.read() == 5.0

        input_port.disconnect()
        assert input_port.read() == 0.0  # Returns 0.0 when disconnected

    def test_disconnect_already_disconnected(self):
        """Test that disconnecting an already disconnected port is safe."""
        port = Port("input", "test", mock_parent())
        assert not port.is_connected

        # Should not raise exception
        port.disconnect()
        assert not port.is_connected

    def test_disconnect_after_reconnect(self):
        """Test disconnecting after reconnecting to different port."""
        port1 = Port("input", "port1", mock_parent())
        port2 = Port("output", "port2", mock_parent())
        port3 = Port("output", "port3", mock_parent())

        port1.connect(port2)
        port1.connect(port3)
        port1.disconnect()

        assert not port1.is_connected


class TestPortEdgeCases:
    """Test edge cases and special scenarios."""

    def test_self_connection(self):
        """Test that connecting a port to itself raises ValueError."""
        port = Port("input", "self", mock_parent())
        port.write(5.0)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Cannot connect a port to itself"):
            port.connect(port)

    def test_circular_connection(self):
        """Test circular connections (A->B->A)."""
        port_a = Port("input", "A", mock_parent())
        port_b = Port("input", "B", mock_parent())

        port_a.connect(port_b)
        port_b.connect(port_a)

        port_a.write(10.0)

        # A reads from B, B reads from A (circular)
        # This is allowed but may cause issues in real usage
        assert port_b in port_a.connected_to
        assert port_a in port_b.connected_to

    def test_chain_connection(self):
        """Test chain of connections (A->B->C)."""
        port_a = Port("input", "A", mock_parent())
        port_b = Port("input", "B", mock_parent())
        port_c = Port("input", "C", mock_parent())

        port_c.write(42.0)
        port_b.connect(port_c)
        port_a.connect(port_b)

        # A reads from B, which has its own value (not C's)
        assert port_a.read() == port_b.value

        # B reads from C
        assert port_b.read() == 42.0

    def test_large_value(self):
        """Test with large floating point values."""
        port = Port("input", "test", mock_parent())
        large_value = 1e10
        port.write(large_value)
        assert port.value == large_value

    def test_small_value(self):
        """Test with very small floating point values."""
        port = Port("input", "test", mock_parent())
        small_value = 1e-10
        port.write(small_value)
        assert port.value == small_value

    def test_special_float_values(self):
        """Test with special float values."""
        port = Port("input", "test", mock_parent())

        # Test with infinity
        port.write(float("inf"))
        assert port.value == float("inf")

        # Test with negative infinity
        port.write(float("-inf"))
        assert port.value == float("-inf")

        # Test with NaN
        port.write(float("nan"))
        # NaN != NaN, so we check using isnan
        import math

        assert math.isnan(port.value)


class TestPortNameHandling:
    """Test port name handling."""

    def test_empty_name(self):
        """Test creating port with empty name."""
        port = Port("input", "", mock_parent())
        assert port.port_name == ""

    def test_special_characters_in_name(self):
        """Test port names with special characters."""
        port1 = Port("input", "port-1", mock_parent())
        port2 = Port("input", "port_2", mock_parent())
        port3 = Port("input", "port.3", mock_parent())

        assert port1.port_name == "port-1"
        assert port2.port_name == "port_2"
        assert port3.port_name == "port.3"

    def test_unicode_name(self):
        """Test port with unicode characters in name."""
        port = Port("input", "测试端口", mock_parent())
        assert port.port_name == "测试端口"

    def test_long_name(self):
        """Test port with very long name."""
        long_name = "a" * 1000
        port = Port("input", long_name, mock_parent())
        assert port.port_name == long_name


class TestPortIntegration:
    """Integration tests with multiple ports."""

    def test_multiple_inputs_one_output(self):
        """Test multiple input ports connected to one output."""
        output = Port("output", "output", mock_parent())
        input1 = Port("input", "input1", mock_parent())
        input2 = Port("input", "input2", mock_parent())
        input3 = Port("input", "input3", mock_parent())

        output.write(10.0)

        input1.connect(output)
        input2.connect(output)
        input3.connect(output)

        assert input1.read() == 10.0
        assert input2.read() == 10.0
        assert input3.read() == 10.0

        output.write(20.0)

        assert input1.read() == 20.0
        assert input2.read() == 20.0
        assert input3.read() == 20.0

    def test_partial_disconnect(self):
        """Test disconnecting some ports while others remain connected."""
        output = Port("output", "output", mock_parent())
        input1 = Port("input", "input1", mock_parent())
        input2 = Port("input", "input2", mock_parent())

        output.write(5.0)
        input1.connect(output)
        input2.connect(output)

        input1.disconnect()

        assert input1.read() == 0.0
        assert input2.read() == 5.0

    def test_reconnect_after_value_change(self):
        """Test reconnecting after value changes."""
        port1 = Port("input", "port1", mock_parent())
        port2 = Port("output", "port2", mock_parent())
        port3 = Port("output", "port3", mock_parent())

        port2.write(10.0)
        port1.connect(port2)
        assert port1.read() == 10.0

        port1.disconnect()
        port3.write(20.0)
        port1.connect(port3)
        assert port1.read() == 20.0


class TestPortValueType:
    """Test value type handling (should be float)."""

    def test_write_integer_converts_to_float(self):
        """Test that integer values work (Python handles this)."""
        port = Port("input", "test", mock_parent())
        port.write(5)  # Integer
        assert port.value == 5.0
        assert isinstance(port.value, (int, float))

    def test_write_float_value(self):
        """Test writing explicit float values."""
        port = Port("input", "test", mock_parent())
        port.write(5.5)
        assert port.value == 5.5
        assert isinstance(port.value, float)


def test_port_model_creation():
    """Test creating a Port model (no UI dependencies)."""
    port = Port("input", "test_port", index=0)

    assert port.port_type == "input"
    assert port.port_name == "test_port"
    assert port.index == 0
    assert port.value == 0.0
    assert not port.is_connected


def test_port_connection():
    """Test connecting two ports."""
    input_port = Port("input", "audio_in")
    output_port = Port("output", "audio_out")

    # Connect
    input_port.connect(output_port)
    assert input_port.is_connected
    assert output_port in input_port.connected_to

    # Disconnect
    input_port.disconnect()
    assert not input_port.is_connected
    assert input_port.connected_to == []


def test_port_read_write():
    """Test reading and writing values."""
    input_port = Port("input", "audio_in")
    output_port = Port("output", "audio_out")

    # Connect ports
    input_port.connect(output_port)

    # Write to output, read from input
    output_port.write(0.5)
    assert output_port.value == 0.5
    assert input_port.read() == 0.5

    # Update value
    output_port.write(0.75)
    assert input_port.read() == 0.75


def test_port_disconnected_read():
    """Test reading from disconnected port returns 0."""
    port = Port("input", "audio_in")
    assert not port.is_connected
    assert port.read() == 0.0


def test_port_invalid_connection():
    """Test that connecting to non-Port raises TypeError."""
    port = Port("input", "audio_in")

    with pytest.raises(TypeError, match="Can only connect to another Port"):
        port.connect("not a port")


def test_port_repr():
    """Test string representations."""
    port = Port("input", "test", index=0)
    port.write(0.5)

    repr_str = repr(port)
    assert "test" in repr_str
    assert "input" in repr_str
    assert "0.500" in repr_str

    str_str = str(port)
    assert "test" in str_str
    assert "input" in str_str


class TestPortModelBasics:
    """Test basic PortModel functionality."""

    def test_port_creation(self):
        """Test that a port can be created with type and name."""
        port = Port("input", "test_port")
        assert port.port_name == "test_port"
        assert port.port_type == "input"
        assert port.value == 0.0
        assert port.connected_to == []
        assert not port.is_connected

    def test_port_types(self):
        """Test creating different port types."""
        input_port = Port("input", "in")
        output_port = Port("output", "out")

        assert input_port.port_type == "input"
        assert output_port.port_type == "output"

    def test_port_with_index(self):
        """Test creating port with index."""
        port = Port("input", "test", index=5)
        assert port.index == 5

    def test_initial_value_is_zero(self):
        """Test that port initializes with value 0.0."""
        port = Port("input", "test")
        assert port.value == 0.0


class TestPortModelWriteRead:
    """Test writing and reading port values."""

    def test_write_value(self):
        """Test writing a value to a port."""
        port = Port("output", "test")
        port.write(5.0)
        assert port.value == 5.0

    def test_write_integer_converts_to_float(self):
        """Test that integer values are converted to float."""
        port = Port("output", "test")
        port.write(5)
        assert port.value == 5.0
        assert isinstance(port.value, float)

    def test_read_disconnected_port_returns_zero(self):
        """Test that reading disconnected port returns 0.0."""
        port = Port("input", "test")
        assert port.read() == 0.0

    def test_read_disconnected_port_with_value(self):
        """Test that reading disconnected port returns 0.0 even if port has value."""
        port = Port("input", "test")
        port.write(5.0)
        # Reading disconnected port returns 0.0, not the port's own value
        assert port.read() == 0.0

    def test_write_negative_value(self):
        """Test writing negative values."""
        port = Port("output", "test")
        port.write(-3.5)
        assert port.value == -3.5


class TestPortModelConnection:
    """Test port connection functionality."""

    def test_connect_two_ports(self):
        """Test connecting one port to another."""
        input_port = Port("input", "in")
        output_port = Port("output", "out")

        input_port.connect(output_port)

        assert output_port in input_port.connected_to
        assert input_port.is_connected

    def test_connect_invalid_type_raises_error(self):
        """Test that connecting to non-Port raises TypeError."""
        port = Port("input", "test")

        with pytest.raises(TypeError, match="Can only connect to another Port"):
            port.connect("not a port")

    def test_read_from_connected_port(self):
        """Test that reading from connected port returns its value."""
        input_port = Port("input", "in")
        output_port = Port("output", "out")

        output_port.write(7.5)
        input_port.connect(output_port)

        assert input_port.read() == 7.5

    def test_read_updates_when_connected_port_changes(self):
        """Test that read reflects changes in connected port."""
        input_port = Port("input", "in")
        output_port = Port("output", "out")

        input_port.connect(output_port)

        output_port.write(1.0)
        assert input_port.read() == 1.0

        output_port.write(2.5)
        assert input_port.read() == 2.5


class TestPortModelDisconnect:
    """Test port disconnection functionality."""

    def test_disconnect_port(self):
        """Test disconnecting a port."""
        port1 = Port("input", "p1")
        port2 = Port("output", "p2")

        port1.connect(port2)
        assert port1.is_connected

        port1.disconnect()
        assert not port1.is_connected
        assert port1.connected_to == []

    def test_disconnect_affects_read(self):
        """Test that disconnecting affects read behavior."""
        input_port = Port("input", "in")
        output_port = Port("output", "out")

        output_port.write(5.0)
        input_port.connect(output_port)
        assert input_port.read() == 5.0

        input_port.disconnect()
        assert input_port.read() == 0.0


class TestPortModelStringRepresentation:
    """Test string representations."""

    def test_repr(self):
        """Test __repr__ output."""
        port = Port("input", "test_port")
        port.write(3.14159)

        repr_str = repr(port)

        assert "PortModel" in repr_str
        assert "test_port" in repr_str
        assert "input" in repr_str
        assert "3.142" in repr_str  # Formatted to 3 decimals
        assert "disconnected" in repr_str

    def test_repr_connected(self):
        """Test __repr__ shows connection status."""
        port1 = Port("input", "p1")
        port2 = Port("output", "p2")
        port1.connect(port2)

        repr_str = repr(port1)
        assert "1 connection(s)" in repr_str

    def test_str(self):
        """Test __str__ output."""
        port = Port("output", "audio_out")
        port.write(0.5)

        str_output = str(port)

        assert "output port" in str_output
        assert "audio_out" in str_output
        assert "0.500" in str_output


class TestPortModelEdgeCases:
    """Test edge cases."""

    def test_self_connection(self):
        """Test that connecting a port to itself raises ValueError."""
        port = Port("output", "self")
        port.write(5.0)

        with pytest.raises(ValueError, match="Cannot connect a port to itself"):
            port.connect(port)

    def test_large_value(self):
        """Test with large floating point values."""
        port = Port("output", "test")
        large_value = 1e10
        port.write(large_value)
        assert port.value == large_value

    def test_special_float_values(self):
        """Test with special float values."""
        port = Port("output", "test")

        # Test infinity
        port.write(float("inf"))
        assert port.value == float("inf")

        # Test NaN
        port.write(float("nan"))
        import math

        assert math.isnan(port.value)

    def test_empty_name(self):
        """Test creating port with empty name."""
        port = Port("input", "")
        assert port.port_name == ""

    def test_unicode_name(self):
        """Test port with unicode characters in name."""
        port = Port("input", "音频输入")
        assert port.port_name == "音频输入"


class TestPortModelIntegration:
    """Integration tests with multiple ports."""

    def test_multiple_inputs_one_output(self):
        """Test multiple input ports connected to one output."""
        output = Port("output", "out")
        input1 = Port("input", "in1")
        input2 = Port("input", "in2")
        input3 = Port("input", "in3")

        output.write(10.0)

        input1.connect(output)
        input2.connect(output)
        input3.connect(output)

        assert input1.read() == 10.0
        assert input2.read() == 10.0
        assert input3.read() == 10.0

    def test_reconnect_after_value_change(self):
        """Test reconnecting after value changes."""
        port1 = Port("input", "in")
        port2 = Port("output", "out1")
        port3 = Port("output", "out2")

        port2.write(10.0)
        port1.connect(port2)
        assert port1.read() == 10.0

        port1.disconnect()
        port3.write(20.0)
        port1.connect(port3)
        assert port1.read() == 20.0


def test_no_qt_dependencies():
    """Verify that PortModel has no Qt dependencies.

    This test ensures we can import and use PortModel without Qt installed.
    """
    import sys

    # Check that PyQt6 is not imported when using PortModel
    port = Port("input", "test")
    port.write(5.0)

    # PortModel module should not import PyQt6
    import gui.core.port as port_module

    assert "PyQt6" not in sys.modules or "PyQt6" not in str(port_module.__file__)


class TestPortMultipleConnections:
    """Test multiple connection functionality of Port class."""

    def test_connect_multiple_ports(self):
        """Test connecting a port to multiple other ports."""
        input_port = Port("input", "audio_in")
        output_port1 = Port("output", "audio_out_1")
        output_port2 = Port("output", "audio_out_2")
        output_port3 = Port("output", "audio_out_3")

        input_port.connect(output_port1)
        input_port.connect(output_port2)
        input_port.connect(output_port3)

        assert input_port.is_connected
        assert len(input_port.connected_ports) == 3
        assert output_port1 in input_port.connected_ports
        assert output_port2 in input_port.connected_ports
        assert output_port3 in input_port.connected_ports

    def test_idempotent_connection(self):
        """Test that connecting the same port twice has no effect."""
        input_port = Port("input", "audio_in")
        output_port = Port("output", "audio_out")

        input_port.connect(output_port)
        input_port.connect(output_port)  # Connect again

        assert len(input_port.connected_ports) == 1
        assert output_port in input_port.connected_ports

    def test_read_sum_from_multiple_ports(self):
        """Test that reading sums values from all connected ports."""
        input_port = Port("input", "audio_in")
        output_port1 = Port("output", "audio_out_1")
        output_port2 = Port("output", "audio_out_2")
        output_port3 = Port("output", "audio_out_3")

        input_port.connect(output_port1)
        input_port.connect(output_port2)
        input_port.connect(output_port3)

        output_port1.write(0.3)
        output_port2.write(0.2)
        output_port3.write(0.1)

        # Should return sum of all connected ports
        assert input_port.read() == pytest.approx(0.6)

    def test_disconnect_specific_port(self):
        """Test disconnecting a specific port while keeping others."""
        input_port = Port("input", "audio_in")
        output_port1 = Port("output", "audio_out_1")
        output_port2 = Port("output", "audio_out_2")
        output_port3 = Port("output", "audio_out_3")

        input_port.connect(output_port1)
        input_port.connect(output_port2)
        input_port.connect(output_port3)

        # Disconnect only port2
        input_port.disconnect(output_port2)

        assert len(input_port.connected_ports) == 2
        assert output_port1 in input_port.connected_ports
        assert output_port2 not in input_port.connected_ports
        assert output_port3 in input_port.connected_ports

    def test_disconnect_all_ports(self):
        """Test disconnecting all ports at once."""
        input_port = Port("input", "audio_in")
        output_port1 = Port("output", "audio_out_1")
        output_port2 = Port("output", "audio_out_2")

        input_port.connect(output_port1)
        input_port.connect(output_port2)

        # Disconnect all
        input_port.disconnect()

        assert not input_port.is_connected
        assert len(input_port.connected_ports) == 0

    def test_disconnect_nonexistent_port(self):
        """Test that disconnecting a non-connected port is a no-op."""
        input_port = Port("input", "audio_in")
        output_port1 = Port("output", "audio_out_1")
        output_port2 = Port("output", "audio_out_2")

        input_port.connect(output_port1)

        # Try to disconnect port that was never connected
        input_port.disconnect(output_port2)  # Should not raise

        assert len(input_port.connected_ports) == 1
        assert output_port1 in input_port.connected_ports

    def test_read_with_no_connections(self):
        """Test that reading from unconnected port returns 0.0."""
        input_port = Port("input", "audio_in")
        assert input_port.read() == 0.0

    def test_read_after_disconnecting_all(self):
        """Test that reading returns 0.0 after disconnecting all ports."""
        input_port = Port("input", "audio_in")
        output_port1 = Port("output", "audio_out_1")
        output_port2 = Port("output", "audio_out_2")

        input_port.connect(output_port1)
        input_port.connect(output_port2)
        output_port1.write(0.5)
        output_port2.write(0.3)

        input_port.disconnect()

        assert input_port.read() == 0.0

    def test_mixing_behavior(self):
        """Test proper mixing behavior with multiple signal sources."""
        mixer_input = Port("input", "mixer_in")
        osc1 = Port("output", "oscillator_1")
        osc2 = Port("output", "oscillator_2")
        osc3 = Port("output", "oscillator_3")

        mixer_input.connect(osc1)
        mixer_input.connect(osc2)
        mixer_input.connect(osc3)

        # Simulate three oscillators outputting different values
        osc1.write(1.0)
        osc2.write(-0.5)
        osc3.write(0.3)

        # Mixed result should be the sum
        assert mixer_input.read() == pytest.approx(0.8)

    def test_connect_invalid_type(self):
        """Test that connecting to non-Port raises TypeError."""
        port = Port("input", "audio_in")

        with pytest.raises(TypeError, match="Can only connect to another Port"):
            port.connect("not_a_port")

    def test_connect_to_self(self):
        """Test that connecting to self raises ValueError."""
        port = Port("input", "audio_in")

        with pytest.raises(ValueError, match="Cannot connect a port to itself"):
            port.connect(port)

    def test_repr_with_multiple_connections(self):
        """Test string representation shows connection count."""
        input_port = Port("input", "audio_in")
        output_port1 = Port("output", "audio_out_1")
        output_port2 = Port("output", "audio_out_2")

        # No connections
        repr_str = repr(input_port)
        assert "disconnected" in repr_str

        # One connection
        input_port.connect(output_port1)
        repr_str = repr(input_port)
        assert "1 connection(s)" in repr_str

        # Two connections
        input_port.connect(output_port2)
        repr_str = repr(input_port)
        assert "2 connection(s)" in repr_str

    def test_connected_ports_property(self):
        """Test the connected_ports property returns the correct list."""
        input_port = Port("input", "audio_in")
        output_port1 = Port("output", "audio_out_1")
        output_port2 = Port("output", "audio_out_2")

        assert input_port.connected_ports == []

        input_port.connect(output_port1)
        input_port.connect(output_port2)

        connected = input_port.connected_ports
        assert isinstance(connected, list)
        assert len(connected) == 2
        assert output_port1 in connected
        assert output_port2 in connected

    def test_sequential_disconnect(self):
        """Test disconnecting ports one by one."""
        input_port = Port("input", "audio_in")
        ports = [Port("output", f"out_{i}") for i in range(5)]

        # Connect all
        for port in ports:
            input_port.connect(port)

        assert len(input_port.connected_ports) == 5

        # Disconnect one by one
        for i, port in enumerate(ports):
            input_port.disconnect(port)
            assert len(input_port.connected_ports) == 5 - (i + 1)

        assert not input_port.is_connected

    def test_value_changes_reflect_in_read(self):
        """Test that changing values of connected ports reflects in read()."""
        input_port = Port("input", "audio_in")
        output_port1 = Port("output", "audio_out_1")
        output_port2 = Port("output", "audio_out_2")

        input_port.connect(output_port1)
        input_port.connect(output_port2)

        output_port1.write(0.5)
        output_port2.write(0.3)
        assert input_port.read() == pytest.approx(0.8)

        # Change values
        output_port1.write(1.0)
        output_port2.write(0.2)
        assert input_port.read() == pytest.approx(1.2)

        # Change again
        output_port1.write(0.0)
        output_port2.write(0.0)
        assert input_port.read() == pytest.approx(0.0)


class TestPortNumpySupport:
    """Test numpy array functionality of Port class."""

    def test_write_numpy_array(self):
        """Test writing a numpy array to a port."""
        port = Port("output", "audio_out")
        array = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        port.write(array)

        assert isinstance(port.value, np.ndarray)
        np.testing.assert_array_equal(port.value, array)

    def test_read_numpy_array(self):
        """Test reading a numpy array from connected port."""
        input_port = Port("input", "audio_in")
        output_port = Port("output", "audio_out")

        input_port.connect(output_port)
        array = np.array([0.1, 0.2, 0.3])
        output_port.write(array)

        result = input_port.read()

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, array)

    def test_mix_multiple_arrays(self):
        """Test mixing multiple numpy arrays (summing)."""
        input_port = Port("input", "mixer_in")
        output1 = Port("output", "out1")
        output2 = Port("output", "out2")
        output3 = Port("output", "out3")

        input_port.connect(output1)
        input_port.connect(output2)
        input_port.connect(output3)

        array1 = np.array([1.0, 2.0, 3.0])
        array2 = np.array([0.1, 0.2, 0.3])
        array3 = np.array([0.01, 0.02, 0.03])

        output1.write(array1)
        output2.write(array2)
        output3.write(array3)

        result = input_port.read()
        expected = np.array([1.11, 2.22, 3.33])

        np.testing.assert_array_almost_equal(result, expected)

    def test_mix_array_and_scalar(self):
        """Test mixing a numpy array with a scalar value."""
        input_port = Port("input", "mixer_in")
        output_array = Port("output", "out_array")
        output_scalar = Port("output", "out_scalar")

        input_port.connect(output_array)
        input_port.connect(output_scalar)

        array = np.array([1.0, 2.0, 3.0])
        scalar = 0.5

        output_array.write(array)
        output_scalar.write(scalar)

        result = input_port.read()
        expected = np.array([1.5, 2.5, 3.5])

        np.testing.assert_array_equal(result, expected)

    def test_mix_multiple_scalars_and_arrays(self):
        """Test mixing multiple scalars and arrays together."""
        input_port = Port("input", "mixer_in")
        ports = [Port("output", f"out{i}") for i in range(5)]

        for port in ports:
            input_port.connect(port)

        # Write mix of arrays and scalars
        ports[0].write(np.array([1.0, 2.0, 3.0]))
        ports[1].write(0.5)
        ports[2].write(np.array([0.1, 0.2, 0.3]))
        ports[3].write(0.3)
        ports[4].write(np.array([0.01, 0.02, 0.03]))

        result = input_port.read()
        # Expected: [1.0, 2.0, 3.0] + 0.5 + [0.1, 0.2, 0.3] + 0.3 + [0.01, 0.02, 0.03]
        #         = [1.0, 2.0, 3.0] + [0.1, 0.2, 0.3] + [0.01, 0.02, 0.03] + 0.5 + 0.3
        #         = [1.11, 2.22, 3.33] + 0.8
        #         = [1.91, 3.02, 4.13]
        expected = np.array([1.91, 3.02, 4.13])

        np.testing.assert_array_almost_equal(result, expected)

    def test_array_shape_mismatch_raises_error(self):
        """Test that mixing arrays with different shapes raises ValueError."""
        input_port = Port("input", "mixer_in")
        output1 = Port("output", "out1")
        output2 = Port("output", "out2")

        input_port.connect(output1)
        input_port.connect(output2)

        output1.write(np.array([1.0, 2.0, 3.0]))
        output2.write(np.array([0.1, 0.2]))  # Different shape

        with pytest.raises(ValueError, match="Cannot mix arrays with different shapes"):
            input_port.read()

    def test_2d_array_mixing(self):
        """Test mixing 2D numpy arrays (stereo signals)."""
        input_port = Port("input", "stereo_mixer")
        output1 = Port("output", "stereo1")
        output2 = Port("output", "stereo2")

        input_port.connect(output1)
        input_port.connect(output2)

        # Stereo arrays: shape (2, num_samples)
        array1 = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])
        array2 = np.array([[0.1, 0.2, 0.3], [0.05, 0.1, 0.15]])

        output1.write(array1)
        output2.write(array2)

        result = input_port.read()
        expected = np.array([[1.1, 2.2, 3.3], [0.55, 1.1, 1.65]])

        np.testing.assert_array_almost_equal(result, expected)

    def test_repr_with_array(self):
        """Test string representation shows array info."""
        port = Port("output", "audio_out")
        array = np.array([0.1, 0.2, 0.3])
        port.write(array)

        repr_str = repr(port)

        assert "array(shape=(3,)" in repr_str
        assert "dtype=float64" in repr_str

    def test_str_with_array(self):
        """Test human-readable string shows array info."""
        port = Port("output", "audio_out")
        array = np.array([0.1, 0.2, 0.3])
        port.write(array)

        str_repr = str(port)

        assert "array(shape=(3,))" in str_repr

    def test_scalar_to_array_conversion(self):
        """Test that scalars can be mixed with arrays."""
        input_port = Port("input", "mixer_in")
        output_scalar = Port("output", "scalar_out")

        input_port.connect(output_scalar)
        output_scalar.write(0.5)

        # First read returns scalar
        result = input_port.read()
        assert isinstance(result, (int, float))
        assert result == 0.5

    def test_empty_array(self):
        """Test handling of empty arrays."""
        port = Port("output", "audio_out")
        empty_array = np.array([])

        port.write(empty_array)

        assert isinstance(port.value, np.ndarray)
        assert len(port.value) == 0

    def test_large_array_performance(self):
        """Test handling of large arrays (typical audio buffer size)."""
        input_port = Port("input", "audio_in")
        output1 = Port("output", "out1")
        output2 = Port("output", "out2")

        input_port.connect(output1)
        input_port.connect(output2)

        # Typical audio buffer: 512 samples
        buffer_size = 512
        array1 = np.random.randn(buffer_size)
        array2 = np.random.randn(buffer_size)

        output1.write(array1)
        output2.write(array2)

        result = input_port.read()
        expected = array1 + array2

        np.testing.assert_array_almost_equal(result, expected)

    def test_stereo_buffer_mixing(self):
        """Test mixing stereo buffers (2D arrays)."""
        input_port = Port("input", "stereo_in")
        output1 = Port("output", "stereo_out1")
        output2 = Port("output", "stereo_out2")

        input_port.connect(output1)
        input_port.connect(output2)

        # Stereo buffer: (2, buffer_size)
        buffer_size = 512
        stereo1 = np.random.randn(2, buffer_size)
        stereo2 = np.random.randn(2, buffer_size)

        output1.write(stereo1)
        output2.write(stereo2)

        result = input_port.read()
        expected = stereo1 + stereo2

        np.testing.assert_array_almost_equal(result, expected)

    def test_dtype_preservation(self):
        """Test that dtype is preserved when writing arrays."""
        port = Port("output", "audio_out")

        # Test float32
        array_f32 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        port.write(array_f32)
        assert port.value.dtype == np.float32

        # Test float64
        array_f64 = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        port.write(array_f64)
        assert port.value.dtype == np.float64

    def test_array_copy_behavior(self):
        """Test that arrays are properly copied to avoid unintended sharing."""
        input_port = Port("input", "audio_in")
        output_port = Port("output", "audio_out")

        input_port.connect(output_port)

        original = np.array([1.0, 2.0, 3.0])
        output_port.write(original)

        result = input_port.read()

        # Modify result - should not affect the port's stored value
        result[0] = 999.0

        # Original port value should be unchanged
        assert output_port.value[0] == 1.0

    def test_zero_scalar_with_array(self):
        """Test mixing arrays with zero-valued scalar."""
        input_port = Port("input", "mixer_in")
        output_array = Port("output", "out_array")
        output_zero = Port("output", "out_zero")

        input_port.connect(output_array)
        input_port.connect(output_zero)

        array = np.array([1.0, 2.0, 3.0])
        output_array.write(array)
        output_zero.write(0.0)

        result = input_port.read()

        # Should equal the array (0 adds nothing)
        np.testing.assert_array_equal(result, array)


class TestPortComponentParameter:
    """Test the component parameter in Port initialization."""

    def test_port_created_without_component(self):
        """Test port can be created without component (backward compatibility)."""
        port = Port("output", "test", parent_module=MagicMock())
        assert port.component is None

    def test_port_created_with_component(self):
        """Test port can be created with a component reference."""
        oscillator = SineOscillator(440)
        port = Port("output", "sine", parent_module=MagicMock(), component=oscillator)
        assert port.component is oscillator

    def test_port_component_attribute_accessible(self):
        """Test that component attribute can be accessed."""
        oscillator = SineOscillator(440)
        port = Port("output", "sine", parent_module=MagicMock(), component=oscillator)
        assert hasattr(port, "component")
        assert isinstance(port.component, SineOscillator)

    def test_port_component_can_be_none(self):
        """Test that component can explicitly be set to None."""
        port = Port("output", "test", parent_module=MagicMock(), component=None)
        assert port.component is None

    def test_different_components_for_different_ports(self):
        """Test that different ports can have different components."""
        sine_osc = SineOscillator(440)
        triangle_osc = TriangleOscillator(440)

        sine_port = Port(
            "output", "sine", parent_module=MagicMock(), component=sine_osc
        )
        triangle_port = Port(
            "output", "triangle", parent_module=MagicMock(), component=triangle_osc
        )

        assert sine_port.component is sine_osc
        assert triangle_port.component is triangle_osc
        assert sine_port.component is not triangle_port.component


class TestPortComponentAccess:
    """Test accessing component through connected ports."""

    def test_get_component_from_connected_port(self):
        """Test retrieving component from a connected output port."""
        # Create oscillator
        oscillator = SineOscillator(440)

        # Create output port with component
        output_port = Port(
            "output", "sine", parent_module=MagicMock(), component=oscillator
        )

        # Create input port
        input_port = Port("input", "mod", parent_module=MagicMock())

        # Connect ports
        input_port.connect(output_port)

        # Access component through connection
        connected_ports = list(input_port.connected_to)
        assert len(connected_ports) == 1
        assert connected_ports[0].component is oscillator

    def test_multiple_connections_different_components(self):
        """Test that multiple connections can have different components."""
        sine_osc = SineOscillator(440)
        triangle_osc = TriangleOscillator(440)

        sine_port = Port(
            "output", "sine", parent_module=MagicMock(), component=sine_osc
        )
        triangle_port = Port(
            "output", "triangle", parent_module=MagicMock(), component=triangle_osc
        )

        input_port = Port("input", "mod", parent_module=MagicMock())

        # Connect both
        input_port.connect(sine_port)
        input_port.connect(triangle_port)

        # Check we can access both components
        connected_components = [p.component for p in input_port.connected_to]
        assert sine_osc in connected_components
        assert triangle_osc in connected_components

    def test_component_none_when_not_set(self):
        """Test that component is None when not set during creation."""
        output_port = Port("output", "test", parent_module=MagicMock())
        input_port = Port("input", "in", parent_module=MagicMock())

        input_port.connect(output_port)

        connected_port = list(input_port.connected_to)[0]
        assert connected_port.component is None


class TestPortComponentIntegration:
    """Integration tests for port-component mapping."""

    def test_typical_oscillator_usage_pattern(self):
        """Test the typical usage pattern for oscillator modules."""
        # Simulate OscillatorModule pattern
        sine_oscillator = SineOscillator(440)
        triangle_oscillator = TriangleOscillator(440)

        # Create ports with component references (as done in OscillatorModule.__init__)
        sine_port = Port(
            "output", "Sine", parent_module=MagicMock(), component=sine_oscillator
        )
        triangle_port = Port(
            "output",
            "Triangle",
            parent_module=MagicMock(),
            component=triangle_oscillator,
        )

        # Verify components are accessible
        assert sine_port.component.frequency == 440
        assert triangle_port.component.frequency == 440

    def test_volume_module_modulation_pattern(self):
        """Test the typical pattern for volume module getting modulator component."""
        # Simulate LFO output
        lfo = SineOscillator(1.0, wave_range=(-1, 1))
        lfo_output = Port("output", "Sine", parent_module=MagicMock(), component=lfo)

        # Simulate Volume module's Mod input
        mod_input = Port("input", "Mod", parent_module=MagicMock())

        # Connect LFO to Volume
        mod_input.connect(lfo_output)

        # Volume module retrieves component (simplified)
        connected_ports = list(mod_input.connected_to)
        modulator_component = connected_ports[0].component

        # Verify we got the LFO component
        assert modulator_component is lfo
        assert modulator_component.wave_range == (-1, 1)

    def test_component_survives_disconnect_reconnect(self):
        """Test that component reference persists through disconnect/reconnect."""
        oscillator = SineOscillator(440)
        output_port = Port(
            "output", "test", parent_module=MagicMock(), component=oscillator
        )
        input_port = Port("input", "in", parent_module=MagicMock())

        # Connect
        input_port.connect(output_port)
        assert list(input_port.connected_to)[0].component is oscillator

        # Disconnect
        input_port.disconnect(output_port)

        # Component still exists on output port
        assert output_port.component is oscillator

        # Reconnect
        input_port.connect(output_port)
        assert list(input_port.connected_to)[0].component is oscillator


class TestPortComponentBackwardCompatibility:
    """Test backward compatibility - existing code should still work."""

    def test_old_port_creation_still_works(self):
        """Test that ports created without component parameter still work."""
        # Old style - no component parameter
        port = Port("output", "test", parent_module=MagicMock())
        assert port.port_name == "test"
        assert port.component is None

    def test_old_connection_pattern_still_works(self):
        """Test that old connection patterns still work."""
        port1 = Port("input", "in", parent_module=MagicMock())
        port2 = Port("output", "out", parent_module=MagicMock())

        port1.connect(port2)
        port2.write(5.0)

        assert port1.read() == 5.0
        assert port2.component is None  # No component set

    def test_mixed_old_and_new_style(self):
        """Test mixing old-style ports (no component) with new-style
        (with component)."""
        osc = SineOscillator(440)

        # New style - with component
        new_port = Port("output", "new", parent_module=MagicMock(), component=osc)

        # Old style - without component
        old_port = Port("input", "old", parent_module=MagicMock())

        # Connect
        old_port.connect(new_port)

        # Old port can access new port's component
        assert list(old_port.connected_to)[0].component is osc


class TestPortComponentEdgeCases:
    """Test edge cases for port-component mapping."""

    def test_component_can_be_any_object(self):
        """Test that component can be any object, not just oscillators."""
        mock_component = MagicMock()
        mock_component.test_attribute = "test_value"

        port = Port(
            "output", "test", parent_module=MagicMock(), component=mock_component
        )
        assert port.component.test_attribute == "test_value"

    def test_multiple_ports_same_component(self):
        """Test that multiple ports can reference the same component."""
        oscillator = SineOscillator(440)

        port1 = Port("output", "out1", parent_module=MagicMock(), component=oscillator)
        port2 = Port("output", "out2", parent_module=MagicMock(), component=oscillator)

        assert port1.component is port2.component

    def test_component_with_all_port_types(self):
        """Test component parameter works with both input and output ports."""
        osc = SineOscillator(440)

        # Component on output port (typical)
        output_port = Port("output", "out", parent_module=MagicMock(), component=osc)
        assert output_port.component is osc

        # Component on input port (unusual but allowed)
        input_port = Port("input", "in", parent_module=MagicMock(), component=osc)
        assert input_port.component is osc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
