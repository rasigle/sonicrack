"""Test cases for the Port class.

NOTE: These tests have been updated to work with the refactored architecture.
The Port class now uses PortWidget (UI) + Port (logic).

For pure logic tests without Qt, see tests/core/test_port_model.py
"""

from unittest.mock import MagicMock

import pytest

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

        # Check connection via model (connected_to property returns None in PortWidget)
        assert input_port.connected_to is output_port
        assert input_port.is_connected

    def test_connect_changes_connected_to(self):
        """Test that connect sets the connected_to attribute."""
        port1 = Port("input", "port1", mock_parent())
        port2 = Port("output", "port2", mock_parent())

        assert not port1.is_connected
        port1.connect(port2)
        assert port1.is_connected
        assert port1.connected_to is port2

    def test_connect_multiple_times(self):
        """Test that connecting to different ports updates connection."""
        port1 = Port("input", "port1", mock_parent())
        port2 = Port("output", "port2", mock_parent())
        port3 = Port("output", "port3", mock_parent())

        port1.connect(port2)
        assert port1.connected_to is port2

        port1.connect(port3)
        assert port1.connected_to is port3

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
        output_port.write(5.0)    # Output port has different value
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
        assert port1.connected_to is None

    def test_disconnect_clears_connection(self):
        """Test that disconnect clears the connected_to attribute."""
        port1 = Port("input", "port1", mock_parent())
        port2 = Port("output", "port2", mock_parent())

        port1.connect(port2)
        port1.disconnect()

        assert port1.connected_to is None

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
        """Test connecting a port to itself."""
        port = Port("input", "self", mock_parent())
        port.write(5.0)
        port.connect(port)

        # Port is connected to itself
        assert port.connected_to is port
        assert port.is_connected
        assert port.read() == 5.0

    def test_circular_connection(self):
        """Test circular connections (A->B->A)."""
        port_a = Port("input", "A", mock_parent())
        port_b = Port("input", "B", mock_parent())

        port_a.connect(port_b)
        port_b.connect(port_a)

        port_a.write(10.0)

        # A reads from B, B reads from A (circular)
        # This is allowed but may cause issues in real usage
        assert port_a.connected_to is port_b
        assert port_b.connected_to is port_a

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
        port.write(float('inf'))
        assert port.value == float('inf')

        # Test with negative infinity
        port.write(float('-inf'))
        assert port.value == float('-inf')

        # Test with NaN
        port.write(float('nan'))
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
    assert input_port.connected_to == output_port

    # Disconnect
    input_port.disconnect()
    assert not input_port.is_connected
    assert input_port.connected_to is None


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

    with pytest.raises(TypeError, match="Can only connect to PortModel"):
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
        assert port.connected_to is None
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

        assert input_port.connected_to is output_port
        assert input_port.is_connected

    def test_connect_invalid_type_raises_error(self):
        """Test that connecting to non-PortModel raises TypeError."""
        port = Port("input", "test")

        with pytest.raises(TypeError, match="Can only connect to PortModel"):
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
        assert port1.connected_to is None

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
        assert "connected" in repr_str

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
        """Test connecting a port to itself."""
        port = Port("output", "self")
        port.write(5.0)
        port.connect(port)

        assert port.connected_to is port
        assert port.is_connected
        assert port.read() == 5.0

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
        port.write(float('inf'))
        assert port.value == float('inf')

        # Test NaN
        port.write(float('nan'))
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
    assert 'PyQt6' not in sys.modules or 'PyQt6' not in str(port_module.__file__)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
