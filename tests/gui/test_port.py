"""Test cases for the Port class."""

import pytest
from src.gui.port import Port


class TestPortBasics:
    """Test basic Port functionality."""

    def test_port_creation(self):
        """Test that a port can be created with a name."""
        port = Port("test_port")
        assert port.name == "test_port"
        assert port.value == 0.0
        assert port.connected_to is None
        assert not port.is_connected

    def test_port_name_storage(self):
        """Test that port stores its name correctly."""
        port1 = Port("input")
        port2 = Port("output")
        assert port1.name == "input"
        assert port2.name == "output"

    def test_initial_value_is_zero(self):
        """Test that port initializes with value 0.0."""
        port = Port("test")
        assert port.value == 0.0


class TestPortWriteRead:
    """Test writing and reading port values."""

    def test_write_value(self):
        """Test writing a value to a port."""
        port = Port("test")
        port.write(5.0)
        assert port.value == 5.0

    def test_write_multiple_values(self):
        """Test writing multiple values overwrites previous value."""
        port = Port("test")
        port.write(1.0)
        assert port.value == 1.0
        port.write(2.0)
        assert port.value == 2.0
        port.write(3.5)
        assert port.value == 3.5

    def test_read_disconnected_port_returns_zero(self):
        """Test that reading a disconnected port returns 0.0."""
        port = Port("test")
        assert port.read() == 0.0

    def test_read_disconnected_port_with_value(self):
        """Test that reading disconnected port returns 0.0 even if port has value."""
        port = Port("test")
        port.write(5.0)
        # Reading disconnected port returns 0.0, not the port's own value
        assert port.read() == 0.0

    def test_write_negative_value(self):
        """Test writing negative values."""
        port = Port("test")
        port.write(-3.5)
        assert port.value == -3.5

    def test_write_zero(self):
        """Test writing zero value."""
        port = Port("test")
        port.write(10.0)
        port.write(0.0)
        assert port.value == 0.0


class TestPortConnection:
    """Test port connection functionality."""

    def test_connect_two_ports(self):
        """Test connecting one port to another."""
        input_port = Port("input")
        output_port = Port("output")

        input_port.connect(output_port)

        assert input_port.connected_to is output_port
        assert input_port.is_connected

    def test_connect_changes_connected_to(self):
        """Test that connect sets the connected_to attribute."""
        port1 = Port("port1")
        port2 = Port("port2")

        assert port1.connected_to is None
        port1.connect(port2)
        assert port1.connected_to is port2

    def test_connect_multiple_times(self):
        """Test that connecting to different ports updates connection."""
        port1 = Port("port1")
        port2 = Port("port2")
        port3 = Port("port3")

        port1.connect(port2)
        assert port1.connected_to is port2

        port1.connect(port3)
        assert port1.connected_to is port3

    def test_is_connected_property(self):
        """Test the is_connected property."""
        port1 = Port("port1")
        port2 = Port("port2")

        assert not port1.is_connected

        port1.connect(port2)
        assert port1.is_connected


class TestPortReadConnected:
    """Test reading values from connected ports."""

    def test_read_from_connected_port(self):
        """Test that reading from connected port returns its value."""
        input_port = Port("input")
        output_port = Port("output")

        output_port.write(7.5)
        input_port.connect(output_port)

        assert input_port.read() == 7.5

    def test_read_updates_when_connected_port_changes(self):
        """Test that read reflects changes in connected port."""
        input_port = Port("input")
        output_port = Port("output")

        input_port.connect(output_port)

        output_port.write(1.0)
        assert input_port.read() == 1.0

        output_port.write(2.5)
        assert input_port.read() == 2.5

        output_port.write(0.0)
        assert input_port.read() == 0.0

    def test_read_ignores_own_value_when_connected(self):
        """Test that reading connected port ignores the reading port's own value."""
        input_port = Port("input")
        output_port = Port("output")

        input_port.write(100.0)  # Input port has its own value
        output_port.write(5.0)    # Output port has different value
        input_port.connect(output_port)

        # Should read from connected port, not own value
        assert input_port.read() == 5.0

    def test_read_with_zero_value_connection(self):
        """Test reading from connected port that has zero value."""
        input_port = Port("input")
        output_port = Port("output")

        output_port.write(0.0)
        input_port.connect(output_port)

        assert input_port.read() == 0.0


class TestPortDisconnect:
    """Test port disconnection functionality."""

    def test_disconnect_port(self):
        """Test disconnecting a port."""
        port1 = Port("port1")
        port2 = Port("port2")

        port1.connect(port2)
        assert port1.is_connected

        port1.disconnect()
        assert not port1.is_connected
        assert port1.connected_to is None

    def test_disconnect_clears_connection(self):
        """Test that disconnect clears the connected_to attribute."""
        port1 = Port("port1")
        port2 = Port("port2")

        port1.connect(port2)
        port1.disconnect()

        assert port1.connected_to is None

    def test_disconnect_affects_read(self):
        """Test that disconnecting affects read behavior."""
        input_port = Port("input")
        output_port = Port("output")

        output_port.write(5.0)
        input_port.connect(output_port)
        assert input_port.read() == 5.0

        input_port.disconnect()
        assert input_port.read() == 0.0  # Returns 0.0 when disconnected

    def test_disconnect_already_disconnected(self):
        """Test that disconnecting an already disconnected port is safe."""
        port = Port("test")
        assert not port.is_connected

        # Should not raise exception
        port.disconnect()
        assert not port.is_connected

    def test_disconnect_after_reconnect(self):
        """Test disconnecting after reconnecting to different port."""
        port1 = Port("port1")
        port2 = Port("port2")
        port3 = Port("port3")

        port1.connect(port2)
        port1.connect(port3)
        port1.disconnect()

        assert not port1.is_connected


class TestPortEdgeCases:
    """Test edge cases and special scenarios."""

    def test_self_connection(self):
        """Test connecting a port to itself."""
        port = Port("self")
        port.write(5.0)
        port.connect(port)

        # Port is connected to itself
        assert port.connected_to is port
        assert port.is_connected
        assert port.read() == 5.0

    def test_circular_connection(self):
        """Test circular connections (A->B->A)."""
        port_a = Port("A")
        port_b = Port("B")

        port_a.connect(port_b)
        port_b.connect(port_a)

        port_a.write(10.0)

        # A reads from B, B reads from A (circular)
        # This is allowed but may cause issues in real usage
        assert port_a.connected_to is port_b
        assert port_b.connected_to is port_a

    def test_chain_connection(self):
        """Test chain of connections (A->B->C)."""
        port_a = Port("A")
        port_b = Port("B")
        port_c = Port("C")

        port_c.write(42.0)
        port_b.connect(port_c)
        port_a.connect(port_b)

        # A reads from B, which has its own value (not C's)
        assert port_a.read() == port_b.value

        # B reads from C
        assert port_b.read() == 42.0

    def test_large_value(self):
        """Test with large floating point values."""
        port = Port("test")
        large_value = 1e10
        port.write(large_value)
        assert port.value == large_value

    def test_small_value(self):
        """Test with very small floating point values."""
        port = Port("test")
        small_value = 1e-10
        port.write(small_value)
        assert port.value == small_value

    def test_special_float_values(self):
        """Test with special float values."""
        port = Port("test")

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
        port = Port("")
        assert port.name == ""

    def test_special_characters_in_name(self):
        """Test port names with special characters."""
        port1 = Port("port-1")
        port2 = Port("port_2")
        port3 = Port("port.3")

        assert port1.name == "port-1"
        assert port2.name == "port_2"
        assert port3.name == "port.3"

    def test_unicode_name(self):
        """Test port with unicode characters in name."""
        port = Port("测试端口")
        assert port.name == "测试端口"

    def test_long_name(self):
        """Test port with very long name."""
        long_name = "a" * 1000
        port = Port(long_name)
        assert port.name == long_name


class TestPortIntegration:
    """Integration tests with multiple ports."""

    def test_multiple_inputs_one_output(self):
        """Test multiple input ports connected to one output."""
        output = Port("output")
        input1 = Port("input1")
        input2 = Port("input2")
        input3 = Port("input3")

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
        output = Port("output")
        input1 = Port("input1")
        input2 = Port("input2")

        output.write(5.0)
        input1.connect(output)
        input2.connect(output)

        input1.disconnect()

        assert input1.read() == 0.0
        assert input2.read() == 5.0

    def test_reconnect_after_value_change(self):
        """Test reconnecting after value changes."""
        port1 = Port("port1")
        port2 = Port("port2")
        port3 = Port("port3")

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
        port = Port("test")
        port.write(5)  # Integer
        assert port.value == 5.0
        assert isinstance(port.value, (int, float))

    def test_write_float_value(self):
        """Test writing explicit float values."""
        port = Port("test")
        port.write(5.5)
        assert port.value == 5.5
        assert isinstance(port.value, float)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

