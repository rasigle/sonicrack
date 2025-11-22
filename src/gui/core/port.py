"""Port data model (pure Python logic, no Qt dependencies).

This module provides the core data model for signal ports without any UI concerns.
The UI layer is in src.gui.widgets.port_widget.PortWidget.

Architecture:
- PortModel (this file): Data and connection logic
- PortWidget (port_widget.py): Qt graphics and interaction
- Port (port.py): Legacy wrapper for backward compatibility
"""

from __future__ import annotations

import logging

import numpy as np


class Port:
    """Data model for a signal port supporting multiple connections and numpy arrays.

    Handles only the logic and data:
    - Value storage and retrieval (float or numpy array)
    - Connection management (supports multiple connections)
    - Type information

    Supports both scalar (float) and vectorized (numpy array) signal processing.
    No UI concerns - can be tested without Qt or any graphics framework.

    Example (scalar):
        >>> input_port = Port("input", "audio_in")
        >>> output_port1 = Port("output", "audio_out_1")
        >>> output_port2 = Port("output", "audio_out_2")
        >>> input_port.connect(output_port1)
        >>> input_port.connect(output_port2)
        >>> output_port1.write(0.3)
        >>> output_port2.write(0.2)
        >>> input_port.read()  # Returns sum: 0.5
        0.5

    Example (vectorized):
        >>> input_port = Port("input", "audio_in")
        >>> output_port = Port("output", "audio_out")
        >>> input_port.connect(output_port)
        >>> output_port.write(np.array([0.1, 0.2, 0.3]))
        >>> input_port.read()
        array([0.1, 0.2, 0.3])
    """

    def __init__(
        self,
        port_type: str,  # "input" or "output"
        port_name: str,
        index: int = 0,
    ):
        """Initialize a port model.

        Args:
            port_type: Type of port ("input" or "output")
            port_name: Name identifier for the port
            index: Optional index for ordering multiple ports
        """
        self.port_type = port_type
        self.port_name = port_name

        self.connected_to: list[Port] = []

        self.index = index

        # Data state - can be scalar or numpy array
        self.value: float | np.ndarray = 0.0

    def connect(self, other: Port) -> None:
        """Connect this port to another port.

        Idempotent: connecting the same port twice has no effect.

        Args:
            other: The port to connect to

        Raises:
            TypeError: If other is not a Port instance
            ValueError: If attempting to connect to self
        """
        if not isinstance(other, Port):
            raise TypeError(
                f"Can only connect to another Port, got {type(other).__name__}"
            )
        if other is self:
            raise ValueError("Cannot connect a port to itself")
        if other in self.connected_to:
            return  # Already connected, no-op

        logging.debug(f"Port connected: {self.port_name} -> {other.port_name}")
        self.connected_to.append(other)

    def disconnect(self, other: Port | None = None) -> None:
        """Disconnect from a specific connected port, or all if other is None.

        Args:
            other: Specific port to disconnect from, or None to disconnect all
        """
        if other is None:
            if self.connected_to:
                logging.debug(f"Port disconnected (all): {self.port_name}")
            self.connected_to.clear()
            return

        try:
            self.connected_to.remove(other)
            logging.debug(f"Port disconnected: {self.port_name} -/-> {other.port_name}")
        except ValueError:
            # Port not in list; no-op
            pass

    def read(self) -> float | np.ndarray:
        """Read value from connected ports.

        Returns the sum of values from all connected ports (mixing).
        Supports both scalar and numpy array values.

        For arrays, they must all have the same shape, otherwise a ValueError is raised.

        Returns:
            Sum of values from connected ports (float or np.ndarray), or 0.0 if not connected
        """
        if not self.connected_to:
            return 0.0

        # Collect all values
        values = [p.value for p in self.connected_to]

        # Check if any value is a numpy array
        has_arrays = any(isinstance(v, np.ndarray) for v in values)

        if not has_arrays:
            # All scalars - simple sum
            return sum(values)

        # Mixed or all arrays - need to handle carefully
        result = None
        for value in values:
            if result is None:
                # Initialize with first value
                result = np.array(value) if not isinstance(value, np.ndarray) else value.copy()
            else:
                # Add subsequent values
                if isinstance(value, np.ndarray):
                    if result.shape != value.shape:
                        raise ValueError(
                            f"Cannot mix arrays with different shapes: {result.shape} vs {value.shape}"
                        )
                    result = result + value
                else:
                    # Scalar - broadcast across array
                    result = result + value

        return result if result is not None else 0.0

    def write(self, value: float | np.ndarray) -> None:
        """Write a value to this port.

        Accepts both scalar float values and numpy arrays for batch processing.

        Args:
            value: The value to write (float or numpy array)
        """
        if isinstance(value, np.ndarray):
            self.value = value
        else:
            self.value = float(value)

    @property
    def is_connected(self) -> bool:
        """Check if port is connected to at least one other port.

        Returns:
            True if connected to one or more ports, False otherwise
        """
        return len(self.connected_to) > 0

    @property
    def connected_ports(self) -> list[Port]:
        """Get the list of currently connected ports.

        Returns:
            List of connected Port instances
        """
        return self.connected_to

    def __repr__(self) -> str:
        """String representation for debugging.

        Returns:
            Debug string with port details
        """
        conn_count = len(self.connected_to)
        conn_status = f"{conn_count} connection(s)" if conn_count else "disconnected"

        # Format value display
        if isinstance(self.value, np.ndarray):
            value_str = f"array(shape={self.value.shape}, dtype={self.value.dtype})"
        else:
            value_str = f"{self.value:.3f}"

        return (
            f"PortModel(name='{self.port_name}', type='{self.port_type}', "
            f"value={value_str}, {conn_status})"
        )

    def __str__(self) -> str:
        """Human-readable string representation.

        Returns:
            Readable string
        """
        if isinstance(self.value, np.ndarray):
            value_str = f"array(shape={self.value.shape})"
        else:
            value_str = f"{self.value:.3f}"
        return f"{self.port_type} port '{self.port_name}' = {value_str}"
