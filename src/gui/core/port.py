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
from typing import List, Optional


class Port:
    """Data model for a signal port supporting multiple connections.

    Handles only the logic and data:
    - Value storage and retrieval
    - Connection management (supports multiple connections)
    - Type information

    No UI concerns - can be tested without Qt or any graphics framework.

    Example:
        >>> input_port = Port("input", "audio_in")
        >>> output_port1 = Port("output", "audio_out_1")
        >>> output_port2 = Port("output", "audio_out_2")
        >>> input_port.connect(output_port1)
        >>> input_port.connect(output_port2)
        >>> output_port1.write(0.3)
        >>> output_port2.write(0.2)
        >>> input_port.read()  # Returns sum: 0.5
        0.5
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
        self.index = index

        # Data state
        self.value: float = 0.0
        self.connected_to: List[Port] = []

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

    def disconnect(self, other: Optional[Port] = None) -> None:
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

    def read(self) -> float:
        """Read value from connected ports.

        Returns the sum of values from all connected ports (mixing).

        Returns:
            Sum of values from connected ports, or 0.0 if not connected
        """
        if not self.connected_to:
            return 0.0
        return sum(p.value for p in self.connected_to)

    def write(self, value: float) -> None:
        """Write a value to this port.

        Args:
            value: The value to write (will be converted to float)
        """
        self.value = float(value)

    @property
    def is_connected(self) -> bool:
        """Check if port is connected to at least one other port.

        Returns:
            True if connected to one or more ports, False otherwise
        """
        return len(self.connected_to) > 0

    @property
    def connected_ports(self) -> List[Port]:
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
        return (
            f"PortModel(name='{self.port_name}', type='{self.port_type}', "
            f"value={self.value:.3f}, {conn_status})"
        )

    def __str__(self) -> str:
        """Human-readable string representation.

        Returns:
            Readable string
        """
        return f"{self.port_type} port '{self.port_name}' = {self.value:.3f}"
