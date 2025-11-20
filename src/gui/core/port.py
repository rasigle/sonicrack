"""Port data model (pure Python logic, no Qt dependencies).

This module provides the core data model for signal ports without any UI concerns.
The UI layer is in src.gui.widgets.port_widget.PortWidget.

Architecture:
- PortModel (this file): Data and connection logic
- PortWidget (port_widget.py): Qt graphics and interaction
- Port (port.py): Legacy wrapper for backward compatibility
"""

from __future__ import annotations
from typing import Optional


class Port:
    """Data model for a signal port.

    Handles only the logic and data:
    - Value storage and retrieval
    - Connection management
    - Type information

    No UI concerns - can be tested without Qt or any graphics framework.

    Example:
        >>> input_port = Port("input", "audio_in")
        >>> output_port = Port("output", "audio_out")
        >>> input_port.connect(output_port)
        >>> output_port.write(0.5)
        >>> input_port.read()
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
        self.connected_to: Optional[Port] = None

    def connect(self, other: Port) -> None:
        """Connect this port to another port.

        Args:
            other: The port to connect to

        Raises:
            TypeError: If other is not a PortModel instance
        """
        if not isinstance(other, Port):
            raise TypeError(
                f"Can only connect to PortModel, got {type(other).__name__}"
            )
        self.connected_to = other

    def disconnect(self) -> None:
        """Disconnect from any connected port."""
        self.connected_to = None

    def read(self) -> float:
        """Read value from connected port.

        Returns:
            Value from connected port, or 0.0 if not connected
        """
        if self.connected_to:
            return self.connected_to.value
        return 0.0

    def write(self, value: float) -> None:
        """Write a value to this port.

        Args:
            value: The value to write (will be converted to float)
        """
        self.value = value

    @property
    def is_connected(self) -> bool:
        """Check if port is connected to another port.

        Returns:
            True if connected, False otherwise
        """
        return self.connected_to is not None

    def __repr__(self) -> str:
        """String representation for debugging.

        Returns:
            Debug string with port details
        """
        conn_status = "connected" if self.is_connected else "disconnected"
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
