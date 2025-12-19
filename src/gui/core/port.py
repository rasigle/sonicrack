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
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


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
        parent_module=None,  # Reference to the module that owns this port
        component=None,  # Reference to the engine component (e.g., SineOscillator)
    ):
        """Initialize a port model.

        Args:
            port_type: Type of port ("input" or "output")
            port_name: Name identifier for the port
            index: Optional index for ordering multiple ports
            parent_module: Reference to the module that owns this port
            component: Optional reference to the engine component associated with this
                port (e.g., SineOscillator for a "Sine" output port)
        """
        self.port_type: str = port_type
        self.port_name: str = port_name
        self.parent_module = parent_module
        self.component = component  # Direct reference to engine component

        self.connected_to: list[Port] = []

        self.index: int = index

        # Data state - can be scalar or numpy array
        self.value: float | np.ndarray = 0.0


    def connect(self, other: Port) -> None:
        """Connect this port to another port (bidirectional).

        Idempotent: connecting the same port twice has no effect.
        Creates a bidirectional connection so both ports know they're connected.

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

        logger.debug(f"Port connected: {self.port_name} <-> {other.port_name}")

        # Bidirectional connection: both ports track the connection
        self.connected_to.append(other)
        if self not in other.connected_to:
            other.connected_to.append(self)

    def disconnect(self, other: Port | None = None) -> None:
        """Disconnect from a specific connected port (bidirectional), or all if other
        is None.

        Args:
            other: Specific port to disconnect from, or None to disconnect all
        """
        if other is None:
            if self.connected_to:
                logger.debug(f"Port disconnected (all): {self.port_name}")
                # Remove this port from all connected ports
                for connected_port in self.connected_to:
                    try:
                        connected_port.connected_to.remove(self)
                    except ValueError:
                        pass
            self.connected_to.clear()
            # Clear port data to prevent stale audio
            self.clear()
            return

        try:
            self.connected_to.remove(other)
            # Also remove from other side (bidirectional)
            if self in other.connected_to:
                other.connected_to.remove(self)
            logger.debug(
                f"Port disconnected: {self.port_name} <-/-> {other.port_name}"
            )
            # Clear port data to prevent stale audio
            self.clear()
        except ValueError:
            # Port not in list; no-op
            pass

    def clear(self) -> None:
        """Clear the port's data value.

        This is called when disconnecting to prevent stale audio data
        from continuing to play after a module is deleted.
        """
        self.value = 0.0
        logger.debug(f"Port data cleared: {self.port_name}")

    def read(self, num_samples: int | None = None) -> float | np.ndarray:
        """Read value from connected ports, triggering upstream generation if needed.

        This method now supports the pull-based architecture by:
        1. Triggering upstream modules to generate samples (if num_samples provided)
        2. Reading and mixing values from all connected ports

        Args:
            num_samples: Number of samples to request from upstream modules.
                If None, just returns cached values (legacy behavior).

        Returns:
            Sum of values from connected ports (float or np.ndarray), or 0.0 if not
            connected
        """
        if not self.connected_to:
            if num_samples is not None and num_samples > 0:
                # Return zeros with correct shape for audio processing
                return np.zeros(num_samples, dtype=np.float32)
            return 0.0

        # Trigger upstream module generation if num_samples is provided
        if num_samples is not None:
            for connected_port in self.connected_to:
                # Safety check: ensure parent_module still exists (not deleted during
                # shutdown)
                try:
                    if not connected_port.parent_module:
                        continue

                    # Skip non-processing modules (like visualizers)
                    if hasattr(connected_port.parent_module, 'is_processing_module') and \
                       not connected_port.parent_module.is_processing_module:
                        continue

                    if hasattr(connected_port.parent_module, "ensure_samples_ready"):
                        # Pull-based: ask upstream module to generate samples
                        connected_port.parent_module.ensure_samples_ready(num_samples)
                    elif hasattr(connected_port.parent_module, "process"):
                        # Fallback: call process() for modules not yet updated
                        connected_port.parent_module.process(num_samples)
                except (RuntimeError, AttributeError):
                    # Module was deleted (Qt cleanup during shutdown) - skip it
                    continue

        # Collect all values
        values = [p.value for p in self.connected_to]

        # Check if any value is a numpy array
        has_arrays = any(isinstance(v, np.ndarray) for v in values)

        if not has_arrays:
            # All scalars - simple sum
            return sum(values)

        # Mixed or all arrays - need to handle carefully
        # If num_samples was requested, ensure all arrays match that size
        result = None
        for value in values:
            if result is None:
                # Initialize with first value
                if isinstance(value, np.ndarray):
                    # If num_samples specified and array size doesn't match, adjust it
                    if num_samples is not None and len(value) != num_samples:
                        if len(value) < num_samples:
                            # Pad with zeros
                            padded = np.zeros(num_samples, dtype=value.dtype)
                            padded[:len(value)] = value
                            result = padded
                        else:
                            # Truncate to requested size
                            result = value[:num_samples].copy()
                    else:
                        result = value.copy()
                else:
                    result = np.array(value)
            else:
                # Add subsequent values
                if isinstance(value, np.ndarray):
                    # Adjust array size to match result if needed
                    if result.shape != value.shape:
                        if len(value) < len(result):
                            # Pad with zeros
                            padded = np.zeros(len(result), dtype=value.dtype)
                            padded[:len(value)] = value
                            result = result + padded
                        else:
                            # Truncate to match result size
                            result = result + value[:len(result)]
                    else:
                        result = result + value
                else:
                    # Scalar - broadcast across array
                    result = result + value

        return result if result is not None else 0.0

    def write(self, value: float | np.ndarray) -> None:
        """Write a value to this port.

        Accepts both scalar float values and numpy arrays for batch processing.
        Also notifies any connected visualization modules so they can observe
        the samples without pulling.

        Args:
            value: The value to write (float or numpy array)
        """
        if isinstance(value, np.ndarray):
            self.value = value
        elif isinstance(value, (float, int)):
            self.value = float(value)
        elif isinstance(value, Sequence):
            self.value = np.array(value, dtype=np.float32)
        else:
            raise TypeError(
                f"Port write value must be float or np.ndarray, "
                f"got {type(value).__name__}"
            )
        # Note: Visualizers poll port.value directly - no notifications needed!
        # This ensures ZERO interference with audio thread


    def peek_recent(self, num_samples: int | None = None) -> float | np.ndarray:
        """Peek at recent samples without triggering upstream generation.

        This is useful for visualization modules that want to read samples
        without affecting the audio processing pipeline.

        Args:
            num_samples: Optional hint for desired sample count (not enforced)

        Returns:
            Current cached value from the port
        """
        return self.read(num_samples=None)  # Read without triggering generation

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
