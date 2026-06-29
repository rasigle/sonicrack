"""Core signal-port data model with no Qt dependencies.

The UI layer wraps this class in ``src.gui.widgets.port_widget.PortWidget``.
Ports store current values, connection state, and passive tap history used by
visualization modules.
"""

from __future__ import annotations

import contextlib
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
        self._latest_buffer: float | np.ndarray = 0.0
        self._tap_history: np.ndarray | None = None
        self._tap_history_limit = 65536

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
                    with contextlib.suppress(ValueError):
                        connected_port.connected_to.remove(self)
            self.connected_to.clear()
            # Clear port data to prevent stale audio
            self.clear()
            return

        try:
            self.connected_to.remove(other)
            # Also remove from other side (bidirectional)
            if self in other.connected_to:
                other.connected_to.remove(self)
            logger.debug(f"Port disconnected: {self.port_name} <-/-> {other.port_name}")
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
        self._latest_buffer = 0.0
        self._tap_history = None
        logger.debug(f"Port data cleared: {self.port_name}")

    def read(self, num_samples: int | None = None) -> float | np.ndarray:
        """Read value from connected ports.

        During engine-owned renders, this delegates to the active render context.
        The context owns graph ordering and cached input values; ports no longer
        trigger upstream module processing themselves. Outside a render context,
        this method keeps legacy value-mixing behavior for tests and direct
        inspection.

        Args:
            num_samples: Optional number of samples expected by the caller.

        Returns:
            Sum of values from connected ports (float or np.ndarray), or 0.0 if not
            connected
        """
        render_context = None
        if num_samples is not None:
            try:
                from src.gui.audio_engine import get_active_render_context

                render_context = get_active_render_context()
            except ImportError:
                render_context = None

        if render_context is not None:
            return render_context.read_port(self)

        if not self.connected_to:
            if num_samples is not None and num_samples > 0:
                return np.zeros(num_samples, dtype=np.float32)
            return 0.0

        # Collect all values
        values = [p.value for p in self.connected_to]

        # Check if any value is a numpy array
        has_arrays = any(isinstance(v, np.ndarray) for v in values)

        if not has_arrays:
            # All scalars - simple sum
            return sum(values)

        # Mixed or all arrays - need to handle carefully
        # Legacy behavior: when no sample count is requested, mismatched array shapes
        # are considered an error instead of being silently padded or truncated.
        if num_samples is None:
            array_shapes = {v.shape for v in values if isinstance(v, np.ndarray)}
            if len(array_shapes) > 1:
                raise ValueError("Cannot mix arrays with different shapes")

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
                            padded[: len(value)] = value
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
                    assert result is not None
                    result_array = np.asarray(result)

                    # Adjust array size to match result if needed
                    if result_array.shape != value.shape:
                        result_length = len(result_array)
                        if len(value) < result_length:
                            # Pad with zeros
                            padded = np.zeros(result_length, dtype=value.dtype)
                            padded[: len(value)] = value
                            result = result_array + padded
                        else:
                            # Truncate to match result size
                            result = result_array + value[:result_length]
                    else:
                        result = result_array + value
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
            self._latest_buffer = value.copy()
            self._append_tap_history(value)
        elif isinstance(value, (float, int)):
            self.value = float(value)
            self._latest_buffer = self.value
            self._append_tap_history(np.array([self.value], dtype=np.float32))
        elif isinstance(value, Sequence):
            self.value = np.array(value, dtype=np.float32)
            self._latest_buffer = self.value.copy()
            self._append_tap_history(self.value)
        else:
            raise TypeError(
                f"Port write value must be float or np.ndarray, "
                f"got {type(value).__name__}"
            )
        # Visualizers read passive tap history; ports do not notify or pull DSP.

    def peek_recent(self, num_samples: int | None = None) -> float | np.ndarray:
        """Peek at recent samples without triggering upstream generation.

        This is useful for visualization modules that want to read samples
        without affecting the audio processing pipeline.

        Args:
            num_samples: Optional hint for desired sample count (not enforced)

        Returns:
            Current cached value from the port
        """
        value = self._latest_buffer
        if self._tap_history is not None and self._tap_history.size > 0:
            if num_samples is not None and num_samples > 0:
                return self._tap_history[-num_samples:].copy()
            return self._tap_history.copy()
        if isinstance(value, np.ndarray):
            if num_samples is not None and num_samples > 0:
                return value[-num_samples:].copy()
            return value.copy()
        return value

    def _append_tap_history(self, value: np.ndarray) -> None:
        """Append rendered samples to the passive tap history."""
        samples = np.asarray(value)
        if samples.size == 0:
            return
        if samples.ndim == 0:
            samples = samples.reshape(1)

        if self._tap_history is None:
            self._tap_history = samples.copy()
            return

        if self._tap_history.ndim != samples.ndim:
            self._tap_history = samples.copy()
            return

        if samples.ndim > 1 and self._tap_history.shape[1:] != samples.shape[1:]:
            self._tap_history = samples.copy()
            return

        self._tap_history = np.concatenate((self._tap_history, samples), axis=0)
        if len(self._tap_history) > self._tap_history_limit:
            self._tap_history = self._tap_history[-self._tap_history_limit :].copy()

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
