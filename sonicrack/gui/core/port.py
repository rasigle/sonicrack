"""Core signal-port data model with no Qt dependencies.

The UI layer wraps this class in ``sonicrack.gui.widgets.port_widget.PortWidget``.
Ports store current values, connection state, and passive tap history used by
visualization modules.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from enum import StrEnum

import numpy as np

from sonicrack.utils.common_utils import enum_from_value

logger = logging.getLogger(__name__)


class PortType(StrEnum):
    """Enum describing port type."""

    INPUT = "input"
    OUTPUT = "output"


class PortSignal(StrEnum):
    """Signal/unit contract for patch-cable compatibility checks."""

    UNKNOWN = "unknown"
    AUDIO = "audio"
    FREQUENCY_HZ = "frequency_hz"
    PITCH_CV = "pitch_cv"
    GATE = "gate"
    CONTROL_CV = "control_cv"


_ANALOG_PATCH_SIGNALS = frozenset(
    {
        PortSignal.AUDIO,
        PortSignal.PITCH_CV,
        PortSignal.GATE,
        PortSignal.CONTROL_CV,
    }
)


def normalize_port_signal(signal: str | PortSignal | None) -> PortSignal:
    """Normalize a signal-kind value to a PortSignal enum."""
    if signal is None:
        return PortSignal.UNKNOWN

    if isinstance(signal, PortSignal):
        return signal

    try:
        return PortSignal(signal)
    except ValueError:
        return PortSignal.UNKNOWN


def _orient_connection(source: Port, target: Port) -> tuple[Port, Port]:
    """Return ports ordered as output/source then input/target when possible."""
    if source.port_type == PortType.OUTPUT and target.port_type == PortType.INPUT:
        return source, target
    if source.port_type == PortType.INPUT and target.port_type == PortType.OUTPUT:
        return target, source
    return source, target


def _signals_are_patch_compatible(
    source_signal: PortSignal,
    target_signal: PortSignal,
) -> bool:
    """Return whether a source signal can safely drive a target signal."""
    if source_signal == target_signal:
        return True
    if source_signal == PortSignal.UNKNOWN or target_signal == PortSignal.UNKNOWN:
        return True
    if (
        source_signal == PortSignal.FREQUENCY_HZ
        or target_signal == PortSignal.FREQUENCY_HZ
    ):
        return False
    return (
        source_signal in _ANALOG_PATCH_SIGNALS
        and target_signal in _ANALOG_PATCH_SIGNALS
    )


def port_signals_compatible(source: Port, target: Port) -> bool:
    """Return whether two ports can be safely connected by signal kind."""
    oriented_source, oriented_target = _orient_connection(source, target)
    return _signals_are_patch_compatible(
        oriented_source.signal,
        oriented_target.signal,
    )


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
        port_type: str | PortType,  # "input" or "output"
        port_name: str,
        index: int = 0,
        parent_module=None,  # Reference to the module that owns this port
        component=None,  # Reference to the engine component (e.g., SineOscillator)
        signal: str | PortSignal | None = None,
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
        self.port_type: PortType = enum_from_value(PortType, port_type)
        self.port_name: str = port_name
        self.parent_module = parent_module
        self.component = component  # Direct reference to engine component
        self.signal = normalize_port_signal(signal)

        self.connected_to: list[Port] = []

        self.index: int = index

        # Data state - can be scalar or numpy array
        self.value: float | np.ndarray = 0.0
        self._latest_buffer: float | np.ndarray = 0.0
        self._tap_history: np.ndarray | None = None
        self._tap_history_limit = 65536
        self._tap_write_index = 0
        self._tap_count = 0
        self._tap_history_enabled = False

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
        if not port_signals_compatible(self, other):
            raise ValueError(
                "Cannot connect incompatible port signals: "
                f"{self.port_name} ({self.signal}) to "
                f"{other.port_name} ({other.signal})"
            )
        if other in self.connected_to:
            return  # Already connected, no-op

        logger.debug(f"Port connected: {self.port_name} <-> {other.port_name}")

        # Bidirectional connection: both ports track the connection
        self.connected_to.append(other)
        if self not in other.connected_to:
            other.connected_to.append(self)
        self._sync_tap_history_subscription()
        other._sync_tap_history_subscription()

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
            self._sync_tap_history_subscription()
            return

        try:
            self.connected_to.remove(other)
            # Also remove from other side (bidirectional)
            if self in other.connected_to:
                other.connected_to.remove(self)
                other._sync_tap_history_subscription()
            logger.debug(f"Port disconnected: {self.port_name} <-/-> {other.port_name}")
            # Clear port data to prevent stale audio
            self.clear()
            self._sync_tap_history_subscription()
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
        self._tap_write_index = 0
        self._tap_count = 0
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
            with contextlib.suppress(ImportError):
                from sonicrack.gui.audio_engine import get_active_render_context

                render_context = get_active_render_context()

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

        if num_samples is None:
            first_array = next(v for v in values if isinstance(v, np.ndarray))
            result = np.zeros_like(first_array)
            for value in values:
                result = result + value
            return result

        # If num_samples was requested, ensure all arrays match that size while
        # preserving any channel dimensions after the sample axis.
        first_array = next(v for v in values if isinstance(v, np.ndarray))
        if len(first_array) < num_samples:
            target_shape = (num_samples, *first_array.shape[1:])
        else:
            target_shape = first_array[:num_samples].shape

        result = np.zeros(target_shape, dtype=np.float32)
        for value in values:
            if isinstance(value, np.ndarray):
                if len(value) < num_samples:
                    padded = np.zeros(target_shape, dtype=value.dtype)
                    padded[: len(value)] = value
                    result = result + padded
                else:
                    result = result + value[:num_samples]
            else:
                result = result + value

        return result

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
            self._update_tap_history(value)

        elif isinstance(value, (float, int)):
            self.value = float(value)
            self._latest_buffer = self.value
            if self._tap_history_enabled:
                self._append_tap_history(np.array([self.value], dtype=np.float32))

        elif isinstance(value, Sequence):
            array_value = np.array(value, dtype=np.float32)
            self.value = array_value
            self._update_tap_history(array_value)

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
            ordered = self._ordered_tap_history()
            if num_samples is not None and num_samples > 0:
                return ordered[-num_samples:].copy()
            return ordered.copy()
        if isinstance(value, np.ndarray):
            if num_samples is not None and num_samples > 0:
                return value[-num_samples:].copy()
            return value.copy()
        return value

    def _update_tap_history(self, value: np.ndarray) -> None:
        """Update latest buffer and tap history for array values.

        Args:
            value: Array value to store
        """
        if self._tap_history_enabled:
            self._latest_buffer = value.copy()
            self._append_tap_history(value)
        else:
            self._latest_buffer = value

    def _append_tap_history(self, value: np.ndarray) -> None:
        """Append rendered samples to bounded passive tap history."""
        samples = np.asarray(value)
        if samples.size == 0:
            return
        if samples.ndim == 0:
            samples = samples.reshape(1)
        if len(samples) > self._tap_history_limit:
            samples = samples[-self._tap_history_limit :]

        if self._tap_history is None:
            self._allocate_tap_history(samples)
            self._write_tap_samples(samples)
            return

        if self._tap_history.ndim != samples.ndim:
            self._allocate_tap_history(samples)
            self._write_tap_samples(samples)
            return

        if samples.ndim > 1 and self._tap_history.shape[1:] != samples.shape[1:]:
            self._allocate_tap_history(samples)
            self._write_tap_samples(samples)
            return

        self._write_tap_samples(samples)

    def _allocate_tap_history(self, samples: np.ndarray) -> None:
        """Allocate or reset the fixed-size visualizer tap buffer."""
        shape = (self._tap_history_limit, *samples.shape[1:])
        self._tap_history = np.zeros(shape, dtype=samples.dtype)
        self._tap_write_index = 0
        self._tap_count = 0

    def _write_tap_samples(self, samples: np.ndarray) -> None:
        """Write samples into the circular tap buffer."""
        if self._tap_history is None:
            return

        count = len(samples)
        end = self._tap_write_index + count
        if end <= self._tap_history_limit:
            self._tap_history[self._tap_write_index : end] = samples
        else:
            first_count = self._tap_history_limit - self._tap_write_index
            self._tap_history[self._tap_write_index :] = samples[:first_count]
            self._tap_history[: end % self._tap_history_limit] = samples[first_count:]

        self._tap_write_index = end % self._tap_history_limit
        self._tap_count = min(self._tap_count + count, self._tap_history_limit)

    def _ordered_tap_history(self) -> np.ndarray:
        """Return tap history in chronological order."""
        if self._tap_history is None or self._tap_count == 0:
            return np.empty(0, dtype=np.float32)
        if self._tap_count < self._tap_history_limit:
            return self._tap_history[: self._tap_count]
        return np.concatenate(
            (
                self._tap_history[self._tap_write_index :],
                self._tap_history[: self._tap_write_index],
            ),
            axis=0,
        )

    def _sync_tap_history_subscription(self) -> None:
        """Enable audio tap history only while a visualization input is attached."""
        should_enable = self.port_type == "output" and any(
            _is_visualization_input(connected_port)
            for connected_port in self.connected_to
        )
        if should_enable == self._tap_history_enabled:
            return

        self._tap_history_enabled = should_enable
        self._tap_history = None
        self._tap_write_index = 0
        self._tap_count = 0

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
            f"signal='{self.signal}', value={value_str}, {conn_status})"
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


def _is_visualization_input(port: Port) -> bool:
    """Return True when a port belongs to a visualization sink input."""
    if port.port_type != PortType.INPUT:
        return False

    module = port.parent_module
    metadata = getattr(module, "metadata", None)
    category = getattr(metadata, "category", None)
    return (
        category == "Visualization"
        or getattr(category, "value", None) == "Visualization"
    )
