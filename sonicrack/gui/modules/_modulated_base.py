"""Base class for modulated components (Volume, Panner, etc.)

This provides common functionality for modules that have:
- A main parameter knob
- A modulation input port
- Knob that disables when modulation is connected
- Automatic CV range specification for proper signal scaling
"""

import logging
import threading
from typing import Any

import numpy as np

from sonicrack.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)


class PortModulatorAdapter:
    """Adapter that makes a port act like a generator/modulator for engine components.

    This allows engine components like ModulatedVolume, ModulatedPanner to read
    from GUI ports as if they were generator components.

    Automatically handles CV range conversion when source and destination ranges differ.
    Supports modulation amount/depth control for blending between base value and
    modulation.
    """

    def __init__(
        self,
        port,
        num_samples: int,
        expected_range: tuple[float, float] = (0.0, 1.0),
        modulation_amount: float = 1.0,
    ):
        """Initialize the adapter.

        Args:
            port: The port to read from (can be Port or PortWidget)
            num_samples: Number of samples to read per iteration
            expected_range: Expected CV range for the destination (default: [0, 1])
            modulation_amount: How much modulation to apply (0.0 = none, 1.0 = full)
        """
        # Extract actual Port if we received a PortWidget
        from sonicrack.gui.core.port import Port
        from sonicrack.gui.widgets.port_widget import PortWidget

        if isinstance(port, PortWidget):
            self.port: Port = port.port
        else:
            self.port: Port = port

        self.num_samples = num_samples
        self.expected_range = expected_range
        self.modulation_amount = modulation_amount
        self._index = 0
        self._buffer = None

        # Check if we need CV scaling
        self._needs_scaling = False
        self._scale = 1.0
        self._offset = 0.0

        # Get source output range if available
        source_module = self._get_source_module()
        if source_module and hasattr(source_module, "get_cv_output_range"):
            source_range = source_module.get_cv_output_range()
            if source_range != expected_range:
                self._needs_scaling = True
                # Calculate scaling parameters: output = input * scale + offset
                in_min, in_max = source_range
                out_min, out_max = expected_range
                in_range = in_max - in_min
                out_range = out_max - out_min
                self._scale = out_range / in_range
                self._offset = out_min - (in_min * self._scale)
                logger.debug(
                    f"PortModulatorAdapter: CV scaling enabled "
                    f"{source_range} → {expected_range}, "
                    f"scale={self._scale:.3f}, offset={self._offset:.3f}"
                )

    def _get_source_module(self):
        """Get the source module connected to this port."""
        if not self.port:
            return None

        # Try cables first (for GUI PortWidget or mocks)
        if hasattr(self.port, "cables") and self.port.cables:
            for cable in self.port.cables:
                # The port is an input, so we want the start_port's parent (source)
                if (
                    hasattr(cable, "start_port")
                    and cable.start_port
                    and hasattr(cable.start_port, "parent_module")
                ):
                    return cable.start_port.parent_module

        # Try connected_to (for direct Port API)
        if hasattr(self.port, "connected_to") and self.port.connected_to:
            # Get the first connected port
            for connected_port in self.port.connected_to:
                # Return the parent module of the connected port
                if hasattr(connected_port, "parent_module"):
                    return connected_port.parent_module

        return None

    def __iter__(self):
        """Reset iterator."""
        self._index = 0
        return self

    def __next__(self) -> float:
        """Get next sample from port buffer.

        Note: This is for scalar iteration. Vectorized mode should use
        get_samples() instead for performance.
        """
        if self._buffer is None or self._index >= len(self._buffer):
            # Read new buffer from port
            from sonicrack.gui.core.runtime_helpers import read_samples

            self._buffer = read_samples(self.port, self.num_samples)
            self._index = 0

        value = float(self._buffer[self._index])
        self._index += 1

        # Apply CV scaling if needed
        if self._needs_scaling:
            value = value * self._scale + self._offset
            # Clamp to expected range
            value = max(self.expected_range[0], min(self.expected_range[1], value))

        # Apply modulation amount (depth control)
        if self.modulation_amount < 1.0:
            range_center = (self.expected_range[0] + self.expected_range[1]) / 2
            value = range_center + (value - range_center) * self.modulation_amount

        return value

    def get_samples(self, n: int, mode: str = "vectorized") -> np.ndarray:
        """Get samples from port (engine API).

        This matches the engine's component API where get_samples() is the
        standard method for getting audio buffers.

        Args:
            n: Number of samples to get
            mode: Sampling mode (ignored, always uses vectorized)

        Returns:
            Array of samples scaled by modulation amount
        """
        from sonicrack.gui.core.runtime_helpers import read_samples

        samples = read_samples(self.port, n)

        # Apply CV scaling if needed
        if self._needs_scaling:
            samples = samples * self._scale + self._offset
            # Clamp to expected range
            samples = np.clip(samples, self.expected_range[0], self.expected_range[1])

        # Apply modulation amount (depth control)
        # Scale modulation around the center of the expected range
        if self.modulation_amount < 1.0:
            range_center = (self.expected_range[0] + self.expected_range[1]) / 2
            samples = range_center + (samples - range_center) * self.modulation_amount

        return samples

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Get vectorized samples from port (compatibility method).

        Args:
            n: Number of samples to get

        Returns:
            Array of samples
        """
        return self.get_samples(n, mode="vectorized")


class ModulatedModuleBase(ModuleWidget):
    """Base class for modules with modulation input.

    Provides:
    - update_knob_state() - Updates knob enabled state based on connections
    - get_modulation_inputs() - Returns ["Mod"]
    - get_cv_range() - Returns expected CV range for modulation input
    - Common modulation handling logic

    Subclasses must:
    - Set self.control_knob to the main parameter knob
    - Implement create_modulated_component(mod_comp)
    - Implement create_unmodulated_component()
    - Optionally override get_cv_range() to specify expected CV range
    """

    def __init__(self, *args, **kwargs):
        """Initialize modulated module base."""
        super().__init__(*args, **kwargs)
        self.control_knob = None  # Subclass must set this
        self.modulator_component = None
        self.port_adapter = (
            None  # Store adapter reference for updating modulation_amount
        )
        self._is_modulated = False  # Track current modulation state
        self._component_lock = threading.RLock()  # Thread-safe component switching

    def on_port_connection_changed(self, port_name: str, is_connected: bool):
        """Handle port connection/disconnection events.

        This automatically switches between modulated and unmodulated components
        when the Mod port connection state changes. Components are prepared here
        to keep process_runtime() fast and simple.

        Args:
            port_name: Name of the port that changed
            is_connected: True if connected, False if disconnected
        """
        if port_name != "Mod":
            return

        # Update component based on connection state - THREAD SAFE
        with self._component_lock:
            if is_connected and not self._is_modulated:
                # Switch to modulated component - prepare it now
                logger.debug(
                    f"{self.__class__.__name__}: Switching to modulated component"
                )
                self._is_modulated = True
                # Prepare modulated component with port adapter
                # Use a default buffer size, it will adapt at runtime
                self.prepare_modulated_component(num_samples=512)
            elif not is_connected and self._is_modulated:
                # Switch to unmodulated component - create it now
                logger.debug(
                    f"{self.__class__.__name__}: Switching to unmodulated component"
                )
                self._is_modulated = False
                # Pre-create the unmodulated component
                self.component = self.create_unmodulated_component()

        # Update knob state
        self.update_knob_state()

    def prepare_modulated_component(self, num_samples: int):
        """Prepare the modulated component with port adapter.

        This creates the modulated component using a PortModulatorAdapter
        to bridge port-based signal flow with component-based processing.

        Args:
            num_samples: Buffer size for the adapter
        """
        # Find the Mod port
        mod_port = None
        for port in self.input_ports:
            if port.port_name == "Mod":
                mod_port = port
                break

        if mod_port is None:
            logger.warning(f"{self.__class__.__name__}: Mod port not found!")
            return

        # Get expected CV range for this module
        expected_range = self.get_cv_range("Mod")

        # Create port adapter that acts like a generator/modulator with CV scaling
        self.port_adapter = PortModulatorAdapter(
            mod_port, num_samples, expected_range, modulation_amount=1.0
        )

        # Use the abstract method to create the proper modulated component
        self.component = self.create_modulated_component(self.port_adapter)
        logger.debug(
            f"{self.__class__.__name__}: Created modulated component with port adapter "
            f"(CV range: {expected_range})"
        )

    def update_knob_state(self):
        """Update knob tooltip based on port connections.

        When modulation is connected, the knob controls modulation depth/amount.
        When no modulation, the knob controls the base parameter value.
        Knob is always enabled to provide control in both modes.
        """
        if not self.control_knob:
            logger.warning(f"{self.__class__.__name__}: control_knob not set!")
            return

        # Check if Mod port has any connections
        mod_port = None
        for port in self.input_ports:
            if port.port_name == "Mod":
                mod_port = port
                break

        # Check both GUI cables and direct port connections
        has_modulation = False
        if mod_port:
            # Check GUI cables (PortWidget)
            cables_connected = (
                hasattr(mod_port, "cables")
                and mod_port.cables
                and len(mod_port.cables) > 0
            )
            # Check direct port connections (Port.connected_to)
            port_connected = (
                hasattr(mod_port, "port")
                and hasattr(mod_port.port, "connected_to")
                and mod_port.port.connected_to
                and len(mod_port.port.connected_to) > 0
            )
            # If mod_port is a Port directly (not PortWidget)
            direct_connected = False
            if hasattr(mod_port, "connected_to"):
                connected_to = getattr(mod_port, "connected_to", None)
                direct_connected = (
                    connected_to is not None
                    and hasattr(connected_to, "__len__")
                    and len(connected_to) > 0
                )

            has_modulation = cables_connected or port_connected or direct_connected

        if has_modulation:
            # Modulation connected - knob controls modulation depth/amount
            self.control_knob.setEnabled(True)
            self.control_knob.setStyleSheet("")  # Normal appearance
            self.control_knob.setToolTip(
                f"{self.control_knob.label} - Modulation Depth\n"
                f"0.0 = No modulation (fixed value)\n"
                f"1.0 = Full modulation"
            )
        else:
            # No modulation - knob controls base parameter value
            self.control_knob.setEnabled(True)
            self.control_knob.setStyleSheet("")
            self.control_knob.setToolTip(
                f"Manual {self.control_knob.label.lower()} control"
            )

    def get_modulation_inputs(self) -> list[str]:
        """Return modulation input ports.

        All modulated modules have a "Mod" port.
        """
        return ["Mod"]

    def get_cv_range(self, port_name: str = "Mod") -> tuple[float, float]:
        """Get expected CV range for a modulation input.

        This is used by the patch compiler to automatically insert CV scalers
        when connecting sources with different output ranges.

        Args:
            port_name: Name of the modulation port

        Returns:
            Tuple of (min, max) expected CV values

        Default implementation returns (0.0, 1.0) for unipolar modulation.
        Override in subclasses for different ranges (e.g., Panner uses (-1, 1)).

        Examples:
            - Volume/Clipper: (0.0, 1.0) - unipolar
            - Panner: (-1.0, 1.0) - bipolar
        """
        _ = port_name
        return 0.0, 1.0  # Default: unipolar [0, 1]

    @staticmethod
    def _describe_component(component: Any) -> str:
        """Return a stable component description for logging."""
        if component is None:
            return "None"

        descriptor = getattr(type(component), "descriptor", None)
        descriptor_name = getattr(descriptor, "name", None)
        if isinstance(descriptor_name, str) and descriptor_name:
            return f"{type(component).__name__}({descriptor_name})"

        return type(component).__name__

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the engine component with or without modulation.

        Subclasses should NOT override this. Instead, implement:
        - create_modulated_component(mod_comp)
        - create_unmodulated_component()
        """
        # Check if modulation is provided
        mod_comp = None
        if modulation_components and "Mod" in modulation_components:
            mod_comp = modulation_components["Mod"]
        elif self.modulator_component:
            mod_comp = self.modulator_component

        logger.debug(
            "%s modulator: %s", type(self).__name__, self._describe_component(mod_comp)
        )

        if mod_comp:
            # Modulation connected - update UI and create modulated component
            if self.control_knob:
                self.control_knob.setEnabled(False)
                self.control_knob.setStyleSheet("opacity: 0.5;")
                self.control_knob.setToolTip(
                    f"{self.control_knob.label} controlled by Mod input (CV)"
                )

            return self.create_modulated_component(mod_comp)

        # No modulation - update UI and create simple component
        if self.control_knob:
            self.control_knob.setEnabled(True)
            self.control_knob.setStyleSheet("")
            self.control_knob.setToolTip(
                f"Manual {self.control_knob.label.lower()} control"
            )

        return self.create_unmodulated_component()

    def create_modulated_component(self, mod_comp):
        """Create the modulated version of the component.

        Args:
            mod_comp: The modulation component

        Returns:
            Modulated component instance

        Must be implemented by subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement create_modulated_component()"
        )

    def create_unmodulated_component(self):
        """Create the simple (non-modulated) version of the component.

        Returns:
            Simple component instance

        Must be implemented by subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement create_unmodulated_component()"
        )

    def safe_process_component(self, input_samples: np.ndarray) -> np.ndarray:
        """Thread-safe wrapper for processing samples through the component.

        This method should be used in process_runtime() to ensure thread-safe
        access to self.component while the GUI thread might be switching it.

        Args:
            input_samples: Input samples to process

        Returns:
            Processed samples
        """
        with self._component_lock:
            if self.component is None:
                # Component not initialized yet
                return np.zeros_like(input_samples, dtype=np.float32)
            return self.component(input_samples)
