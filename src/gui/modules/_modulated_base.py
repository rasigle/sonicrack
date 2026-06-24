"""Base class for modulated components (Volume, Panner, etc.)

This provides common functionality for modules that have:
- A main parameter knob
- A modulation input port
- Knob that disables when modulation is connected
- Automatic CV range specification for proper signal scaling
"""

import logging
from typing import Any

from src.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)


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

    def update_knob_state(self):
        """Update knob enabled state based on port connections.

        This is called when connections change to provide immediate visual feedback,
        even if compilation fails.
        """
        if not self.control_knob:
            logger.warning(f"{self.__class__.__name__}: control_knob not set!")
            return

        # Check if Mod port has any cables connected
        mod_port = None
        for port in self.input_ports:
            if port.port_name == "Mod":
                mod_port = port
                break

        has_modulation = mod_port and len(mod_port.cables) > 0

        if has_modulation:
            # Modulation connected - disable knob
            self.control_knob.setEnabled(False)
            self.control_knob.setStyleSheet("opacity: 0.5;")
            self.control_knob.setToolTip(
                f"{self.control_knob.label} controlled by Mod input (CV)"
            )
        else:
            # No modulation - enable knob
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

    def process(self, num_samples: int = 1):
        raise NotImplementedError(f"{type(self).__name__} must implement process()")
