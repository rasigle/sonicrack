import logging

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from soniclab import ModulatedPanner, Panner

from sonicrack.gui.modules._modulated_base import ModulatedModuleBase
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)


@register_module()
class PannerModule(ModulatedModuleBase):
    """Panner module for stereo positioning with modulation support."""

    runtime_kind = "panner"

    metadata = ModuleMetadata(
        title="Panner (Mod)",
        category=ModuleCategory.MODIFIER,
        description="Stereo panner with modulation input",
    )

    def __init__(self):
        """Initialize panner module."""
        super().__init__(
            width=140,
            height=180,
            color=QColor(180, 80, 180),
        )

        # Add ports
        self.in_port = self.add_input("In")
        self.mod_port = self.add_input("Mod")
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Pan knob
        self.pan_knob = Knob(
            label="Pan", min_value=-1.0, max_value=1.0, default_value=0.0
        )
        self.pan_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("position", self.pan_knob.get_value())
        )
        layout.addWidget(self.pan_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("position", self.pan_knob)

        # Set control_knob for base class functionality
        self.control_knob = self.pan_knob

        # Create initial unmodulated component
        self.component = self.create_unmodulated_component()

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Panner requires the In port to be connected."""
        return ["In"]

    def get_cv_range(self, port_name: str = "Mod") -> tuple[float, float]:
        """Panner expects bipolar CV range [-1, 1] for pan position.

        Returns:
            (-1.0, 1.0) - bipolar range for pan control
        """
        return -1.0, 1.0

    # Implement abstract methods from ModulatedModuleBase
    def create_modulated_component(self, mod_comp):
        """Create ModulatedPanner with modulation."""
        return ModulatedPanner(mod_comp)

    def create_unmodulated_component(self):
        """Create simple Panner without modulation."""
        position = self.pan_knob.get_value()
        return Panner(position)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Pan the connected input for one render cycle.

        The component (Panner or ModulatedPanner) is prepared by the connection
        handler, so we just call it directly without branching.
        """
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        samples = read_samples(self.in_port, num_samples)

        # Thread-safe component access with lock
        with self._component_lock:
            # Runtime safety net if a disconnect missed UI notifications.
            self.ensure_modulation_component_state(num_samples)

            # Get position from knob
            position = float_parameter(parameters, "position", self.pan_knob.get_value)

            if self.mod_port.is_connected:
                # When modulated: knob controls modulation depth (0.0 to 1.0)
                # Map position range [-1, 1] to modulation amount [0.0, 1.0]
                modulation_amount = (position + 1.0) / 2.0  # Maps [-1, 1] to [0, 1]
                if self.port_adapter is not None:
                    self.port_adapter.modulation_amount = modulation_amount
            else:
                # When unmodulated: knob controls position directly
                if self.component is not None:
                    self.component.position = position

            # Call component directly - works for both Panner and ModulatedPanner
            if self.component is not None:
                left, right = self.component(samples)
                self.out_port.write(np.column_stack((left, right)))
            else:
                self.out_port.write(silence(num_samples))
