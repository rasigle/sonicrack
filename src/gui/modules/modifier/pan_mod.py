import logging

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import ModulatedPanner, Panner
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, read_samples, silence
from src.gui.module_registry import register_module
from src.gui.modules._modulated_base import ModulatedModuleBase
from src.gui.widgets import Knob

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
        self.pan_knob = Knob("Pan", -1.0, 1.0, 0.0)
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
        """Pan the connected input for one render cycle."""
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        samples = read_samples(self.in_port, num_samples)
        position = float_parameter(parameters, "position", self.pan_knob.get_value)

        if samples.ndim == 1:
            left_gain = np.sqrt(0.5 * (1.0 - position))
            right_gain = np.sqrt(0.5 * (1.0 + position))
            self.out_port.write(
                np.column_stack((samples * left_gain, samples * right_gain))
            )
            return

        panned = samples.copy()
        panned[:, 0] *= np.sqrt(0.5 * (1.0 - position))
        panned[:, 1] *= np.sqrt(0.5 * (1.0 + position))
        self.out_port.write(panned)
