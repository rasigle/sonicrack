from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import ModulatedPanner, Panner
from src.gui.audio_module_interface import ModuleCategory, ModuleMetadata
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.module_registry import register_module


@register_module()
class PannerModule(ModuleWidget):
    """Panner module for stereo positioning."""

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
        self.in_port = self.add_input_port("In")
        self.mod_port = self.add_input_port("Mod")
        self.out_port = self.add_output_port("Out")

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

        self.component = self.create_component()

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Panner requires the In port to be connected."""
        return ["In"]

    def get_modulation_inputs(self) -> list[str]:
        """Panner can optionally use Mod port for modulation."""
        return ["Mod"]

    def create_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the panner component."""
        pan = self.pan_knob.get_value()

        # Check if modulation is provided
        mod_comp = None
        if modulation_components and "Mod" in modulation_components:
            mod_comp = modulation_components["Mod"]

        if mod_comp:
            return ModulatedPanner(mod_comp)

        return Panner(pan)
