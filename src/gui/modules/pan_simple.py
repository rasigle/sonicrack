from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import Panner
from src.gui.audio_module_interface import ModuleCategory
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


class SimplePannerModule(ModuleWidget):
    """Simple panner module without modulation input."""

    def __init__(self):
        """Initialize simple panner module."""
        super().__init__(
            width=140,
            height=150,
            color=QColor(160, 60, 160),
        )

        # Add ports (no modulation input)
        self.in_port = self.add_input_port("In")
        self.out_port = self.add_output_port("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Pan knob
        self.pan_knob = Knob("Pan", -1.0, 1.0, 0.0)
        self.pan_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("pan", self.pan_knob.get_value())
        )
        layout.addWidget(self.pan_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("pan", self.pan_knob)

        self.component = self.create_component()

    # AudioModuleInterface implementation
    @property
    def module_title(self) -> str:
        """Return the module title."""
        return "Panner"

    @property
    def module_category(self) -> ModuleCategory:
        """Return MODIFIER since this modifies audio input."""
        return ModuleCategory.MODIFIER

    def get_required_inputs(self) -> list[str]:
        """Panner requires the In port to be connected."""
        return ["In"]

    def create_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the panner component."""
        pan = self.pan_knob.get_value()
        return Panner(pan)
