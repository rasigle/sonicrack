from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.gui.audio_module_interface import ModuleCategory
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


class OutputModule(ModuleWidget):
    """Output module (sink for audio)."""

    def __init__(self):
        """Initialize output module."""
        super().__init__(
            width=140,
            height=120,
            color=QColor(200, 80, 80),
        )

        # Add input port
        self.in_port = self.add_input_port("In")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Master volume
        self.volume_knob = Knob("Master", 0.0, 1.0, 0.7)
        self.volume_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "master_volume", self.volume_knob.get_value()
            )
        )
        layout.addWidget(self.volume_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("master_volume", self.volume_knob)

        self.input_component = None

    # AudioModuleInterface implementation
    @property
    def module_title(self) -> str:
        """Return the module title."""
        return "Output"

    @property
    def module_category(self) -> ModuleCategory:
        """Return OUTPUT since this is the terminal node."""
        return ModuleCategory.OUTPUT

    def get_required_inputs(self) -> list[str]:
        """Output requires the In port to be connected."""
        return ["In"]

    def create_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Output doesn't create a component, it returns the input component."""
        if input_components and len(input_components) > 0:
            return input_components[0]
        return self.input_component

    def get_master_volume(self) -> float:
        """Get the master volume level."""
        return self.volume_knob.get_value()
