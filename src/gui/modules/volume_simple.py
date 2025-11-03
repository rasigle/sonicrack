from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import Volume
from src.gui.audio_module_interface import ModuleCategory
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


class SimpleVolumeModule(ModuleWidget):
    """Simple volume/gain module without modulation input."""

    def __init__(self):
        """Initialize simple volume module."""
        super().__init__(
            width=140,
            height=150,
            color=QColor(160, 100, 60),
        )

        # Add ports (no modulation input)
        self.in_port = self.add_input_port("In")
        self.out_port = self.add_output_port("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Volume knob
        self.volume_knob = Knob("Gain", 0.0, 2.0, 1.0)
        self.volume_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("volume", self.volume_knob.get_value())
        )
        layout.addWidget(self.volume_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("volume", self.volume_knob)

        self.component = self.create_component()

    # AudioModuleInterface implementation
    @property
    def module_title(self) -> str:
        """Return the module title."""
        return "Volume"

    @property
    def module_category(self) -> ModuleCategory:
        """Return MODIFIER since this modifies audio input."""
        return ModuleCategory.MODIFIER

    def get_required_inputs(self) -> list[str]:
        """Volume requires the In port to be connected."""
        return ["In"]

    def create_component(
        self,
        input_components: Optional[list[Any]] = None,
        modulation_components: Optional[dict[str, Any]] = None,
    ):
        """Create the volume component."""
        volume = self.volume_knob.get_value()
        return Volume(volume)
