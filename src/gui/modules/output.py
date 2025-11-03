from typing import Dict, Any, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGraphicsProxyWidget

from src.gui.widgets.module_widget import ModuleWidget
from src.gui.widgets import Knob
from src.gui.audio_module_interface import ModuleCategory

TITLE = "Output"


class OutputModule(ModuleWidget):
    """Output module (sink for audio)."""

    def __init__(self):
        """Initialize output module."""
        super().__init__(
            "Output",
            width=140,
            height=120,
            color=QColor(200, 80, 80),
        )

        # Add input port
        self.in_port = self.add_input_port("In")

        # Create control widget
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Master volume
        self.volume_knob = Knob("Master", 0.0, 1.0, 0.7)
        self.volume_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "master_volume", self.volume_knob.get_value()
            )
        )
        layout.addWidget(self.volume_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)

        # Add controls as proxy widget
        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(self.controls_widget)
        self.proxy.setPos(0, 42)

        self.input_component = None

    # AudioModuleInterface implementation
    @property
    def module_category(self) -> ModuleCategory:
        """Return OUTPUT since this is the terminal node."""
        return ModuleCategory.OUTPUT

    def get_required_inputs(self) -> list[str]:
        """Output requires the In port to be connected."""
        return ["In"]

    def create_component(
        self,
        input_components: Optional[list[Any]] = None,
        modulation_components: Optional[dict[str, Any]] = None,
    ):
        """Output doesn't create a component, it returns the input component."""
        if input_components and len(input_components) > 0:
            return input_components[0]
        return self.input_component

    def get_master_volume(self) -> float:
        """Get the master volume level."""
        return self.volume_knob.get_value()

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"master_volume": self.volume_knob.get_value()}

    def set_parameters(self, params: dict[str, Any]):
        """Set parameters from dictionary."""
        if "master_volume" in params:
            self.volume_knob.set_value(params["master_volume"])
