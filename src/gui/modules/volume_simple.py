from typing import Dict, Any, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGraphicsProxyWidget

from src.engine import Volume
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.widgets import Knob
from src.gui.audio_module_interface import ModuleCategory

TITLE = "Volume"


class SimpleVolumeModule(ModuleWidget):
    """Simple volume/gain module without modulation input."""

    def __init__(self):
        """Initialize simple volume module."""
        super().__init__(
            TITLE,
            width=140,
            height=150,
            color=QColor(160, 100, 60),
        )

        # Add ports (no modulation input)
        self.in_port = self.add_input_port("In")
        self.out_port = self.add_output_port("Out")

        # Create control widget
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Volume knob
        self.volume_knob = Knob("Gain", 0.0, 2.0, 1.0)
        self.volume_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("volume", self.volume_knob.get_value())
        )
        layout.addWidget(self.volume_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)

        # Add controls as proxy widget
        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(self.controls_widget)
        self.proxy.setPos(0, 42)

        self.component = self.create_component()

    # AudioModuleInterface implementation
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

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"volume": self.volume_knob.get_value()}

    def set_parameters(self, params: dict[str, Any]):
        """Set parameters from dictionary."""
        if "volume" in params:
            self.volume_knob.set_value(params["volume"])
