from typing import Dict, Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGraphicsProxyWidget

from engine import ModulatedVolume, Volume
from gui.widgets.module_widget import ModuleWidget
from gui.widgets import Knob

TITLE = "Volume"

class VolumeModule(ModuleWidget):
    """Volume/Gain module."""

    def __init__(self):
        """Initialize volume module."""
        super().__init__(TITLE, category= "modifier", width=140, height=180, color=QColor(180, 120, 80))

        # Add ports
        self.in_port = self.add_input_port("In")
        self.mod_port = self.add_input_port("Mod")
        self.out_port = self.add_output_port("Out")

        # Create control widget
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Volume knob
        self.volume_knob = Knob("Volume", 0.0, 2.0, 1.0)
        self.volume_knob.value_changed.connect(lambda: self.parameter_changed.emit("volume", self.volume_knob.get_value()))
        layout.addWidget(self.volume_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)

        # Add controls as proxy widget
        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(self.controls_widget)
        self.proxy.setPos(0, 42)

        self.input_component = None
        self.modulator_component = None
        self.component = self.create_component()

    def create_component(self):
        """Create the volume component."""
        volume = self.volume_knob.get_value()

        if self.modulator_component:
            return ModulatedVolume(self.modulator_component)
        return Volume(volume)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {"volume": self.volume_knob.get_value()}

    def set_parameters(self, params: Dict[str, Any]):
        """Set parameters from dictionary."""
        if "volume" in params:
            self.volume_knob.set_value(params["volume"])
