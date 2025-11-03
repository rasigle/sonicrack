from typing import Dict, Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGraphicsProxyWidget

from engine import ModulatedPanner, Panner
from gui.widgets.module_widget import ModuleWidget
from gui.widgets import Knob


TITLE = "Panner (Mod)"

class PannerModule(ModuleWidget):
    """Panner module for stereo positioning."""

    def __init__(self):
        """Initialize panner module."""
        super().__init__(TITLE, category="modifier", width=140, height=180, color=QColor(180, 80, 180))

        # Add ports
        self.in_port = self.add_input_port("In")
        self.mod_port = self.add_input_port("Mod")
        self.out_port = self.add_output_port("Out")

        # Create control widget
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Pan knob
        self.pan_knob = Knob("Pan", -1.0, 1.0, 0.0)
        self.pan_knob.value_changed.connect(lambda: self.parameter_changed.emit("pan", self.pan_knob.get_value()))
        layout.addWidget(self.pan_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)

        # Add controls as proxy widget
        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(self.controls_widget)
        self.proxy.setPos(0, 42)

        self.modulator_component = None

        self.component = self.create_component()

    def create_component(self):
        """Create the panner component."""
        pan = self.pan_knob.get_value()

        if self.modulator_component:
            return ModulatedPanner(self.modulator_component)

        return Panner(pan)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {"pan": self.pan_knob.get_value()}

    def set_parameters(self, params: Dict[str, Any]):
        """Set parameters from dictionary."""
        if "pan" in params:
            self.pan_knob.set_value(params["pan"])
