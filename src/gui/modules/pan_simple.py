from typing import Dict, Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGraphicsProxyWidget

from engine import Panner
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.widgets import Knob

TITLE = "Panner"

class SimplePannerModule(ModuleWidget):
    """Simple panner module without modulation input."""

    def __init__(self):
        """Initialize simple panner module."""
        super().__init__(TITLE, category="modifier", width=140, height=150, color=QColor(160, 60, 160))

        # Add ports (no modulation input)
        self.in_port = self.add_input_port("In")
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

        self.component = self.create_component()

    def create_component(self):
        """Create the panner component."""
        pan = self.pan_knob.get_value()
        return Panner(pan)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {"pan": self.pan_knob.get_value()}

    def set_parameters(self, params: Dict[str, Any]):
        """Set parameters from dictionary."""
        if "pan" in params:
            self.pan_knob.set_value(params["pan"])
