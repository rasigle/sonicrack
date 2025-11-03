from typing import Dict, Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGraphicsProxyWidget

from engine import WaveAdder
from gui.widgets.module_widget import ModuleWidget

TITLE = "Mixer"

class MixerModule(ModuleWidget):
    """Mixer module for combining multiple audio signals.

    Uses WaveAdder to mix multiple inputs together (averages them).
    """

    def __init__(self):
        """Initialize mixer module."""
        super().__init__(TITLE, category="mixer", width=160, height=200, color=QColor(100, 150, 100))

        # Add multiple input ports
        self.in1_port = self.add_input_port("In 1")
        self.in2_port = self.add_input_port("In 2")
        self.in3_port = self.add_input_port("In 3")
        self.in4_port = self.add_input_port("In 4")

        # Add output port
        self.out_port = self.add_output_port("Out")

        # Create control widget
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Info label
        info_label = QLabel("Mixes multiple\ninputs together")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("font-size: 9px; color: #ccc;")
        layout.addWidget(info_label)

        self.controls_widget.setLayout(layout)

        # Add controls as proxy widget
        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(self.controls_widget)
        self.proxy.setPos(0, 42)

        self.input_components = []  # Will be populated by patch compiler

        self.component = self.create_component()

    def create_component(self):
        """Create the mixer component.

        Note: This will be created by the patch compiler which has access
        to the actual connected input components.
        """
        if self.input_components and len(self.input_components) > 0:
            return WaveAdder(*self.input_components)
        return None

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {}

    def set_parameters(self, params: Dict[str, Any]):
        """Set parameters from dictionary."""
        pass
