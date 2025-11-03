from typing import Dict, Any, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGraphicsProxyWidget

from src.engine import Clipper
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.widgets import Knob
from src.gui.audio_module_interface import ModuleCategory

TITLE = "Clipper"


class ClipperModule(ModuleWidget):
    """Clipper module for distortion/limiting."""

    def __init__(self):
        """Initialize clipper module."""
        super().__init__(
            title=TITLE,
            width=140,
            height=180,
            color=QColor(200, 150, 80),
        )

        # Add ports
        self.in_port = self.add_input_port("In")
        self.out_port = self.add_output_port("Out")

        # Create control widget
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Threshold knob
        self.threshold_knob = Knob("Threshold", 0.1, 1.0, 1.0)
        self.threshold_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "threshold", self.threshold_knob.get_value()
            )
        )
        layout.addWidget(self.threshold_knob, alignment=Qt.AlignmentFlag.AlignCenter)

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
        """Clipper requires the In port to be connected."""
        return ["In"]

    def create_component(
        self,
        input_components: Optional[list[Any]] = None,
        modulation_components: Optional[dict[str, Any]] = None,
    ):
        """Create the clipper component."""
        threshold = self.threshold_knob.get_value()
        return Clipper((-threshold, threshold))

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {"threshold": self.threshold_knob.get_value()}

    def set_parameters(self, params: dict[str, Any]):
        """Set parameters from dictionary."""
        if "threshold" in params:
            self.threshold_knob.set_value(params["threshold"])
