import logging

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from src.engine.effects import Distortion
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.modules._modulated_base import ModulatedModuleBase
from src.gui.widgets import Knob

logger = logging.getLogger(__name__)


DEFAULT_DISTORTION_TYPE = "Distortion"


@register_module()
class DistortionModule(ModulatedModuleBase):
    """Distortion module"""

    metadata = ModuleMetadata(
        title="Distortion",
        category=ModuleCategory.MODIFIER,
        description="Apply distortion effect to audio signal",
    )

    def __init__(self):
        """Initialize panner module."""
        super().__init__(
            width=220,
            height=240,
            color=QColor(180, 80, 180),
        )

        # Add ports
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.mod_port = self.add_input("CV_Drive")
        self.mod_port = self.add_input("CV_Mix")
        self.component = None

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        knobs_row = QHBoxLayout()
        knobs_row.setSpacing(10)

        self.drive_knob = Knob("Drive", 0.0, 1.0, 0.5)
        self.drive_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("drive", self.drive_knob.get_value())
        )
        knobs_row.addWidget(self.drive_knob)

        self.mix_knob = Knob("Mix", 0.0, 1.0, 0.5)
        self.mix_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("mix", self.mix_knob.get_value())
        )
        knobs_row.addWidget(self.mix_knob)
        layout.addLayout(knobs_row)

        self.distortion_combo = QtWidgets.QComboBox()
        self.distortion_combo.addItems(["soft", "hard", "fuzz", "tube"])
        self.distortion_combo.currentTextChanged.connect(self._on_type_changed)
        layout.addWidget(QtWidgets.QLabel("Type:"))
        layout.addWidget(self.distortion_combo)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("drive", self.drive_knob)
        self.register_parameter("mix", self.drive_knob)

        # Set control_knob for base class functionality
        self.control_knob = self.drive_knob

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Distortion requires the In port to be connected."""
        return ["In"]

    def process(self, num_samples: int = 1):
        """Process audio through the distortion effect.

        Args:
            num_samples: Number of samples to process
        """
        if not self.in_port.is_connected:
            self.out_port.write(np.zeros(num_samples, dtype=np.float32))
            return

        samples = self.in_port.read(num_samples)
        if samples is None:
            self.out_port.write(np.zeros(num_samples, dtype=np.float32))
            return

        if self.component is None:
            self.component = self.create_engine_component()

        self.component.drive = self.drive_knob.get_value()
        self.component.mix = self.mix_knob.get_value()
        self.component.distortion_type = self.distortion_combo.currentText()
        self.out_port.write(self.component(samples))

    def get_cv_range(self, port_name: str = "Mod") -> tuple[float, float]:
        """Distortion expects bipolar CV range [-1, 1].

        Returns:
            (-1.0, 1.0) - bipolar range
        """
        return -1.0, 1.0

    def _on_type_changed(self, distortion_type: str):
        """Handle distortion type change."""
        self.component = self.create_engine_component()
        self.parameter_changed.emit("distortion_type", distortion_type)

    # Implement abstract methods from ModulatedModuleBase
    def create_modulated_component(self, mod_comp):
        """Create modulated distortion (not implemented yet)."""

    def create_unmodulated_component(self):
        """Create simple Distortion without modulation."""
        drive = self.drive_knob.get_value()
        mix = self.mix_knob.get_value()
        return Distortion(drive=drive, mix=mix)
