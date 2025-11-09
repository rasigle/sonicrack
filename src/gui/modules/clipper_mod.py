"""Modulated Clipper module - clipper with CV threshold control."""

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import ModulatedClipper, Clipper
from src.gui.audio_module_interface import ModuleCategory, ModuleMetadata
from src.gui.module_registry import register_module
from src.gui.modules._modulated_base import ModulatedModuleBase
from src.gui.widgets import Knob

logger = logging.getLogger(__name__)


@register_module()
class ClipperModulatedModule(ModulatedModuleBase):
    """Clipper module with modulation support for dynamic threshold control."""

    metadata = ModuleMetadata(
        title="Clipper (Mod)",
        category=ModuleCategory.MODIFIER,
        description="Audio clipper with CV threshold control",
    )

    def __init__(self):
        """Initialize modulated clipper module."""
        super().__init__(
            width=140,
            height=180,
            color=QColor(200, 150, 80),
        )

        # Add ports
        self.in_port = self.add_input_port("In")
        self.mod_port = self.add_input_port("Mod")
        self.out_port = self.add_output_port("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Threshold knob (0.1 to 1.0, default 1.0 = no clipping)
        self.threshold_knob = Knob("Threshold", 0.1, 1.0, 1.0)
        self.threshold_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "wave_range",
                (-self.threshold_knob.get_value(), self.threshold_knob.get_value()),
            )
        )
        layout.addWidget(self.threshold_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("threshold", self.threshold_knob)

        # Set control_knob for base class functionality
        self.control_knob = self.threshold_knob

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Clipper requires the In port to be connected."""
        return ["In"]

    # Implement abstract methods from ModulatedModuleBase
    def create_modulated_component(self, mod_comp):
        """Create ModulatedClipper with modulation.

        The modulator controls the clipping threshold.
        """
        return ModulatedClipper(mod_comp)

    def create_unmodulated_component(self):
        """Create simple Clipper without modulation."""
        threshold = self.threshold_knob.get_value()
        return Clipper((-threshold, threshold))
