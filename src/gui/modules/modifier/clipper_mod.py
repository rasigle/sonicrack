"""Modulated Clipper module - clipper with CV threshold control."""

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import Clipper, ModulatedClipper
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, read_samples, silence
from src.gui.module_registry import register_module
from src.gui.modules._modulated_base import ModulatedModuleBase
from src.gui.widgets import Knob

logger = logging.getLogger(__name__)


@register_module()
class ClipperModulatedModule(ModulatedModuleBase):
    """Clipper module with modulation support for dynamic threshold control."""

    runtime_kind = "clipper"

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
        self.in_port = self.add_input("In")
        self.mod_port = self.add_input("Mod")
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Threshold knob (0.1 to 1.0, default 1.0 = no clipping)
        self.threshold_knob = Knob(
            label="Threshold",
            description="Sets the clipping threshold",
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
        )
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

        self.component: Clipper | None = None

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

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Clip the connected input for one render cycle."""
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        samples = read_samples(self.in_port, num_samples)
        threshold = float_parameter(
            parameters, "threshold", self.threshold_knob.get_value
        )
        if self.component is None:
            self.component = self.create_unmodulated_component()
        self.component.wave_range = (-threshold, threshold)

        self.out_port.write(samples.clip(-threshold, threshold))
