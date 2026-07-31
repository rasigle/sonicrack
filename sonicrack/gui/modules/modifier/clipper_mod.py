"""Modulated Clipper module - clipper with CV threshold control."""

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from soniclab import Clipper, ModulatedClipper

from sonicrack.gui.modules._modulated_base import ModulatedModuleBase
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter
from sonicrack.runtime.specs import RuntimeParameters

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
            height=150,
            color=QColor(200, 150, 80),
        )

        self._setup_modulated_io()

        self.threshold_knob = Knob(
            label="Threshold",
            description="Sets the clipping threshold",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        layout = self._begin_controls()
        self.bind_parameter_knob(self.threshold_knob, "threshold")
        layout.addWidget(self.threshold_knob, alignment=Qt.AlignmentFlag.AlignCenter)
        self._finish_controls(layout)

        self.register_parameter("threshold", self.threshold_knob)
        self.control_knob = self.threshold_knob
        self.component = self.create_unmodulated_component()

    def get_required_inputs(self) -> list[str]:
        """Clipper requires the In port to be connected."""
        return ["In"]

    def create_modulated_component(self, mod_comp):
        """Create ModulatedClipper with modulation."""
        return ModulatedClipper(mod_comp)

    def create_unmodulated_component(self):
        """Create simple Clipper without modulation."""
        threshold = self.threshold_knob.get_value()
        return Clipper((-threshold, threshold))

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Clip the connected input for one render cycle."""
        samples = self.read_modulated_input_or_silence(num_samples)
        if samples is None:
            return

        with self._component_lock:
            threshold_value = float_parameter(
                parameters, "threshold", self.threshold_knob.get_value
            )

            if self.mod_port.is_connected:
                if self.port_adapter is not None:
                    self.port_adapter.modulation_amount = threshold_value
            elif self.component is not None:
                self.component.wave_range = (-threshold_value, threshold_value)

            if self.component is not None:
                self.out_port.write(self.component(samples))
            else:
                self._write_silence(num_samples)
