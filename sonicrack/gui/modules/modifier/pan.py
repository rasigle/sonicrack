from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from soniclab import ModulatedPanner, Panner

from sonicrack.gui.modules._modulated_base import ModulatedModuleBase
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)


@register_module()
class PannerModule(ModulatedModuleBase):
    """Stereo panner with optional modulation support.

    Without a cable on Mod, the Pan knob sets stereo position.
    With Mod connected, the knob controls modulation depth and CV drives pan.
    """

    runtime_kind = "panner"

    metadata = ModuleMetadata(
        title="Panner",
        category=ModuleCategory.MODIFIER,
        description="Stereo panner with optional CV modulation",
    )

    def __init__(self):
        """Initialize panner module."""
        super().__init__(
            width=140,
            height=150,
            color=QColor(180, 80, 180),
        )

        self._setup_modulated_io()

        self.pan_knob = Knob(
            label="Pan", min_value=-1.0, max_value=1.0, default_value=0.0
        )
        layout = self._begin_controls()
        self.bind_parameter_knob(
            self.pan_knob, "position", on_change=self._on_pan_changed
        )
        layout.addWidget(self.pan_knob, alignment=Qt.AlignmentFlag.AlignCenter)
        self._finish_controls(layout)

        self.register_parameter("position", self.pan_knob)
        self.control_knob = self.pan_knob
        self.component = self.create_unmodulated_component()

    def _on_pan_changed(self, pan_value: float) -> None:
        """Update pan position when not modulated."""
        if self.component is not None and not self.mod_port.is_connected:
            self.component.position = pan_value
            logger.debug(f"Panner: position value set to {pan_value:.3f}")

    def get_required_inputs(self) -> list[str]:
        """Panner requires the In port to be connected."""
        return ["In"]

    def get_cv_range(self, port_name: str = "Mod") -> tuple[float, float]:
        """Panner expects bipolar CV range [-1, 1] for pan position."""
        return -1.0, 1.0

    def create_modulated_component(self, mod_comp):
        """Create ModulatedPanner with modulation."""
        return ModulatedPanner(mod_comp)

    def create_unmodulated_component(self):
        """Create simple Panner without modulation."""
        return Panner(self.pan_knob.get_value())

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Pan the connected input for one render cycle."""
        samples = self.read_modulated_input_or_silence(num_samples)
        if samples is None:
            return

        with self._component_lock:
            position = float_parameter(parameters, "position", self.pan_knob.get_value)

            if self.mod_port.is_connected:
                modulation_amount = (position + 1.0) / 2.0
                if self.port_adapter is not None:
                    self.port_adapter.modulation_amount = modulation_amount
            elif self.component is not None:
                self.component.position = position

            if self.component is not None:
                left, right = self.component(samples)
                self.out_port.write(np.column_stack((left, right)))
            else:
                self._write_silence(num_samples)
