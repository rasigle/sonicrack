from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import Panner
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, read_samples, silence
from src.gui.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget

if TYPE_CHECKING:
    from src.gui.core.port import Port

logger = logging.getLogger(__name__)


@register_module()
class SimplePannerModule(ModuleWidget):
    """Simple panner module without modulation input."""

    runtime_kind = "panner"

    metadata = ModuleMetadata(
        title="Panner",
        category=ModuleCategory.MODIFIER,
        description="Simple stereo panner without modulation input",
    )

    def __init__(self):
        """Initialize simple panner module."""
        super().__init__(
            width=140,
            height=175,
            color=QColor(160, 60, 160),
        )

        # Create engine component FIRST (before ports)
        self.component = self.create_engine_component()

        # Add ports with component reference
        self.in_port: Port = self.add_input("In")
        self.out_port: Port = self.add_output("Out", component=self.component)

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Pan knob
        self.pan_knob = Knob(
            label="Pan", min_value=-1.0, max_value=1.0, default_value=0.0
        )
        self.pan_knob.value_changed.connect(self._on_pan_changed)
        layout.addWidget(self.pan_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("position", self.pan_knob)

    def _on_pan_changed(self):
        """Handle pan knob changes by updating pan component.

        This method is called whenever the pan knob value changes.
        """
        pan_value = self.pan_knob.get_value()

        # Update the Pan component wave_range (click-free)
        if self.component:
            self.component.position = pan_value
            logger.debug(f"Panner: position value set to {pan_value:.3f}")

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Panner requires the In port to be connected."""
        return ["In"]

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the panner component."""
        return Panner(0.0)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Pan the connected input for one render cycle."""
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        samples = read_samples(self.in_port, num_samples)

        # Update component position
        position = float_parameter(parameters, "position", self.pan_knob.get_value)
        self.component.position = position

        # Use component to pan the samples - output as stereo (N, 2)
        left, right = self.component.pan_vectorized(samples)
        self.out_port.write(np.column_stack((left, right)))
