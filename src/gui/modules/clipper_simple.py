import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import Clipper
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.core.module_registry import register_module

logger = logging.getLogger(__name__)


@register_module()
class ClipperModule(ModuleWidget):
    """Clipper module for distortion/limiting."""

    metadata = ModuleMetadata(
        title="Clipper",
        category=ModuleCategory.MODIFIER,
        description="Audio clipper for distortion/limiting",
    )

    def __init__(self):
        """Initialize clipper module."""
        super().__init__(
            width=140,
            height=180,
            color=QColor(200, 150, 80),
        )

        # Add ports
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Threshold knob
        self.threshold_knob = Knob("Threshold", 0.0, 1.0, 1.0)
        self.threshold_knob.value_changed.connect(self._on_threshold_changed)
        layout.addWidget(self.threshold_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("threshold", self.threshold_knob)

        self.component = self.create_engine_component()

    def _on_threshold_changed(self):
        """Handle threshold knob changes by updating Clipper component wave_range.

        This method is called whenever the threshold knob value changes.
        It updates the clipper's wave range to [-threshold, +threshold].
        """
        new_threshold = self.threshold_knob.get_value()
        logger.debug(f"Clipper: threshold knob changed to {new_threshold:.3f}")

        # Update the Clipper component wave_range (click-free)
        if self.component:
            self.component.wave_range = (-new_threshold, new_threshold)
            logger.debug(f"Clipper: wave_range set to +/- {new_threshold:.3f}")

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Clipper requires the In port to be connected."""
        return ["In"]

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the clipper component."""
        threshold = self.threshold_knob.get_value()
        return Clipper((-threshold, threshold))

    def process(self, num_samples: int = 1):
        """Apply clipping/limiting to input signal.

        Reads from the input port, applies clipping based on threshold, and writes to
        the output port.

        Args:
            num_samples: Number of samples to process

        Note:
            This method is called in the pull-based architecture to generate output samples.
        """
        if not self.in_port.is_connected:
            self.out_port.write(np.zeros(num_samples, dtype=np.float32))
            return

        # Read input signal - MUST pass num_samples to trigger upstream generation
        input_signal = self.in_port.read(num_samples)
        if input_signal is None:
            self.out_port.write(np.zeros(num_samples, dtype=np.float32))
            return

        # Apply clipping through the component
        output_signal = self.component(input_signal)

        # Write to output port
        self.out_port.write(output_signal)
