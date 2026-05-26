from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import Panner
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget

if TYPE_CHECKING:
    from src.gui.core.port import Port

logger = logging.getLogger(__name__)


@register_module()
class SimplePannerModule(ModuleWidget):
    """Simple panner module without modulation input."""

    metadata = ModuleMetadata(
        title="Panner",
        category=ModuleCategory.MODIFIER,
        description="Simple stereo panner without modulation input",
    )

    def __init__(self):
        """Initialize simple panner module."""
        super().__init__(
            width=140,
            height=150,
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
        self.pan_knob = Knob("Pan", -1.0, 1.0, 0.0)
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

    def process(self, num_samples: int = 1):
        """Apply stereo panning to input signal.

        Reads from the input port, applies panning control, and writes to the output
        port.

        Args:
            num_samples: Number of samples to process (default: 1 for per-sample
                processing)
        """
        if not self.in_port.is_connected:
            self.out_port.write(np.zeros(num_samples, dtype=np.float32))
            return

        # Read input signal - MUST pass num_samples to trigger upstream generation
        input_signal = self.in_port.read(num_samples)
        if input_signal is None:
            self.out_port.write(np.zeros(num_samples, dtype=np.float32))
            return

        # Apply panning - use vectorized method for proper stereo output
        # Panner returns tuple (left, right), need to convert to (N, 2) stereo format
        if isinstance(input_signal, np.ndarray) and len(input_signal) > 1:
            # Use vectorized panning for arrays
            left, right = self.component.pan_vectorized(input_signal)
            # Convert to stereo format (N, 2)
            output_signal = np.column_stack((left, right))
        else:
            # Single sample or scalar - use __call__
            left, right = self.component(input_signal)
            output_signal = np.array([[left, right]], dtype=np.float32)

        # Write to output port
        self.out_port.write(output_signal)
