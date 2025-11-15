"""Example custom module plugin.

This demonstrates how to create a custom module that automatically
registers itself using the @register_module decorator.

To use this plugin:
1. Save this file in a 'plugins' directory
2. Call load_plugin("plugins/custom_filter.py") at startup
3. The module will appear in the module registry automatically
"""

from typing import Optional, List, Any
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt

from src.gui.widgets.module_widget import ModuleWidget
from src.gui.widgets import Knob
from src.gui.core.module_registry import register_module
from src.engine.modifier import Modifier
import numpy as np


# Custom audio engine component
class SimpleFilter(Modifier):
    """Simple low-pass filter."""

    def __init__(self, cutoff: float = 1000.0, resonance: float = 0.5):
        super().__init__()
        self.cutoff = cutoff
        self.resonance = resonance
        self.last_output = 0.0

    def get_samples_vectorized(self, num_samples: int) -> np.ndarray:
        """Apply simple filtering."""
        if not self._get_input():
            return np.zeros(num_samples)

        input_samples = self._get_input().get_samples(num_samples)

        # Simple one-pole filter
        alpha = min(1.0, self.cutoff / 22050.0)
        output = np.zeros_like(input_samples)

        if input_samples.ndim == 1:
            for i in range(len(input_samples)):
                output[i] = self.last_output + alpha * (
                    input_samples[i] - self.last_output
                )
                self.last_output = output[i]
        else:
            for i in range(len(input_samples)):
                output[i] = self.last_output + alpha * (
                    input_samples[i] - self.last_output
                )
                self.last_output = output[i]

        return output


# Custom module with automatic registration via decorator
@register_module()
class SimpleFilterModule(ModuleWidget):
    """Simple filter module - plugin example.

    This module demonstrates:
    - Using @register_module decorator for automatic registration
    - Using UI helper methods
    - Using parameter registration
    - Following the standard module pattern
    """

    def __init__(self):
        """Initialize the filter module."""
        super().__init__(
            "Simple Filter", width=180, height=200, color=QColor(100, 200, 150)
        )

        # Add ports
        self.in_port = self.add_input_port("In")
        self.out_port = self.add_output_port("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Add controls
        self.cutoff_knob = Knob("Cutoff", 20, 20000, 1000)
        self.cutoff_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("cutoff", self.cutoff_knob.get_value())
        )
        layout.addWidget(self.cutoff_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.resonance_knob = Knob("Resonance", 0.0, 1.0, 0.5)
        self.resonance_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "resonance", self.resonance_knob.get_value()
            )
        )
        layout.addWidget(self.resonance_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("cutoff", self.cutoff_knob)
        self.register_parameter("resonance", self.resonance_knob)

        self.component = self.create_engine_component()

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> List[str]:
        """Filter requires the In port to be connected."""
        return ["In"]

    def create_engine_component(
        self,
        input_components: Optional[List[Any]] = None,
        modulation_components: Optional[dict[str, Any]] = None,
    ):
        """Create the filter component."""
        cutoff = self.cutoff_knob.get_value()
        resonance = self.resonance_knob.get_value()
        return SimpleFilter(cutoff, resonance)


# The @register_module decorator above automatically registers this module
# when this file is imported or loaded as a plugin!

print("✅ Plugin loaded: Simple Filter Module")
