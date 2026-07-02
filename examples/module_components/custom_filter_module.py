"""Example custom module plugin.

This demonstrates how to create a custom module that automatically
registers itself using the @register_module decorator.

To use this plugin:
1. Save this file in a 'plugins' directory
2. Call load_plugin("plugins/custom_filter.py") at startup
3. The module will appear in the module registry automatically
"""

from typing import Any

import numpy as np
from PyQt6.QtGui import QColor

from src.engine import Modifier
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


# Custom audio engine component
class SimpleFilter(Modifier):
    """Simple low-pass filter."""

    def __init__(self, cutoff: float = 1000.0, resonance: float = 0.5):
        super().__init__()
        self.cutoff = cutoff
        self.resonance = resonance
        self.last_output = 0.0

    def __call__(
        self, val: float | tuple[float, ...]
    ) -> np.ndarray | float | tuple[float, ...]:

        input_samples = np.asarray(val)

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

    metadata = ModuleMetadata(
        title="Simple Filter",
        category=ModuleCategory.MODIFIER,
        description="Simple volume/gain control",
    )

    def __init__(self):
        """Initialize the filter module."""
        super().__init__(width=180, height=200, color=QColor(100, 200, 150))

        # Add ports
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Add controls
        # Add a knob for cutoff frequency
        self.cutoff_knob = Knob(
            "Cutoff",
            description="Sets the cutoff frequency of the filter",
            min_value=20,
            max_value=20000,
            default_value=1000,
            logarithmic=True,
        )
        self.add_control("cutoff", self.cutoff_knob)

        # Add a knob for resonance
        self.resonance_knob = Knob(
            "Resonance",
            description="Adjusts the resonance of the filter",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self.add_control("resonance", self.resonance_knob)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("cutoff", self.cutoff_knob)
        self.register_parameter("resonance", self.resonance_knob)

        self.component = self.create_engine_component()

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Filter requires the In port to be connected."""
        return ["In"]

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the filter component."""
        cutoff = self.cutoff_knob.get_value()
        resonance = self.resonance_knob.get_value()
        return SimpleFilter(cutoff, resonance)


# The @register_module decorator above automatically registers this module
# when this file is imported or loaded as a plugin!

print("✅ Plugin loaded: Simple Filter Module")
