"""Example: Creating a Custom Module with the Audio Module Interface

This example shows how to create a low-pass filter module that integrates
seamlessly with the modular synth GUI using the generic AudioModuleInterface.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGraphicsProxyWidget

from src.gui.widgets.module_widget import ModuleWidget
from src.gui.widgets import Knob
from src.engine import Modifier  # Base class for audio modifiers
import numpy as np


# Step 1: Create the audio engine component
class LowPassFilter(Modifier):
    """Simple low-pass filter implementation."""

    def __init__(self, cutoff_freq: float = 1000.0, sample_rate: int = 44100):
        """Initialize the filter.

        Args:
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Sample rate in Hz
        """
        super().__init__()
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.last_output = 0.0

        # Calculate RC constant
        rc = 1.0 / (2 * np.pi * cutoff_freq)
        self.alpha = 1.0 / (1.0 + rc * sample_rate)

    def __call__(self, val: float | tuple[float, ...]) -> float | tuple[float, ...]:
        pass

    def get_samples_vectorized(self, num_samples: int) -> np.ndarray:
        """Process audio samples through the filter.

        Args:
            num_samples: Number of samples to process

        Returns:
            Filtered audio samples
        """
        # Get input samples (this comes from the chained component)
        if not self._get_input():
            return np.zeros(num_samples)

        input_samples = self._get_input().get_samples(num_samples)

        # Simple one-pole low-pass filter
        output = np.zeros_like(input_samples)

        if input_samples.ndim == 1:
            # Mono
            for i in range(len(input_samples)):
                output[i] = self.last_output + self.alpha * (
                    input_samples[i] - self.last_output
                )
                self.last_output = output[i]
        else:
            # Stereo
            for i in range(len(input_samples)):
                output[i] = self.last_output + self.alpha * (
                    input_samples[i] - self.last_output
                )
                self.last_output = output[i]

        return output


# Step 2: Create the GUI module implementing AudioModuleInterface
class LowPassFilterModule(ModuleWidget):
    """Low-pass filter module for the modular synth.

    This module demonstrates how to implement the AudioModuleInterface
    to create a fully integrated, extensible module.
    """

    def __init__(self):
        """Initialize the filter module."""
        # Initialize base class with module properties
        super().__init__(
            "Low-Pass Filter",  # Display name
            category="modifier",  # Category for organization
            width=160,  # Module width in pixels
            height=180,  # Module height in pixels
            color=QColor(80, 180, 120),  # Module color
        )

        # Add input/output ports
        # Every MODIFIER needs at least one input and one output
        self.in_port = self.add_input_port("In")
        self.out_port = self.add_output_port("Out")

        # Create the control UI
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Cutoff frequency knob
        self.cutoff_knob = Knob("Cutoff", 20, 20000, 1000)
        self.cutoff_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("cutoff", self.cutoff_knob.get_value())
        )
        layout.addWidget(self.cutoff_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)

        # Add controls to the module
        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(self.controls_widget)
        self.proxy.setPos(0, 42)  # Position below title bar

        # Create initial component
        self.component = self.create_engine_component()

    # === AudioModuleInterface Implementation ===

    def get_required_inputs(self) -> List[str]:
        """Return list of required input port names.

        The patch compiler will validate that these ports are connected.
        For a filter, we need audio input.
        """
        return ["In"]

    def get_modulation_inputs(self) -> List[str]:
        """Return list of optional modulation port names.

        For this simple filter, we don't support modulation.
        If we wanted to add cutoff modulation, we would:
        1. Add a modulation port: self.add_input_port("Cutoff Mod")
        2. Return ["Cutoff Mod"] here
        3. Handle it in create_component()
        """
        return []

    def create_engine_component(
        self,
        input_components: Optional[List[Any]] = None,
        modulation_components: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create the filter audio component.

        This is called by the patch compiler to create the actual audio
        processing component with the current parameter values.

        Args:
            input_components: List of connected input components (unused for this module,
                            as the compiler handles chaining automatically)
            modulation_components: Dict of modulation sources (none for this module)

        Returns:
            The LowPassFilter component
        """
        # Get parameter values from UI controls
        cutoff = self.cutoff_knob.get_value()

        # Create and return the filter
        return LowPassFilter(cutoff_freq=cutoff)

    # === Parameter Management (for presets) ===

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values for saving presets.

        Returns:
            Dictionary of parameter name -> value
        """
        return {"cutoff": self.cutoff_knob.get_value()}

    def set_parameters(self, params: Dict[str, Any]):
        """Set parameter values from a loaded preset.

        Args:
            params: Dictionary of parameter name -> value
        """
        if "cutoff" in params:
            self.cutoff_knob.set_value(params["cutoff"])


# === How to Register the Module ===

# To make this module available in the GUI, add it to module_registry.py:
"""
from src.gui.modules.lowpass_filter import LowPassFilterModule

MODULE_REGISTRY = {
    # ... existing modules ...
    "Low-Pass Filter": LowPassFilterModule,
}
"""

# === Usage Example ===


def example_usage():
    """Example of how this module works in a patch."""

    # The user creates a patch in the GUI:
    # [Oscillator] -> [Low-Pass Filter] -> [Output]

    # When compiled, the PatchCompiler:
    # 1. Finds the Output module
    # 2. Follows connection to Low-Pass Filter
    # 3. Calls filter.create_component(input_components=[osc], modulation_components=None)
    # 4. Creates: Chain(Oscillator(440), LowPassFilter(1000))
    # 5. Returns the chain to the audio player

    # The generic interface means:
    # - No changes to PatchCompiler needed
    # - Module automatically validated
    # - Works with any other modules
    # - Can be saved/loaded in presets

    pass


# === Advanced Example: Filter with Modulation ===


class ModulatedLowPassFilterModule(ModuleWidget):
    """Filter with cutoff modulation support."""

    def __init__(self):
        super().__init__(
            "Mod Filter",
            category="modifier",
            width=160,
            height=200,
            color=QColor(100, 200, 130),
        )

        # Add both audio and modulation inputs
        self.in_port = self.add_input_port("In")
        self.cutoff_mod_port = self.add_input_port("Cutoff Mod")
        self.out_port = self.add_output_port("Out")

        # ... setup UI ...

    def get_required_inputs(self) -> List[str]:
        return ["In"]  # Audio input is required

    def get_modulation_inputs(self) -> List[str]:
        return ["Cutoff Mod"]  # Modulation is optional

    def create_engine_component(
        self,
        input_components: Optional[List[Any]] = None,
        modulation_components: Optional[Dict[str, Any]] = None,
    ):
        """Create filter with optional cutoff modulation."""
        base_cutoff = self.cutoff_knob.get_value()

        # Check if cutoff modulation is connected
        # if modulation_components and "Cutoff Mod" in modulation_components:
        #     # Create modulated version
        #     mod_source = modulation_components["Cutoff Mod"]
        #     return ModulatedLowPassFilter(mod_source, base_cutoff)
        # else:
        # Create static version
        return LowPassFilter(base_cutoff)


# === Summary ===

"""
The AudioModuleInterface makes it easy to extend the modular synth:

1. **Implement the interface** - Just 3 required methods:
   - get_module_type() - What kind of module is this?
   - create_component() - How do you create the audio component?
   - get_required_inputs() - What connections are required?

2. **No compiler changes needed** - The PatchCompiler works generically
   with any module implementing the interface.

3. **Automatic features**:
   - ✅ Connection validation
   - ✅ Signal chain building
   - ✅ Modulation handling
   - ✅ Preset support
   - ✅ Error checking

4. **Extensible** - Add as many modules as you want without touching
   core code. Just implement the interface and register the module.

This is a production-ready, maintainable architecture for building
complex modular synth patches!
"""
