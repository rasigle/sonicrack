from typing import Any, Optional

from PyQt6.QtGui import QColor

from src.engine import WaveAdder
from src.gui.audio_module_interface import ModuleCategory
from src.gui.widgets.module_widget import ModuleWidget


class MixerModule(ModuleWidget):
    """Mixer module for combining multiple audio signals.

    Uses WaveAdder to mix multiple inputs together (averages them).
    """

    def __init__(self):
        """Initialize mixer module."""
        super().__init__(width=160, height=200, color=QColor(100, 150, 100))

        # Add multiple input ports
        self.in1_port = self.add_input_port("In 1")
        self.in2_port = self.add_input_port("In 2")
        self.in3_port = self.add_input_port("In 3")
        self.in4_port = self.add_input_port("In 4")

        # Add output port
        self.out_port = self.add_output_port("Out")

        # Use helper methods for UI construction (no controls needed for mixer)
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()
        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.input_components = []  # Will be populated by patch compiler

        self.component = self.create_component()

    # AudioModuleInterface implementation
    @property
    def module_title(self) -> str:
        """Return the module title."""
        return "Mixer"

    @property
    def module_category(self) -> ModuleCategory:
        """Return MIXER since this combines multiple inputs."""
        return ModuleCategory.MIXER

    def create_component(
        self,
        input_components: Optional[list[Any]] = None,
        modulation_components: Optional[dict[str, Any]] = None,
    ):
        """Create the mixer component."""
        if input_components and len(input_components) > 0:
            return WaveAdder(*input_components)
        # Fallback for old code path
        if self.input_components and len(self.input_components) > 0:
            return WaveAdder(*self.input_components)
        return None
