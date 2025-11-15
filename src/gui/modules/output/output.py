from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget
from src.utils.math import db_to_linear


@register_module()
class OutputModule(ModuleWidget):
    """Output module (sink for audio) with professional dB volume control.

    Master Volume Control:
    - Use gain_db for professional audio control (recommended)
    - Range: -60 dB (very quiet) to +6 dB (boost)
    - Default: -3 dB (safe headroom for mixing)
    - Linear knob for dB values (dB is already logarithmic)

    The knob displays dB values but internally converts to linear
    amplitude for the audio engine.
    """

    metadata = ModuleMetadata(
        title="Output",
        category=ModuleCategory.OUTPUT,
        description="Audio output with dB-controlled master volume",
    )

    # Special signal for master volume changes (bypasses hot-swapping)
    master_volume_changed = pyqtSignal(float)

    def __init__(self):
        """Initialize output module."""
        super().__init__(
            width=140,
            height=120,
            color=QColor(200, 80, 80),
        )

        # Add input port
        self.in_port = self.add_input_port("In")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Master volume in dB (professional control)
        # Range: -60 dB to +6 dB, default: -3 dB (safe headroom)
        # Linear knob (dB is already logarithmic scale)
        self.volume_knob = Knob("Master (dB)", -60, 6, -3, logarithmic=False)
        self.volume_knob.value_changed.connect(self._on_volume_changed)
        layout.addWidget(self.volume_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("master_volume_db", self.volume_knob)

        self.input_component = None

    def _on_volume_changed(self):
        """Handle volume knob changes - convert dB to linear and emit."""
        db_value = self.volume_knob.get_value()
        self.master_volume_changed.emit(db_to_linear(db_value))

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Output requires the In port to be connected."""
        return ["In"]

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Output doesn't create a component, it returns the input component."""
        if input_components and len(input_components) > 0:
            return input_components[0]
        return self.input_component

    def get_master_volume(self) -> float:
        """Get the master volume level in linear scale.

        Returns:
            Linear amplitude (0.0 to 2.0+) converted from dB.
        """
        return db_to_linear(self.volume_knob.get_value())
