from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.constants import DEFAULT_GAIN_DB
from src.engine import Volume
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.core.module_registry import register_module


@register_module()
class SimpleVolumeModule(ModuleWidget):
    """Simple volume/gain module without modulation input."""

    metadata = ModuleMetadata(
        title="Volume",
        category=ModuleCategory.MODIFIER,
        description="Simple volume/gain control",
    )

    def __init__(self):
        """Initialize simple volume module."""
        super().__init__(
            width=140,
            height=150,
            color=QColor(160, 100, 60),
        )

        # Add ports (no modulation input)
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Gain in dB (linear mapping of dB values, since dB is already logarithmic)
        # Range: -60 dB (very quiet) to +12 dB (boost)
        # Default: -20 dB (safe for mixing)
        self.gain_knob = Knob("Gain (dB)", -60, 12, DEFAULT_GAIN_DB, logarithmic=False)
        self.gain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("gain_db", self.gain_knob.get_value())
        )
        layout.addWidget(self.gain_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("gain_db", self.gain_knob)

        self.component = self.create_engine_component()

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Volume requires the In port to be connected."""
        return ["In"]

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the volume component."""
        gain_db = self.gain_knob.get_value()
        return Volume(gain_db=gain_db)
