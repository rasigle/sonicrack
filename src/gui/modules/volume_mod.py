from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import ModulatedVolume, Volume
from src.gui.audio_module_interface import ModuleCategory, ModuleMetadata
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.module_registry import register_module


@register_module()
class VolumeModule(ModuleWidget):
    """Volume/Gain module."""

    metadata = ModuleMetadata(
        title="Volume (Mod)",
        category=ModuleCategory.MODIFIER,
        description="Volume control with modulation input",
    )

    def __init__(self):
        """Initialize volume module."""
        super().__init__(
            width=140,
            height=180,
            color=QColor(180, 120, 80),
        )

        # Add ports
        self.in_port = self.add_input_port("In")
        self.mod_port = self.add_input_port("Mod")
        self.out_port = self.add_output_port("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Gain in dB (linear mapping of dB values, since dB is already logarithmic)
        # Range: -60 dB (very quiet) to +12 dB (boost)
        # Default: -20 dB (safe for mixing)
        self.gain_knob = Knob("Gain (dB)", -60, 12, -20, logarithmic=False)
        self.gain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("gain_db", self.gain_knob.get_value())
        )
        layout.addWidget(self.gain_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("gain_db", self.gain_knob)

        self.component = self.create_component()

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Volume requires the In port to be connected."""
        return ["In"]

    def get_modulation_inputs(self) -> list[str]:
        """Volume can optionally use Mod port for modulation."""
        return ["Mod"]

    def create_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the volume component."""

        # Check if modulation is provided
        mod_comp = None
        if modulation_components and "Mod" in modulation_components:
            mod_comp = modulation_components["Mod"]

        if mod_comp:
            return ModulatedVolume(mod_comp)

        gain_db = self.gain_knob.get_value()
        return Volume(gain_db=gain_db)
