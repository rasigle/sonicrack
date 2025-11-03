from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import ModulatedVolume, Volume
from src.gui.audio_module_interface import ModuleCategory
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.module_registry import register_module


@register_module()
class VolumeModule(ModuleWidget):
    """Volume/Gain module."""

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

        # Volume knob
        self.volume_knob = Knob("Volume", 0.0, 2.0, 1.0)
        self.volume_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("volume", self.volume_knob.get_value())
        )
        layout.addWidget(self.volume_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("volume", self.volume_knob)

        self.input_component = None
        self.modulator_component = None
        self.component = self.create_component()

    # AudioModuleInterface implementation
    @property
    def module_title(self) -> str:
        """Return the module title."""
        return "Volume (Mod)"

    @property
    def module_description(self) -> str:
        """Return module description."""
        return "Volume control with modulation input"

    @property
    def module_category(self) -> ModuleCategory:
        """Return MODIFIER since this modifies audio input."""
        return ModuleCategory.MODIFIER

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
        volume = self.volume_knob.get_value()

        # Check if modulation is provided
        mod_comp = None
        if modulation_components and "Mod" in modulation_components:
            mod_comp = modulation_components["Mod"]
        elif self.modulator_component:
            mod_comp = self.modulator_component

        if mod_comp:
            return ModulatedVolume(mod_comp)
        return Volume(volume)
