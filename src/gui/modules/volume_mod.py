import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.constants import DEFAULT_GAIN_DB
from src.engine import ModulatedVolume, Volume
from src.gui.audio_module_interface import ModuleCategory, ModuleMetadata
from src.gui.module_registry import register_module
from src.gui.modules.modulated_base import ModulatedModuleBase
from src.gui.widgets import Knob

logger = logging.getLogger(__name__)


@register_module()
class VolumeModule(ModulatedModuleBase):
    """Volume/Gain module with modulation support."""

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
        self.gain_knob = Knob("Gain (dB)", -60, 12, DEFAULT_GAIN_DB, logarithmic=False)
        self.gain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("gain_db", self.gain_knob.get_value())
        )
        layout.addWidget(self.gain_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("gain_db", self.gain_knob)

        # Set control_knob for base class functionality
        self.control_knob = self.gain_knob

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Volume requires the In port to be connected."""
        return ["In"]

    # Implement abstract methods from ModulatedModuleBase
    def create_modulated_component(self, mod_comp):
        """Create ModulatedVolume with modulation."""
        return ModulatedVolume(mod_comp)

    def create_unmodulated_component(self):
        """Create simple Volume without modulation."""
        gain_db = self.gain_knob.get_value()
        return Volume(gain_db=gain_db)
