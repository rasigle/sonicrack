import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine.effects import Reverb
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.modules._modulated_base import ModulatedModuleBase
from src.gui.widgets import Knob

logger = logging.getLogger(__name__)


@register_module()
class ReverbModule(ModulatedModuleBase):
    """Reverb module"""

    metadata = ModuleMetadata(
        title="Reverb",
        category=ModuleCategory.MODIFIER,
        description="Apply reverb effect to audio signal",
    )

    def __init__(self):
        """Initialize panner module."""
        super().__init__(
            width=140,
            height=280,
            color=QColor(180, 80, 180),
        )

        # Add ports
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Drive knob
        self.room_size_knob = Knob("Room Size", 0.0, 1.0, 0.5)
        self.room_size_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "room_size", self.room_size_knob.get_value()
            )
        )
        layout.addWidget(self.room_size_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        # Damping knob
        self.damping_knob = Knob("Damping", 0.0, 1.0, 0.5)
        self.damping_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "damping", self.damping_knob.get_value()
            )
        )
        layout.addWidget(self.damping_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        # Mix knob
        self.mix_knob = Knob("Mix", 0.0, 1.0, 0.5)
        self.mix_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("mix", self.mix_knob.get_value())
        )
        layout.addWidget(self.mix_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("drive", self.room_size_knob)
        self.register_parameter("mix", self.room_size_knob)

        # Set control_knob for base class functionality
        self.control_knob = self.room_size_knob

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Panner requires the In port to be connected."""
        return ["In"]

    def get_cv_range(self, port_name: str = "Mod") -> tuple[float, float]:
        """Panner expects bipolar CV range [-1, 1] for pan position.

        Returns:
            (-1.0, 1.0) - bipolar range for pan control
        """
        return -1.0, 1.0

    # Implement abstract methods from ModulatedModuleBase
    def create_modulated_component(self, mod_comp):
        """Create ModulatedPanner with modulation."""
        pass

    def create_unmodulated_component(self):
        """Create simple Panner without modulation."""
        room_size = self.room_size_knob.get_value()
        damping = self.damping_knob.get_value()
        mix = self.mix_knob.get_value()
        return Reverb(room_size=room_size, damping=damping, mix=mix)
