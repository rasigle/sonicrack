import logging

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.effects import Reverb

from sonicrack.gui.core.module import ModuleCategory, ModuleMetadata
from sonicrack.gui.core.runtime import RuntimeParameters
from sonicrack.gui.core.runtime_helpers import float_parameter, read_samples, silence
from sonicrack.gui.module_registry import register_module
from sonicrack.gui.modules._modulated_base import ModulatedModuleBase
from sonicrack.gui.widgets import Knob

logger = logging.getLogger(__name__)


@register_module()
class ReverbModule(ModulatedModuleBase):
    """Reverb module"""

    runtime_kind = "reverb"
    metadata = ModuleMetadata(
        title="Reverb",
        category=ModuleCategory.MODIFIER,
        description="Apply reverb effect to audio signal",
    )

    def __init__(self):
        """Initialize panner module."""
        super().__init__(
            width=220,
            height=240,
            color=QColor(180, 80, 180),
        )

        # Add ports
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.component = None

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        knobs_row = QHBoxLayout()
        knobs_row.setSpacing(10)

        self.room_size_knob = Knob(
            label="Room Size",
            description="Adjusts the perceived size of the room",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self.room_size_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "room_size", self.room_size_knob.get_value()
            )
        )
        knobs_row.addWidget(self.room_size_knob)

        self.damping_knob = Knob(
            label="Damping",
            description="Controls the high-frequency damping of the reverb",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self.damping_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "damping", self.damping_knob.get_value()
            )
        )
        knobs_row.addWidget(self.damping_knob)
        layout.addLayout(knobs_row)

        mix_row = QHBoxLayout()
        mix_row.setSpacing(10)
        self.mix_knob = Knob(
            label="Mix",
            description="Controls the dry/wet mix of the reverb effect",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self.mix_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("mix", self.mix_knob.get_value())
        )
        mix_row.addStretch()
        mix_row.addWidget(self.mix_knob)
        mix_row.addStretch()
        layout.addLayout(mix_row)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("room_size", self.room_size_knob)
        self.register_parameter("damping", self.damping_knob)
        self.register_parameter("mix", self.mix_knob)

        # Set control_knob for base class functionality
        self.control_knob = self.room_size_knob

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Reverb requires the In port to be connected."""
        return ["In"]

    def get_cv_range(self, port_name: str = "Mod") -> tuple[float, float]:
        """Reverb expects bipolar CV range [-1, 1].

        Returns:
            (-1.0, 1.0) - bipolar range
        """
        return -1.0, 1.0

    # Implement abstract methods from ModulatedModuleBase
    def create_modulated_component(self, mod_comp):
        """Create modulated reverb (not implemented yet)."""

    def create_unmodulated_component(self):
        """Create simple Reverb without modulation."""
        room_size = self.room_size_knob.get_value()
        damping = self.damping_knob.get_value()
        mix = self.mix_knob.get_value()
        return Reverb(room_size=room_size, damping=damping, mix=mix)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Apply reverb during an engine-owned render cycle."""
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        # Thread-safe component access with lock
        with self._component_lock:
            if self.component is None:
                self.component = self.create_unmodulated_component()

            self.component.room_size = float_parameter(
                parameters, "room_size", self.room_size_knob.get_value
            )
            self.component.damping = float_parameter(
                parameters, "damping", self.damping_knob.get_value
            )
            self.component.mix = float_parameter(
                parameters, "mix", self.mix_knob.get_value
            )

            self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
