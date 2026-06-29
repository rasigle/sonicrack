import logging

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from src.engine.effects import Delay
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.modules._modulated_base import ModulatedModuleBase
from src.gui.runtime import RuntimeParameters
from src.gui.runtime_helpers import float_parameter, read_samples, silence
from src.gui.widgets import Knob

logger = logging.getLogger(__name__)


@register_module()
class DelayModule(ModulatedModuleBase):
    """Delay module"""

    runtime_kind = "delay"
    runtime_input_names = ("In",)
    runtime_output_names = ("Out",)
    runtime_parameter_names = ("delay_time", "feedback", "mix")

    metadata = ModuleMetadata(
        title="Delay",
        category=ModuleCategory.MODIFIER,
        description="Apply delay effect to audio signal",
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

        self.delay_time = Knob("Time", 0.0, 3.0, 0.5)
        self.delay_time.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "delay_time", self.delay_time.get_value()
            )
        )
        knobs_row.addWidget(self.delay_time)

        self.feedback_knob = Knob("Feedback", 0.0, 1.0, 0.5)
        self.feedback_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "feedback", self.feedback_knob.get_value()
            )
        )
        knobs_row.addWidget(self.feedback_knob)
        layout.addLayout(knobs_row)

        mix_row = QHBoxLayout()
        mix_row.setSpacing(10)
        self.mix_knob = Knob("Mix", 0.0, 1.0, 0.5)
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
        self.register_parameter("drive", self.delay_time)
        self.register_parameter("mix", self.delay_time)

        # Set control_knob for base class functionality
        self.control_knob = self.delay_time

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Delay requires the In port to be connected."""
        return ["In"]

    def get_cv_range(self, port_name: str = "Mod") -> tuple[float, float]:
        """Delay expects bipolar CV range [-1, 1].

        Returns:
            (-1.0, 1.0) - bipolar range
        """
        return -1.0, 1.0

    # Implement abstract methods from ModulatedModuleBase
    def create_modulated_component(self, mod_comp):
        """Create modulated delay (not implemented yet)."""

    def create_unmodulated_component(self):
        """Create simple Delay without modulation."""
        delay_time = self.delay_time.get_value()
        feedback = self.feedback_knob.get_value()
        mix = self.mix_knob.get_value()
        return Delay(delay_time=delay_time, feedback=feedback, mix=mix)

    def process_runtime(
        self, num_samples: int, parameters: RuntimeParameters
    ) -> None:
        """Apply delay during an engine-owned render cycle."""
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        if self.component is None:
            self.component = self.create_unmodulated_component()

        self.component.delay_time = float_parameter(
            parameters, "delay_time", self.delay_time.get_value
        )
        self.component.feedback = float_parameter(
            parameters, "feedback", self.feedback_knob.get_value
        )
        self.component.mix = float_parameter(parameters, "mix", self.mix_knob.get_value)

        self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
