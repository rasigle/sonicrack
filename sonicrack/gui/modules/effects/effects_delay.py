import logging

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.effects import Delay

from sonicrack.gui.modules._modulated_base import ModulatedModuleBase
from sonicrack.gui.modules.effects._cv_modulation import (
    ControlRateCvSpec,
    apply_control_rate_cv,
)
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)


@register_module()
class DelayModule(ModulatedModuleBase):
    """Delay module with optional CV for time, feedback, and mix."""

    runtime_kind = "delay"
    metadata = ModuleMetadata(
        title="Delay",
        category=ModuleCategory.MODIFIER,
        description="Apply delay effect to audio signal",
    )

    def __init__(self):
        """Initialize delay module."""
        super().__init__(
            width=220,
            height=260,
            color=QColor(180, 80, 180),
        )

        # Add ports
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.time_cv_port = self.add_input("CV_Time")
        self.feedback_cv_port = self.add_input("CV_Feedback")
        self.mix_cv_port = self.add_input("CV_Mix")
        self.component = None

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        knobs_row = QHBoxLayout()
        knobs_row.setSpacing(10)

        self.time_knob = Knob(
            label="Time",
            description="Sets the delay time",
            min_value=0.001,
            max_value=3.0,
            default_value=0.5,
        )
        self.time_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "delay_time", self.time_knob.get_value()
            )
        )
        knobs_row.addWidget(self.time_knob)

        self.feedback_knob = Knob(
            label="Feedback",
            description="Controls the amount of feedback in the delay line",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self.feedback_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "feedback", self.feedback_knob.get_value()
            )
        )
        knobs_row.addWidget(self.feedback_knob)
        layout.addLayout(knobs_row)

        mix_row = QHBoxLayout()
        mix_row.setSpacing(10)
        self.mix_knob = Knob(
            label="Mix",
            description="Controls the dry/wet mix of the delay effect",
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
        self.register_parameter("delay_time", self.time_knob)
        self.register_parameter("feedback", self.feedback_knob)
        self.register_parameter("mix", self.mix_knob)

        # Set control_knob for base class functionality
        self.control_knob = self.time_knob

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Delay requires the In port to be connected."""
        return ["In"]

    def get_modulation_inputs(self) -> list[str]:
        """Delay accepts CV modulation for time, feedback, and mix."""
        return ["CV_Time", "CV_Feedback", "CV_Mix"]

    def get_cv_range(self, port_name: str = "CV_Time") -> tuple[float, float]:
        """Delay CV inputs expect bipolar offsets [-1, 1]."""
        _ = port_name
        return -1.0, 1.0

    def create_unmodulated_component(self):
        """Create simple Delay without modulation."""
        delay_time = self.time_knob.get_value()
        feedback = self.feedback_knob.get_value()
        mix = self.mix_knob.get_value()
        return Delay(delay_time=delay_time, feedback=feedback, mix=mix)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Apply delay during an engine-owned render cycle."""
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        # Thread-safe component access with lock
        with self._component_lock:
            if self.component is None:
                self.component = self.create_unmodulated_component()

            apply_control_rate_cv(
                self.component,
                parameters,
                (
                    ControlRateCvSpec(
                        "delay_time",
                        "delay_time",
                        self.time_knob.get_value,
                        self.time_cv_port,
                        0.001,
                        3.0,
                    ),
                    ControlRateCvSpec(
                        "feedback",
                        "feedback",
                        self.feedback_knob.get_value,
                        self.feedback_cv_port,
                        0.0,
                        1.0,
                    ),
                    ControlRateCvSpec(
                        "mix",
                        "mix",
                        self.mix_knob.get_value,
                        self.mix_cv_port,
                        0.0,
                        1.0,
                    ),
                ),
                num_samples,
            )

            self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
