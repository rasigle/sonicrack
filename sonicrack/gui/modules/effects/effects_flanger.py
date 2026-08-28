"""Flanger effect module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.effects import Flanger

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class FlangerModule(ModuleWidget):
    """Short modulated delay with feedback (classic comb-notch sweep)."""

    runtime_kind = "flanger"
    metadata = ModuleMetadata(
        title="Flanger",
        category=ModuleCategory.EFFECT,
        description="Short modulated delay flanger with feedback",
    )

    def __init__(self) -> None:
        super().__init__(width=220, height=260, color=QColor(80, 150, 190))
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.component = Flanger(sample_rate=audio_config.sample_rate)

        layout = self._begin_controls()
        row1 = QHBoxLayout()
        self.rate_knob = Knob(
            label="Rate",
            description="LFO rate in Hz",
            min_value=0.05,
            max_value=8.0,
            default_value=0.25,
            logarithmic=True,
        )
        self.rate_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("rate", self.rate_knob.get_value())
        )
        row1.addWidget(self.rate_knob)

        self.depth_knob = Knob(
            label="Depth",
            description="Modulation depth in milliseconds",
            min_value=0.0,
            max_value=5.0,
            default_value=1.5,
        )
        self.depth_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("depth", self.depth_knob.get_value())
        )
        row1.addWidget(self.depth_knob)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.delay_knob = Knob(
            label="Delay",
            description="Base delay time in milliseconds",
            min_value=0.1,
            max_value=10.0,
            default_value=2.0,
        )
        self.delay_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("delay", self.delay_knob.get_value())
        )
        row2.addWidget(self.delay_knob)

        self.feedback_knob = Knob(
            label="Feedback",
            description="Delay feedback (negative inverts the comb)",
            min_value=-0.95,
            max_value=0.95,
            default_value=0.7,
        )
        self.feedback_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "feedback", self.feedback_knob.get_value()
            )
        )
        row2.addWidget(self.feedback_knob)
        layout.addLayout(row2)

        self.mix_knob = Knob(
            label="Mix",
            description="Dry/wet mix",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self.mix_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("mix", self.mix_knob.get_value())
        )
        layout.addWidget(self.mix_knob)
        self._finish_controls(layout)

        self.register_parameter("rate", self.rate_knob)
        self.register_parameter("depth", self.depth_knob)
        self.register_parameter("delay", self.delay_knob)
        self.register_parameter("feedback", self.feedback_knob)
        self.register_parameter("mix", self.mix_knob)
        self._install_sample_rate_listener()

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component = Flanger(
            rate_hz=self.rate_knob.get_value(),
            depth_ms=self.depth_knob.get_value(),
            delay_ms=self.delay_knob.get_value(),
            mix=self.mix_knob.get_value(),
            feedback=self.feedback_knob.get_value(),
            sample_rate=new_sample_rate,
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if self._require_input_or_silence(num_samples):
            return
        self.component.rate_hz = float_parameter(
            parameters, "rate", self.rate_knob.get_value
        )
        self.component.depth_ms = float_parameter(
            parameters, "depth", self.depth_knob.get_value
        )
        self.component.delay_ms = float_parameter(
            parameters, "delay", self.delay_knob.get_value
        )
        self.component.feedback = float_parameter(
            parameters, "feedback", self.feedback_knob.get_value
        )
        self.component.mix = float_parameter(parameters, "mix", self.mix_knob.get_value)
        self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
