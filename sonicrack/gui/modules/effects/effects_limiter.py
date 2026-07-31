"""Peak limiter effect module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from sonicrack.config.audio_config import audio_config
from soniclab.dsp.effects import Limiter
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class LimiterModule(ModuleWidget):
    """Peak limiter for finishing loud patches without digital trash."""

    runtime_kind = "limiter"
    metadata = ModuleMetadata(
        title="Limiter",
        category=ModuleCategory.EFFECT,
        description="Peak limiter with threshold, release, and makeup gain",
    )

    def __init__(self) -> None:
        super().__init__(width=220, height=235, color=QColor(160, 90, 90))
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.component = Limiter(sample_rate=audio_config.sample_rate)

        layout = self._begin_controls()
        row = QHBoxLayout()
        self.threshold_knob = Knob(
            label="Thresh",
            description="Peak threshold (linear)",
            min_value=0.1,
            max_value=1.0,
            default_value=0.9,
        )
        self.threshold_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "threshold", self.threshold_knob.get_value()
            )
        )
        row.addWidget(self.threshold_knob)

        self.release_knob = Knob(
            label="Release",
            description="Release time in milliseconds",
            min_value=5.0,
            max_value=500.0,
            default_value=50.0,
            logarithmic=True,
        )
        self.release_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "release", self.release_knob.get_value()
            )
        )
        row.addWidget(self.release_knob)
        layout.addLayout(row)

        self.makeup_knob = Knob(
            label="Makeup",
            description="Output makeup gain",
            min_value=0.5,
            max_value=4.0,
            default_value=1.0,
        )
        self.makeup_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("makeup", self.makeup_knob.get_value())
        )
        layout.addWidget(self.makeup_knob)
        self._finish_controls(layout)

        self.register_parameter("threshold", self.threshold_knob)
        self.register_parameter("release", self.release_knob)
        self.register_parameter("makeup", self.makeup_knob)
        self._install_sample_rate_listener()

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component = Limiter(
            threshold=self.threshold_knob.get_value(),
            release_ms=self.release_knob.get_value(),
            makeup=self.makeup_knob.get_value(),
            sample_rate=new_sample_rate,
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if self._require_input_or_silence(num_samples):
            return
        self.component.threshold = float_parameter(
            parameters, "threshold", self.threshold_knob.get_value
        )
        self.component.release_ms = float_parameter(
            parameters, "release", self.release_knob.get_value
        )
        self.component.makeup = float_parameter(
            parameters, "makeup", self.makeup_knob.get_value
        )
        self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
