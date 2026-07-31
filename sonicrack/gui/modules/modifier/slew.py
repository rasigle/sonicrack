"""Slew limiter / general glide module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.modifiers import SlewLimiter

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class SlewModule(ModuleWidget):
    """Rate-limit CV changes (portamento / glide on any signal)."""

    runtime_kind = "slew"
    metadata = ModuleMetadata(
        title="Slew",
        category=ModuleCategory.MODIFIER,
        description="Asymmetric slew / glide for pitch or control CV",
    )

    def __init__(self) -> None:
        super().__init__(width=200, height=150, color=QColor(100, 130, 110))
        self.in_port = self.add_input("In", signal=PortSignal.CONTROL_CV)
        self.out_port = self.add_output("Out", signal=PortSignal.CONTROL_CV)
        self.component = SlewLimiter(sample_rate=audio_config.sample_rate)

        layout = self._begin_controls()
        row = QHBoxLayout()
        self.rise_knob = Knob(
            label="Rise",
            description="Time for rising transitions (ms for full 0→1)",
            min_value=0.0,
            max_value=2000.0,
            default_value=50.0,
            logarithmic=True,
        )
        self.rise_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("rise_ms", self.rise_knob.get_value())
        )
        row.addWidget(self.rise_knob)

        self.fall_knob = Knob(
            label="Fall",
            description="Time for falling transitions (ms for full 1→0)",
            min_value=0.0,
            max_value=2000.0,
            default_value=50.0,
            logarithmic=True,
        )
        self.fall_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("fall_ms", self.fall_knob.get_value())
        )
        row.addWidget(self.fall_knob)
        layout.addLayout(row)
        self._finish_controls(layout)

        self.register_parameter("rise_ms", self.rise_knob)
        self.register_parameter("fall_ms", self.fall_knob)
        self._install_sample_rate_listener()

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component.sample_rate = float(new_sample_rate)
        self.component.reset()

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return
        self.component.rise_ms = float_parameter(
            parameters, "rise_ms", self.rise_knob.get_value
        )
        self.component.fall_ms = float_parameter(
            parameters, "fall_ms", self.fall_knob.get_value
        )
        self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
