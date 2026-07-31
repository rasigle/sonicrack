"""Parametric EQ effect module."""

from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from sonicrack.config.audio_config import audio_config
from soniclab.dsp.effects import ParametricEQ
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples, str_parameter
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class EQModule(ModuleWidget):
    """Single-band parametric EQ (peak / low shelf / high shelf)."""

    runtime_kind = "eq"
    metadata = ModuleMetadata(
        title="EQ",
        category=ModuleCategory.EFFECT,
        description="Parametric EQ band for tone finishing",
    )

    def __init__(self) -> None:
        super().__init__(width=220, height=290, color=QColor(130, 130, 100))
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.component = ParametricEQ(sample_rate=audio_config.sample_rate)

        layout = self._begin_controls()
        row1 = QHBoxLayout()
        self.freq_knob = Knob(
            label="Freq",
            description="Center / shelf frequency",
            min_value=40.0,
            max_value=12000.0,
            default_value=1000.0,
            logarithmic=True,
        )
        self.freq_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("frequency", self.freq_knob.get_value())
        )
        row1.addWidget(self.freq_knob)

        self.gain_knob = Knob(
            label="Gain",
            description="Boost/cut in dB",
            min_value=-18.0,
            max_value=18.0,
            default_value=0.0,
        )
        self.gain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("gain_db", self.gain_knob.get_value())
        )
        row1.addWidget(self.gain_knob)
        layout.addLayout(row1)

        self.q_knob = Knob(
            label="Q",
            description="Bandwidth / resonance",
            min_value=0.2,
            max_value=8.0,
            default_value=1.0,
        )
        self.q_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("q", self.q_knob.get_value())
        )
        layout.addWidget(self.q_knob)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["peak", "low_shelf", "high_shelf"])
        self.mode_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("mode", value)
        )
        layout.addWidget(QtWidgets.QLabel("Mode:"))
        layout.addWidget(self.mode_combo)
        self._finish_controls(layout)

        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("gain_db", self.gain_knob)
        self.register_parameter("q", self.q_knob)
        self.register_parameter(
            "mode", self.mode_combo, getter="currentText", setter="setCurrentText"
        )
        self._install_sample_rate_listener()

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component = ParametricEQ(
            frequency=self.freq_knob.get_value(),
            gain_db=self.gain_knob.get_value(),
            q=self.q_knob.get_value(),
            mode=self.mode_combo.currentText(),
            sample_rate=new_sample_rate,
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if self._require_input_or_silence(num_samples):
            return
        self.component.frequency = float_parameter(
            parameters, "frequency", self.freq_knob.get_value
        )
        self.component.gain_db = float_parameter(
            parameters, "gain_db", self.gain_knob.get_value
        )
        self.component.q = float_parameter(parameters, "q", self.q_knob.get_value)
        self.component.mode = str_parameter(
            parameters, "mode", self.mode_combo.currentText
        )
        self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
