"""Parametric EQ effect module."""

from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.effects import ParametricEQ

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.effects._cv_modulation import ControlRateCvSpec
from sonicrack.gui.modules.effects._simple_effect import SimpleEffectModule
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import str_parameter
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class EQModule(SimpleEffectModule):
    """Single-band parametric EQ (peak / low shelf / high shelf)."""

    runtime_kind = "eq"
    metadata = ModuleMetadata(
        title="EQ",
        category=ModuleCategory.EFFECT,
        description="Parametric EQ band for tone finishing",
    )

    def __init__(self) -> None:
        super().__init__(width=220, height=290, color=QColor(130, 130, 100))
        self._setup_effect_io(ParametricEQ(sample_rate=audio_config.sample_rate))

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
        self.bind_parameter_knob(self.freq_knob, "frequency", register=True)
        row1.addWidget(self.freq_knob)

        self.gain_knob = Knob(
            label="Gain",
            description="Boost/cut in dB",
            min_value=-18.0,
            max_value=18.0,
            default_value=0.0,
        )
        self.bind_parameter_knob(self.gain_knob, "gain_db", register=True)
        row1.addWidget(self.gain_knob)
        layout.addLayout(row1)

        self.q_knob = Knob(
            label="Q",
            description="Bandwidth / resonance",
            min_value=0.2,
            max_value=8.0,
            default_value=1.0,
        )
        self.bind_parameter_knob(self.q_knob, "q", register=True)
        layout.addWidget(self.q_knob)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["peak", "low_shelf", "high_shelf"])
        self.mode_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("mode", value)
        )
        layout.addWidget(QtWidgets.QLabel("Mode:"))
        layout.addWidget(self.mode_combo)
        self._finish_controls(layout)

        self.register_parameter(
            "mode", self.mode_combo, getter="currentText", setter="setCurrentText"
        )
        self._install_sample_rate_listener()

    def control_rate_specs(self) -> tuple[ControlRateCvSpec, ...]:
        return (
            ControlRateCvSpec(
                "frequency",
                "frequency",
                self.freq_knob.get_value,
                None,
                self.freq_knob.min_value,
                self.freq_knob.max_value,
            ),
            ControlRateCvSpec(
                "gain_db",
                "gain_db",
                self.gain_knob.get_value,
                None,
                self.gain_knob.min_value,
                self.gain_knob.max_value,
            ),
            ControlRateCvSpec(
                "q",
                "q",
                self.q_knob.get_value,
                None,
                self.q_knob.min_value,
                self.q_knob.max_value,
            ),
        )

    def apply_runtime_parameters(
        self, parameters: RuntimeParameters, num_samples: int
    ) -> None:
        super().apply_runtime_parameters(parameters, num_samples)
        self.component.mode = str_parameter(
            parameters, "mode", self.mode_combo.currentText
        )

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component = ParametricEQ(
            frequency=self.freq_knob.get_value(),
            gain_db=self.gain_knob.get_value(),
            q=self.q_knob.get_value(),
            mode=self.mode_combo.currentText(),
            sample_rate=new_sample_rate,
        )
