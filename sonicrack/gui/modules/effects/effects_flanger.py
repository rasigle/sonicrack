"""Flanger effect module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.effects import Flanger

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.effects._cv_modulation import ControlRateCvSpec
from sonicrack.gui.modules.effects._simple_effect import SimpleEffectModule
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module


@register_module()
class FlangerModule(SimpleEffectModule):
    """Short modulated delay with feedback (classic comb-notch sweep)."""

    runtime_kind = "flanger"
    metadata = ModuleMetadata(
        title="Flanger",
        category=ModuleCategory.EFFECT,
        description="Short modulated delay flanger with feedback",
    )

    def __init__(self) -> None:
        super().__init__(width=220, height=260, color=QColor(80, 150, 190))
        self._setup_effect_io(Flanger(sample_rate=audio_config.sample_rate))

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
        self.bind_parameter_knob(self.rate_knob, "rate", register=True)
        row1.addWidget(self.rate_knob)

        self.depth_knob = Knob(
            label="Depth",
            description="Modulation depth in milliseconds",
            min_value=0.0,
            max_value=5.0,
            default_value=1.5,
        )
        self.bind_parameter_knob(self.depth_knob, "depth", register=True)
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
        self.bind_parameter_knob(self.delay_knob, "delay", register=True)
        row2.addWidget(self.delay_knob)

        self.feedback_knob = Knob(
            label="Feedback",
            description="Delay feedback (negative inverts the comb)",
            min_value=-0.95,
            max_value=0.95,
            default_value=0.7,
        )
        self.bind_parameter_knob(self.feedback_knob, "feedback", register=True)
        row2.addWidget(self.feedback_knob)
        layout.addLayout(row2)

        self.mix_knob = Knob(
            label="Mix",
            description="Dry/wet mix",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self.bind_parameter_knob(self.mix_knob, "mix", register=True)
        layout.addWidget(self.mix_knob)
        self._finish_controls(layout)
        self._install_sample_rate_listener()

    def control_rate_specs(self) -> tuple[ControlRateCvSpec, ...]:
        return (
            ControlRateCvSpec(
                "rate_hz",
                "rate",
                self.rate_knob.get_value,
                None,
                self.rate_knob.min_value,
                self.rate_knob.max_value,
            ),
            ControlRateCvSpec(
                "depth_ms",
                "depth",
                self.depth_knob.get_value,
                None,
                self.depth_knob.min_value,
                self.depth_knob.max_value,
            ),
            ControlRateCvSpec(
                "delay_ms",
                "delay",
                self.delay_knob.get_value,
                None,
                self.delay_knob.min_value,
                self.delay_knob.max_value,
            ),
            ControlRateCvSpec(
                "feedback",
                "feedback",
                self.feedback_knob.get_value,
                None,
                self.feedback_knob.min_value,
                self.feedback_knob.max_value,
            ),
            ControlRateCvSpec(
                "mix",
                "mix",
                self.mix_knob.get_value,
                None,
                self.mix_knob.min_value,
                self.mix_knob.max_value,
            ),
        )

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component = Flanger(
            rate_hz=self.rate_knob.get_value(),
            depth_ms=self.depth_knob.get_value(),
            delay_ms=self.delay_knob.get_value(),
            mix=self.mix_knob.get_value(),
            feedback=self.feedback_knob.get_value(),
            sample_rate=new_sample_rate,
        )
