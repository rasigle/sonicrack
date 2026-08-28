"""Chorus effect module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.effects import Chorus

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.effects._cv_modulation import ControlRateCvSpec
from sonicrack.gui.modules.effects._simple_effect import SimpleEffectModule
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module


@register_module()
class ChorusModule(SimpleEffectModule):
    """Stereo-style mono chorus with rate, depth, delay, and mix."""

    runtime_kind = "chorus"
    metadata = ModuleMetadata(
        title="Chorus",
        category=ModuleCategory.EFFECT,
        description="Modulated delay chorus for width and movement",
    )

    def __init__(self) -> None:
        super().__init__(width=220, height=260, color=QColor(100, 140, 180))
        self._setup_effect_io(Chorus(sample_rate=audio_config.sample_rate))

        layout = self._begin_controls()
        row1 = QHBoxLayout()
        self.rate_knob = Knob(
            label="Rate",
            description="LFO rate in Hz",
            min_value=0.05,
            max_value=5.0,
            default_value=0.8,
            logarithmic=True,
        )
        self.bind_parameter_knob(self.rate_knob, "rate", register=True)
        row1.addWidget(self.rate_knob)

        self.depth_knob = Knob(
            label="Depth",
            description="Modulation depth in milliseconds",
            min_value=0.1,
            max_value=10.0,
            default_value=3.5,
        )
        self.bind_parameter_knob(self.depth_knob, "depth", register=True)
        row1.addWidget(self.depth_knob)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.delay_knob = Knob(
            label="Delay",
            description="Base delay time in milliseconds",
            min_value=5.0,
            max_value=30.0,
            default_value=12.0,
        )
        self.bind_parameter_knob(self.delay_knob, "delay", register=True)
        row2.addWidget(self.delay_knob)

        self.mix_knob = Knob(
            label="Mix",
            description="Dry/wet mix",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self.bind_parameter_knob(self.mix_knob, "mix", register=True)
        row2.addWidget(self.mix_knob)
        layout.addLayout(row2)
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
                "mix",
                "mix",
                self.mix_knob.get_value,
                None,
                self.mix_knob.min_value,
                self.mix_knob.max_value,
            ),
        )

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component = Chorus(
            rate_hz=self.rate_knob.get_value(),
            depth_ms=self.depth_knob.get_value(),
            delay_ms=self.delay_knob.get_value(),
            mix=self.mix_knob.get_value(),
            sample_rate=new_sample_rate,
        )
