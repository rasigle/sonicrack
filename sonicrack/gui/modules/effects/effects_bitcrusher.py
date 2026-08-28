"""Bitcrusher effect module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.effects import Bitcrusher

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.effects._cv_modulation import ControlRateCvSpec
from sonicrack.gui.modules.effects._simple_effect import SimpleEffectModule
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module


@register_module()
class BitcrusherModule(SimpleEffectModule):
    """Bit-depth and sample-rate reduction for lo-fi grit."""

    runtime_kind = "bitcrusher"
    metadata = ModuleMetadata(
        title="Bitcrusher",
        category=ModuleCategory.EFFECT,
        description="Bit-depth and downsample crush",
    )

    def __init__(self) -> None:
        super().__init__(width=200, height=220, color=QColor(160, 90, 70))
        self._setup_effect_io(Bitcrusher(sample_rate=audio_config.sample_rate))

        layout = self._begin_controls()
        row = QHBoxLayout()
        self.bits_knob = Knob(
            label="Bits",
            description="Quantization depth (lower = more crush)",
            min_value=1.0,
            max_value=16.0,
            default_value=8.0,
        )
        self.bind_parameter_knob(self.bits_knob, "bits", register=True)
        row.addWidget(self.bits_knob)

        self.rate_knob = Knob(
            label="Rate",
            description="Hold every Nth sample (1 = full rate)",
            min_value=1.0,
            max_value=32.0,
            default_value=1.0,
        )
        self.bind_parameter_knob(self.rate_knob, "downsample", register=True)
        row.addWidget(self.rate_knob)
        layout.addLayout(row)

        self.mix_knob = Knob(
            label="Mix",
            description="Dry/wet mix",
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
        )
        self.bind_parameter_knob(self.mix_knob, "mix", register=True)
        layout.addWidget(self.mix_knob)
        self._finish_controls(layout)
        self._install_sample_rate_listener()

    def control_rate_specs(self) -> tuple[ControlRateCvSpec, ...]:
        return (
            ControlRateCvSpec(
                "bit_depth",
                "bits",
                self.bits_knob.get_value,
                None,
                self.bits_knob.min_value,
                self.bits_knob.max_value,
            ),
            ControlRateCvSpec(
                "downsample",
                "downsample",
                self.rate_knob.get_value,
                None,
                self.rate_knob.min_value,
                self.rate_knob.max_value,
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
        self.component = Bitcrusher(
            bit_depth=self.bits_knob.get_value(),
            downsample=self.rate_knob.get_value(),
            mix=self.mix_knob.get_value(),
            sample_rate=new_sample_rate,
        )
