"""Bitcrusher effect module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.effects import Bitcrusher

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class BitcrusherModule(ModuleWidget):
    """Bit-depth and sample-rate reduction for lo-fi grit."""

    runtime_kind = "bitcrusher"
    metadata = ModuleMetadata(
        title="Bitcrusher",
        category=ModuleCategory.EFFECT,
        description="Bit-depth and downsample crush",
    )

    def __init__(self) -> None:
        super().__init__(width=200, height=220, color=QColor(160, 90, 70))
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.component = Bitcrusher(sample_rate=audio_config.sample_rate)

        layout = self._begin_controls()
        row = QHBoxLayout()
        self.bits_knob = Knob(
            label="Bits",
            description="Quantization depth (lower = more crush)",
            min_value=1.0,
            max_value=16.0,
            default_value=8.0,
        )
        self.bits_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("bits", self.bits_knob.get_value())
        )
        row.addWidget(self.bits_knob)

        self.rate_knob = Knob(
            label="Rate",
            description="Hold every Nth sample (1 = full rate)",
            min_value=1.0,
            max_value=32.0,
            default_value=1.0,
        )
        self.rate_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "downsample", self.rate_knob.get_value()
            )
        )
        row.addWidget(self.rate_knob)
        layout.addLayout(row)

        self.mix_knob = Knob(
            label="Mix",
            description="Dry/wet mix",
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
        )
        self.mix_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("mix", self.mix_knob.get_value())
        )
        layout.addWidget(self.mix_knob)
        self._finish_controls(layout)

        self.register_parameter("bits", self.bits_knob)
        self.register_parameter("downsample", self.rate_knob)
        self.register_parameter("mix", self.mix_knob)
        self._install_sample_rate_listener()

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component = Bitcrusher(
            bit_depth=self.bits_knob.get_value(),
            downsample=self.rate_knob.get_value(),
            mix=self.mix_knob.get_value(),
            sample_rate=new_sample_rate,
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if self._require_input_or_silence(num_samples):
            return
        self.component.bit_depth = float_parameter(
            parameters, "bits", self.bits_knob.get_value
        )
        self.component.downsample = float_parameter(
            parameters, "downsample", self.rate_knob.get_value
        )
        self.component.mix = float_parameter(parameters, "mix", self.mix_knob.get_value)
        self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
