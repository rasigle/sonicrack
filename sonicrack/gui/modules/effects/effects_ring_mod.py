"""Audio-rate ring modulator effect."""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.modifiers import SignalMult

from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import as_mono, float_parameter, read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class RingModModule(ModuleWidget):
    """Multiply two audio signals (ring modulation) with dry/wet mix."""

    runtime_kind = "ring_mod"
    metadata = ModuleMetadata(
        title="Ring Mod",
        category=ModuleCategory.EFFECT,
        description="Audio-rate ring modulator (In × Carrier)",
    )

    def __init__(self) -> None:
        super().__init__(width=200, height=200, color=QColor(150, 90, 140))
        self.in_port = self.add_input("In", signal=PortSignal.AUDIO)
        self.carrier_port = self.add_input("Carrier", signal=PortSignal.AUDIO)
        self.out_port = self.add_output("Out", signal=PortSignal.AUDIO)
        self.component = SignalMult()

        layout = self._begin_controls()
        row = QHBoxLayout()
        self.amount_knob = Knob(
            label="Amount",
            description="Product scale",
            min_value=0.0,
            max_value=2.0,
            default_value=1.0,
        )
        self.amount_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("amount", self.amount_knob.get_value())
        )
        row.addWidget(self.amount_knob)

        self.mix_knob = Knob(
            label="Mix",
            description="Dry (In) / wet (ring) mix",
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
        )
        self.mix_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("mix", self.mix_knob.get_value())
        )
        row.addWidget(self.mix_knob)
        layout.addLayout(row)
        self._finish_controls(layout)

        self.register_parameter("amount", self.amount_knob)
        self.register_parameter("mix", self.mix_knob)

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return
        dry = as_mono(read_samples(self.in_port, num_samples))
        if self.carrier_port.is_connected:
            carrier = as_mono(read_samples(self.carrier_port, num_samples))
        else:
            carrier = np.ones(num_samples, dtype=np.float32)
        self.component.amount = float_parameter(
            parameters, "amount", self.amount_knob.get_value
        )
        wet = np.asarray(self.component(dry, carrier), dtype=np.float32)
        mix = float_parameter(parameters, "mix", self.mix_knob.get_value)
        if mix <= 0.0:
            self.out_port.write(dry)
            return
        if mix >= 1.0:
            self.out_port.write(wet)
            return
        self.out_port.write(dry * (1.0 - mix) + wet * mix)
