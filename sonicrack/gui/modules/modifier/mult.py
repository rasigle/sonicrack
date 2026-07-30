"""Signal multiplier / ring-mod utility module."""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QColor

from sonicrack.dsp.utilities import SignalMult
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class MultModule(ModuleWidget):
    """Multiply two signals (ring modulation / CV product)."""

    runtime_kind = "mult"
    metadata = ModuleMetadata(
        title="Mult",
        category=ModuleCategory.MODIFIER,
        description="Multiply two signals (ring mod / CV product)",
    )

    def __init__(self) -> None:
        super().__init__(width=200, height=200, color=QColor(110, 100, 90))
        self.a_port = self.add_input("A", signal=PortSignal.AUDIO)
        self.b_port = self.add_input("B", signal=PortSignal.CONTROL_CV)
        self.out_port = self.add_output("Out", signal=PortSignal.AUDIO)
        self.component = SignalMult()

        layout = self._begin_controls()
        self.amount_knob = Knob(
            label="Amount",
            description="Output scale",
            min_value=0.0,
            max_value=2.0,
            default_value=1.0,
        )
        self.amount_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("amount", self.amount_knob.get_value())
        )
        layout.addWidget(self.amount_knob)
        self._finish_controls(layout)

        self.register_parameter("amount", self.amount_knob)

    def get_required_inputs(self) -> list[str]:
        return ["A"]

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if not self.a_port.is_connected:
            self.out_port.write(silence(num_samples))
            return
        a = read_samples(self.a_port, num_samples)
        b = (
            read_samples(self.b_port, num_samples)
            if self.b_port.is_connected
            else np.ones(num_samples, dtype=np.float32)  # unity when B disconnected
        )
        self.component.amount = float_parameter(
            parameters, "amount", self.amount_knob.get_value
        )
        self.out_port.write(self.component(a, b))
