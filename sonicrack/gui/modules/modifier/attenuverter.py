"""Attenuverter utility module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.modifiers import Attenuverter

from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class AttenuverterModule(ModuleWidget):
    """Scale and offset a CV or audio signal (``out = in * amount + offset``)."""

    runtime_kind = "attenuverter"
    metadata = ModuleMetadata(
        title="Attenuverter",
        category=ModuleCategory.MODIFIER,
        description="Scale (±) and offset CV or audio",
    )

    def __init__(self) -> None:
        super().__init__(width=200, height=150, color=QColor(100, 110, 120))
        self.in_port = self.add_input("In", signal=PortSignal.CONTROL_CV)
        self.out_port = self.add_output("Out", signal=PortSignal.CONTROL_CV)
        self.component = Attenuverter()

        layout = self._begin_controls()
        row = QHBoxLayout()
        self.amount_knob = Knob(
            label="Amount",
            description="Gain from -2 to +2 (negative inverts)",
            min_value=-2.0,
            max_value=2.0,
            default_value=1.0,
        )
        self.amount_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("amount", self.amount_knob.get_value())
        )
        row.addWidget(self.amount_knob)

        self.offset_knob = Knob(
            label="Offset",
            description="DC offset added after scaling",
            min_value=-1.0,
            max_value=1.0,
            default_value=0.0,
        )
        self.offset_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("offset", self.offset_knob.get_value())
        )
        row.addWidget(self.offset_knob)
        layout.addLayout(row)
        self._finish_controls(layout)

        self.register_parameter("amount", self.amount_knob)
        self.register_parameter("offset", self.offset_knob)

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return
        self.component.amount = float_parameter(
            parameters, "amount", self.amount_knob.get_value
        )
        self.component.offset = float_parameter(
            parameters, "offset", self.offset_knob.get_value
        )
        self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
