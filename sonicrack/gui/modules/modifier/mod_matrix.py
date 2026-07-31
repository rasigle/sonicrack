"""Simple modulation matrix: 2 sources × 4 destinations with amount + offset."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGridLayout, QLabel

from soniclab.dsp.modifiers import ModMatrix
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class ModMatrixModule(ModuleWidget):
    """Route two CV sources to four destinations with scale and offset.

    Default routing maps Src A → Dest 1 and Src B → Dest 2 at full amount.
    Use amount knobs to mix either source into any destination (mod matrix
    style without cable spaghetti).
    """

    runtime_kind = "mod_matrix"
    NUM_SOURCES = 2
    NUM_DESTS = 4

    metadata = ModuleMetadata(
        title="Mod Matrix",
        category=ModuleCategory.MODIFIER,
        description="2×4 modulation matrix with amount and offset per destination",
    )

    def __init__(self) -> None:
        super().__init__(width=320, height=405, color=QColor(85, 100, 125))
        self.src_ports = [
            self.add_input("Src A", signal=PortSignal.CONTROL_CV),
            self.add_input("Src B", signal=PortSignal.CONTROL_CV),
        ]
        self.dest_ports = [
            self.add_output(f"Dest {i + 1}", signal=PortSignal.CONTROL_CV)
            for i in range(self.NUM_DESTS)
        ]
        self.component = ModMatrix(
            num_sources=self.NUM_SOURCES, num_dests=self.NUM_DESTS
        )

        layout = self._begin_controls(spacing=4)
        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)
        grid.addWidget(QLabel(""), 0, 0)
        grid.addWidget(QLabel("A Amt"), 0, 1)
        grid.addWidget(QLabel("B Amt"), 0, 2)
        grid.addWidget(QLabel("Offset"), 0, 3)

        self.amount_a_knobs: list[Knob] = []
        self.amount_b_knobs: list[Knob] = []
        self.offset_knobs: list[Knob] = []

        for d in range(self.NUM_DESTS):
            grid.addWidget(QLabel(f"D{d + 1}"), d + 1, 0)
            default_a = 1.0 if d == 0 else 0.0
            default_b = 1.0 if d == 1 else 0.0
            amt_a = Knob(
                label="A",
                description=f"Src A amount into Dest {d + 1}",
                min_value=-2.0,
                max_value=2.0,
                default_value=default_a,
            )
            amt_b = Knob(
                label="B",
                description=f"Src B amount into Dest {d + 1}",
                min_value=-2.0,
                max_value=2.0,
                default_value=default_b,
            )
            off = Knob(
                label="Off",
                description=f"DC offset on Dest {d + 1}",
                min_value=-1.0,
                max_value=1.0,
                default_value=0.0,
            )
            for knob, name in (
                (amt_a, f"amt_a_{d}"),
                (amt_b, f"amt_b_{d}"),
                (off, f"offset_{d}"),
            ):
                knob.value_changed.connect(
                    lambda _=None, n=name, k=knob: self.parameter_changed.emit(
                        n, k.get_value()
                    )
                )
            grid.addWidget(amt_a, d + 1, 1)
            grid.addWidget(amt_b, d + 1, 2)
            grid.addWidget(off, d + 1, 3)
            self.amount_a_knobs.append(amt_a)
            self.amount_b_knobs.append(amt_b)
            self.offset_knobs.append(off)
            self.register_parameter(f"amt_a_{d}", amt_a)
            self.register_parameter(f"amt_b_{d}", amt_b)
            self.register_parameter(f"offset_{d}", off)

        layout.addLayout(grid)
        self._finish_controls(layout)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        any_src = any(p.is_connected for p in self.src_ports)
        if not any_src:
            silent = silence(num_samples)
            for port in self.dest_ports:
                port.write(silent)
            return

        sources = [
            read_samples(port, num_samples) if port.is_connected else None
            for port in self.src_ports
        ]
        for d in range(self.NUM_DESTS):
            self.component.amounts[d, 0] = float_parameter(
                parameters, f"amt_a_{d}", self.amount_a_knobs[d].get_value
            )
            self.component.amounts[d, 1] = float_parameter(
                parameters, f"amt_b_{d}", self.amount_b_knobs[d].get_value
            )
            self.component.offsets[d] = float_parameter(
                parameters, f"offset_{d}", self.offset_knobs[d].get_value
            )

        outs = self.component.process(sources, num_samples)
        for port, samples in zip(self.dest_ports, outs, strict=True):
            port.write(samples)
