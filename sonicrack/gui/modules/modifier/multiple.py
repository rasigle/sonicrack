"""Passive mult / signal splitter module."""

from __future__ import annotations

from PyQt6.QtGui import QColor

from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class MultipleModule(ModuleWidget):
    """Fan one CV/audio signal out to four identical outputs."""

    runtime_kind = "multiple"
    metadata = ModuleMetadata(
        title="Multiple",
        category=ModuleCategory.MODIFIER,
        description="1→4 passive mult / signal splitter",
    )

    def __init__(self) -> None:
        # Port-only utility: height sized for 4 output jacks, not empty controls.
        super().__init__(width=140, height=160, color=QColor(95, 105, 115))
        self.in_port = self.add_input("In", signal=PortSignal.CONTROL_CV)
        self.out_ports = [
            self.add_output(f"Out {i + 1}", signal=PortSignal.CONTROL_CV)
            for i in range(4)
        ]

        layout = self._begin_controls()
        self._finish_controls(layout)

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        del parameters
        if not self.in_port.is_connected:
            silent = silence(num_samples)
            for port in self.out_ports:
                port.write(silent)
            return
        samples = read_samples(self.in_port, num_samples)
        for port in self.out_ports:
            # Each write gets its own buffer ownership path via Port.write.
            port.write(samples.copy())
