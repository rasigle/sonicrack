"""Sample-and-hold utility module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from soniclab.dsp.modifiers import SampleAndHold

from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class SampleHoldModule(ModuleWidget):
    """Sample input on clock rising edges; hold between triggers."""

    runtime_kind = "sample_hold"
    metadata = ModuleMetadata(
        title="Sample & Hold",
        category=ModuleCategory.MODIFIER,
        description="Clocked sample-and-hold for CV or audio",
    )

    def __init__(self) -> None:
        # Port-only utility: height sized for 3 jacks, not empty controls.
        super().__init__(width=140, height=140, color=QColor(120, 100, 130))
        self.in_port = self.add_input("In", signal=PortSignal.CONTROL_CV)
        self.clock_port = self.add_input("Clock", signal=PortSignal.TRIGGER)
        self.out_port = self.add_output("Out", signal=PortSignal.CONTROL_CV)
        self.component = SampleAndHold()

        layout = self._begin_controls()
        self._finish_controls(layout)

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        del parameters
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return
        clock = (
            read_samples(self.clock_port, num_samples)
            if self.clock_port.is_connected
            else None
        )
        self.out_port.write(
            self.component(read_samples(self.in_port, num_samples), clock)
        )
