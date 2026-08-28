"""Stereo width (mid/side) module."""

from __future__ import annotations

from PyQt6.QtGui import QColor

# pylint: disable-next=no-name-in-module
from soniclab.dsp.modifiers.spatial import StereoWidth  # type: ignore[import-not-found]

from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import as_stereo, float_parameter, read_samples
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class StereoWidthModule(ModuleWidget):
    """Widen or collapse stereo image via mid/side."""

    runtime_kind = "stereo_width"
    metadata = ModuleMetadata(
        title="Stereo Width",
        category=ModuleCategory.MODIFIER,
        description="Mid/side stereo width (0 = mono, 1 = natural, 2 = wide)",
    )

    def __init__(self) -> None:
        super().__init__(width=160, height=160, color=QColor(90, 130, 170))
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.component = StereoWidth(width=1.0)

        layout = self._begin_controls()
        self.width_knob = Knob(
            label="Width",
            description="0 = mono, 1 = original, 2 = extra wide",
            min_value=0.0,
            max_value=2.0,
            default_value=1.0,
        )
        self.bind_parameter_knob(self.width_knob, "width", register=True)
        layout.addWidget(self.width_knob)
        self._finish_controls(layout)

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if self._require_input_or_silence(num_samples):
            return
        self.component.width = float_parameter(
            parameters, "width", self.width_knob.get_value
        )
        stereo = as_stereo(read_samples(self.in_port, num_samples))
        self.out_port.write(self.component.process(stereo))
