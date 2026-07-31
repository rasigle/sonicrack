"""Pitch CV quantizer module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel
from soniclab.dsp.modifiers import Quantizer

from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import read_samples, silence, str_parameter
from sonicrack.runtime.specs import RuntimeParameters

_ROOT_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@register_module()
class QuantizerModule(ModuleWidget):
    """Quantize 1V/oct pitch CV to a musical scale."""

    runtime_kind = "quantizer"
    metadata = ModuleMetadata(
        title="Quantizer",
        category=ModuleCategory.MODIFIER,
        description="Quantize 1V/oct CV to chromatic or scale degrees",
    )

    def __init__(self) -> None:
        super().__init__(width=220, height=145, color=QColor(90, 120, 140))
        self.in_port = self.add_input("In", signal=PortSignal.PITCH_CV)
        self.out_port = self.add_output("Out", signal=PortSignal.PITCH_CV)
        self.component = Quantizer()

        layout = self._begin_controls()

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(list(Quantizer.SCALES.keys()))
        self.scale_combo.setToolTip("Target scale for pitch quantization")
        self.scale_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("scale", value)
        )
        scale_row.addWidget(self.scale_combo)
        layout.addLayout(scale_row)

        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("Root:"))
        self.root_combo = QComboBox()
        self.root_combo.addItems(_ROOT_NAMES)
        self.root_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("root", value)
        )
        root_row.addWidget(self.root_combo)
        layout.addLayout(root_row)

        self._finish_controls(layout)

        self.register_parameter(
            "scale", self.scale_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter(
            "root", self.root_combo, getter="currentText", setter="setCurrentText"
        )

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        scale = str_parameter(parameters, "scale", self.scale_combo.currentText)
        root_name = str_parameter(parameters, "root", self.root_combo.currentText)
        root = _ROOT_NAMES.index(root_name) if root_name in _ROOT_NAMES else 0

        self.component.scale = scale
        self.component.root = root
        self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
