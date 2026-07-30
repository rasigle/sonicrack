"""Clock source module for synced sequencing."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel
from soniclab.sequencing import StepClock

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, str_parameter
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class ClockModule(ModuleWidget):
    """Musical clock pulse source."""

    runtime_kind = "clock"

    metadata = ModuleMetadata(
        title="Clock",
        category=ModuleCategory.SEQUENCER,
        description="BPM-synced pulse source for sequencers",
    )

    def __init__(self) -> None:
        super().__init__(width=220, height=205, color=QColor(150, 125, 70))

        self.clock_port = self.add_output("Clock")
        self.component = StepClock(sample_rate=audio_config.sample_rate)

        layout = self._begin_controls()

        timing_row = QHBoxLayout()
        timing_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.bpm_knob = Knob(
            label="BPM", min_value=30.0, max_value=300.0, default_value=120.0
        )
        self.bpm_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("bpm", self.bpm_knob.get_value())
        )
        timing_row.addWidget(self.bpm_knob)

        self.swing_knob = Knob(
            label="Swing", min_value=0.0, max_value=0.75, default_value=0.0
        )
        self.swing_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("swing", self.swing_knob.get_value())
        )
        timing_row.addWidget(self.swing_knob)
        layout.addLayout(timing_row)

        division_layout = QHBoxLayout()
        division_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        division_layout.addWidget(QLabel("Division:"))
        self.division_combo = QComboBox()
        self.division_combo.addItems(["1/4", "1/8", "1/16", "1/32"])
        self.division_combo.setCurrentText("1/16")
        self.division_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("division", value)
        )
        division_layout.addWidget(self.division_combo)
        self.run_checkbox = QCheckBox("Run")
        self.run_checkbox.setChecked(True)
        self.run_checkbox.toggled.connect(
            lambda value: self.parameter_changed.emit("running", value)
        )
        division_layout.addWidget(self.run_checkbox)
        layout.addLayout(division_layout)

        self._finish_controls(layout)

        self.register_parameter("bpm", self.bpm_knob)
        self.register_parameter(
            "division",
            self.division_combo,
            getter="currentText",
            setter="setCurrentText",
        )
        self.register_parameter("swing", self.swing_knob)
        self.register_parameter(
            "running",
            self.run_checkbox,
            getter="isChecked",
            setter="setChecked",
        )

        self._install_sample_rate_listener()

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component.sample_rate = new_sample_rate
        self.component.reset()

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        self.component.bpm = float_parameter(parameters, "bpm", self.bpm_knob.get_value)
        self.component.division = str_parameter(
            parameters, "division", self.division_combo.currentText
        )
        self.component.swing = float_parameter(
            parameters, "swing", self.swing_knob.get_value
        )
        running = bool(parameters.get("running", self.run_checkbox.isChecked()))
        self.clock_port.write(self.component.process(num_samples, running=running))
