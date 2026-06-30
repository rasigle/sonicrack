"""Clock source module for synced sequencing."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel

from src.engine.sequencing import StepClock
from src.gui.audio_config import audio_config
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, str_parameter
from src.gui.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


@register_module()
class ClockModule(ModuleWidget):
    """Musical clock pulse source."""

    runtime_kind = "clock"

    metadata = ModuleMetadata(
        title="Clock",
        category=ModuleCategory.SOURCE,
        description="BPM-synced pulse source for sequencers",
    )

    def __init__(self) -> None:
        super().__init__(width=220, height=205, color=QColor(150, 125, 70))

        self.clock_port = self.add_output("Clock")
        self.component = StepClock(sample_rate=audio_config.sample_rate)

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        timing_row = QHBoxLayout()
        timing_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.bpm_knob = Knob("BPM", 30.0, 300.0, 120.0)
        self.bpm_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("bpm", self.bpm_knob.get_value())
        )
        timing_row.addWidget(self.bpm_knob)

        self.swing_knob = Knob("Swing", 0.0, 0.75, 0.0)
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

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

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

        self._sample_rate_listener = self._on_global_sample_rate_changed
        audio_config.add_sample_rate_listener(self._sample_rate_listener)
        self.destroyed.connect(self._cleanup_audio_config_listeners)

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component.sample_rate = new_sample_rate
        self.component.reset()

    def _cleanup_audio_config_listeners(self, *_args: object) -> None:
        audio_config.remove_sample_rate_listener(self._sample_rate_listener)

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
