"""Behringer 182-style dual CV row sequencer module."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)

from src.engine.sequencing import Behringer182Sequencer
from src.gui.audio_config import audio_config
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, read_samples, str_parameter
from src.gui.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


@register_module()
class Behringer182Module(ModuleWidget):
    """Eight-step 182-style analog sequencer with two CV rows and gate outs."""

    runtime_kind = "behringer_182"

    metadata = ModuleMetadata(
        title="Behringer 182",
        category=ModuleCategory.SOURCE,
        description="Eight-step dual CV row sequencer with gate, trigger, and end",
    )

    def __init__(self) -> None:
        super().__init__(width=300, height=760, color=QColor(115, 135, 80))

        self.clock_input = self.add_input("Clock")
        self.reset_input = self.add_input("Reset")
        self.hold_input = self.add_input("Hold")
        self.cv_a_port = self.add_output("CV A")
        self.cv_b_port = self.add_output("CV B")
        self.gate_port = self.add_output("Gate")
        self.trigger_port = self.add_output("Trig")
        self.end_port = self.add_output("End")

        self.component = Behringer182Sequencer(sample_rate=audio_config.sample_rate)
        self._previous_structure_key: tuple[int, str] | None = None
        self._previous_running = True

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout(spacing=6)

        cv_grid = QGridLayout()
        cv_grid.setHorizontalSpacing(4)
        cv_grid.setVerticalSpacing(2)
        cv_grid.addWidget(QLabel("Step"), 0, 0, Qt.AlignmentFlag.AlignCenter)
        cv_grid.addWidget(QLabel("CV A"), 0, 1, Qt.AlignmentFlag.AlignCenter)
        cv_grid.addWidget(QLabel("CV B"), 0, 2, Qt.AlignmentFlag.AlignCenter)

        self.cv_a_knobs: list[Knob] = []
        self.cv_b_knobs: list[Knob] = []
        default_cv_a = [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]
        default_cv_b = [1.0, 0.75, 0.5, 0.25, 0.0, 0.25, 0.5, 0.75]
        for step in range(8):
            step_label = QLabel(str(step + 1))
            step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cv_grid.addWidget(step_label, step + 1, 0)

            a_knob = Knob(
                label=f"A{step + 1}",
                description=f"Step {step + 1} CV A value before range scaling",
                min_value=0.0,
                max_value=1.0,
                default_value=default_cv_a[step],
            )
            a_name = f"cv_a_{step + 1}"
            a_knob.value_changed.connect(
                lambda _value, name=a_name, knob=a_knob: self.parameter_changed.emit(
                    name, knob.get_value()
                )
            )
            self.cv_a_knobs.append(a_knob)
            cv_grid.addWidget(a_knob, step + 1, 1)

            b_knob = Knob(
                label=f"B{step + 1}",
                description=f"Step {step + 1} CV B value before range scaling",
                min_value=0.0,
                max_value=1.0,
                default_value=default_cv_b[step],
            )
            b_name = f"cv_b_{step + 1}"
            b_knob.value_changed.connect(
                lambda _value, name=b_name, knob=b_knob: self.parameter_changed.emit(
                    name, knob.get_value()
                )
            )
            self.cv_b_knobs.append(b_knob)
            cv_grid.addWidget(b_knob, step + 1, 2)

        layout.addLayout(cv_grid)

        self.gates_edit = QLineEdit("1,1,1,1,1,1,1,1")
        self.gates_edit.setToolTip("Eight gate flags. Use 1/0 or x/- values.")
        self.gates_edit.textChanged.connect(
            lambda value: self.parameter_changed.emit("gates", value)
        )
        layout.addWidget(QLabel("Gates:"))
        layout.addWidget(self.gates_edit)

        self.run_button = QPushButton("Stop")
        self.run_button.setCheckable(True)
        self.run_button.setChecked(True)
        self.run_button.setToolTip("Start or stop sequencer clock advancement.")
        self.run_button.toggled.connect(self._on_running_changed)
        layout.addWidget(self.run_button)

        top_row = QHBoxLayout()
        top_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bpm_knob = Knob(
            label="BPM", min_value=20.0, max_value=300.0, default_value=120.0
        )
        self.bpm_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("bpm", self.bpm_knob.get_value())
        )
        top_row.addWidget(self.bpm_knob)

        self.gate_length_knob = Knob(
            label="Gate", min_value=0.0, max_value=1.0, default_value=0.5
        )
        self.gate_length_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "gate_length", self.gate_length_knob.get_value()
            )
        )
        top_row.addWidget(self.gate_length_knob)

        self.range_a_knob = Knob(
            label="A Range", min_value=0.0, max_value=10.0, default_value=5.0
        )
        self.range_a_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "cv_a_range", self.range_a_knob.get_value()
            )
        )
        top_row.addWidget(self.range_a_knob)
        layout.addLayout(top_row)

        range_row = QHBoxLayout()
        range_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.range_b_knob = Knob(
            label="B Range", min_value=0.0, max_value=10.0, default_value=5.0
        )
        self.range_b_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "cv_b_range", self.range_b_knob.get_value()
            )
        )
        range_row.addWidget(self.range_b_knob)
        layout.addLayout(range_row)

        combo_row = QHBoxLayout()
        combo_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        combo_row.addWidget(QLabel("Steps:"))
        self.steps_combo = QComboBox()
        self.steps_combo.addItems([str(value) for value in range(1, 9)])
        self.steps_combo.setCurrentText("8")
        self.steps_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("steps", value)
        )
        combo_row.addWidget(self.steps_combo)

        combo_row.addWidget(QLabel("Div:"))
        self.division_combo = QComboBox()
        self.division_combo.addItems(["1/4", "1/8", "1/16", "1/32"])
        self.division_combo.setCurrentText("1/16")
        self.division_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("division", value)
        )
        combo_row.addWidget(self.division_combo)

        combo_row.addWidget(QLabel("Dir:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["forward", "reverse", "pendulum", "random"])
        self.direction_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("direction", value)
        )
        combo_row.addWidget(self.direction_combo)
        layout.addLayout(combo_row)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        for step, knob in enumerate(self.cv_a_knobs, start=1):
            self.register_parameter(f"cv_a_{step}", knob)
        for step, knob in enumerate(self.cv_b_knobs, start=1):
            self.register_parameter(f"cv_b_{step}", knob)
        self.register_parameter(
            "gates", self.gates_edit, getter="text", setter="setText"
        )
        self.register_parameter(
            "running", self.run_button, getter="isChecked", setter="setChecked"
        )
        self.register_parameter("bpm", self.bpm_knob)
        self.register_parameter("gate_length", self.gate_length_knob)
        self.register_parameter("cv_a_range", self.range_a_knob)
        self.register_parameter("cv_b_range", self.range_b_knob)
        self.register_parameter(
            "steps", self.steps_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter(
            "division",
            self.division_combo,
            getter="currentText",
            setter="setCurrentText",
        )
        self.register_parameter(
            "direction",
            self.direction_combo,
            getter="currentText",
            setter="setCurrentText",
        )

        self._sample_rate_listener = self._on_global_sample_rate_changed
        audio_config.add_sample_rate_listener(self._sample_rate_listener)
        self.destroyed.connect(self._cleanup_audio_config_listeners)

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component.sample_rate = new_sample_rate
        self.component.clock.sample_rate = new_sample_rate
        self.component.reset()

    def _cleanup_audio_config_listeners(self, *_args: object) -> None:
        audio_config.remove_sample_rate_listener(self._sample_rate_listener)

    def _on_running_changed(self, running: bool) -> None:
        self.run_button.setText("Stop" if running else "Start")
        self.parameter_changed.emit("running", running)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        cv_a = self._cv_row_from_parameters(parameters, "cv_a", self.cv_a_knobs)
        cv_b = self._cv_row_from_parameters(parameters, "cv_b", self.cv_b_knobs)
        gates_text = str_parameter(parameters, "gates", self.gates_edit.text)
        steps = int(str_parameter(parameters, "steps", self.steps_combo.currentText))
        direction = str_parameter(
            parameters, "direction", self.direction_combo.currentText
        )
        cv_a_range = float_parameter(
            parameters, "cv_a_range", self.range_a_knob.get_value
        )
        cv_b_range = float_parameter(
            parameters, "cv_b_range", self.range_b_knob.get_value
        )
        running = bool(parameters.get("running", self.run_button.isChecked()))
        if running and not self._previous_running:
            self.component.reset()
        self._previous_running = running

        self.component.cv_a = cv_a
        self.component.cv_b = cv_b
        self.component.gates = self._parse_gates(gates_text)
        self.component.cv_a_range = cv_a_range
        self.component.cv_b_range = cv_b_range

        structure_key = (steps, direction)
        if structure_key != self._previous_structure_key:
            self.component.steps = int(np.clip(steps, 1, 8))
            self.component.direction = direction
            self.component.reset()
            self._previous_structure_key = structure_key

        self.component.gate_length = float_parameter(
            parameters, "gate_length", self.gate_length_knob.get_value
        )
        self.component.configure_clock(
            bpm=float_parameter(parameters, "bpm", self.bpm_knob.get_value),
            division=str_parameter(
                parameters, "division", self.division_combo.currentText
            ),
            swing=0.0,
        )

        clock_pulses = (
            read_samples(self.clock_input, num_samples)
            if self.clock_input.is_connected
            else None
        )
        reset_pulses = (
            read_samples(self.reset_input, num_samples)
            if self.reset_input.is_connected
            else None
        )
        hold_signal = (
            read_samples(self.hold_input, num_samples)
            if self.hold_input.is_connected
            else None
        )
        frame = self.component.process(
            num_samples,
            clock_pulses=clock_pulses,
            reset_pulses=reset_pulses,
            hold_signal=hold_signal,
            run_signal=(
                None
                if running
                else np.zeros(num_samples, dtype=np.float32)
            ),
        )
        self.cv_a_port.write(frame.cv_a)
        self.cv_b_port.write(frame.cv_b)
        if running:
            self.gate_port.write(frame.gate)
            self.trigger_port.write(frame.trigger)
            self.end_port.write(frame.end)
        else:
            stopped = np.zeros(num_samples, dtype=np.float32)
            self.gate_port.write(stopped)
            self.trigger_port.write(stopped)
            self.end_port.write(stopped)

    @staticmethod
    def _parse_cv_row(text: str) -> np.ndarray:
        values: list[float] = []
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(float(item))
            except ValueError:
                values.append(0.0)
        fitted = np.zeros(8, dtype=np.float32)
        if values:
            raw = np.asarray(values, dtype=np.float32)
            fitted[: min(8, len(raw))] = raw[:8]
        return fitted

    @staticmethod
    def _cv_row_from_parameters(
        parameters: RuntimeParameters, prefix: str, knobs: list[Knob]
    ) -> np.ndarray:
        if prefix in parameters:
            return Behringer182Module._parse_cv_row(str(parameters[prefix]))

        values = np.zeros(8, dtype=np.float32)
        for index, knob in enumerate(knobs):
            value = parameters.get(f"{prefix}_{index + 1}", knob.get_value())
            values[index] = float(value)
        return values

    @staticmethod
    def _parse_gates(text: str) -> np.ndarray:
        raw = [item.strip().lower() for item in text.split(",") if item.strip()]
        gates = np.ones(8, dtype=bool)
        for index, item in enumerate(raw[:8]):
            gates[index] = item in {"1", "true", "t", "yes", "y", "x", "on"}
        return gates
