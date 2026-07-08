"""Triggered decay envelope module."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QPushButton
from soniclab.dsp.modulators import DecayEnvelope

from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.port import PortSignal
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, read_samples
from src.gui.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


@register_module()
class DecayEnvelopeModule(ModuleWidget):
    """Attack-decay envelope with gate and accent inputs."""

    runtime_kind = "decay_envelope"

    metadata = ModuleMetadata(
        title="Decay Envelope",
        category=ModuleCategory.MODULATED_SOURCE,
        description="Triggered attack-decay envelope for plucks and filter CV",
    )

    def __init__(self) -> None:
        super().__init__(width=220, height=265, color=QColor(140, 175, 75))

        self.gate_input = self.add_input("Gate", signal=PortSignal.GATE)
        self.accent_input = self.add_input("Accent", signal=PortSignal.CONTROL_CV)
        self.out_port = self.add_output("Out", signal=PortSignal.CONTROL_CV)

        self.component = DecayEnvelope()
        self._previous_gate = 0.0

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout(spacing=6)

        timing_row = QHBoxLayout()
        timing_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.attack_knob = Knob(
            label="Attack",
            description="Sets the attack time of the envelope",
            min_value=0.001,
            max_value=0.2,
            default_value=0.001,
        )
        self.attack_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "attack_duration", self.attack_knob.get_value()
            )
        )
        timing_row.addWidget(self.attack_knob)

        self.decay_knob = Knob(
            label="Decay",
            description="Sets the decay time of the envelope",
            min_value=0.001,
            max_value=2.0,
            default_value=0.001,
        )
        self.decay_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "decay_duration", self.decay_knob.get_value()
            )
        )
        timing_row.addWidget(self.decay_knob)
        layout.addLayout(timing_row)

        amount_row = QHBoxLayout()
        amount_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.amount_knob = Knob(
            label="Amount",
            description="Scales the overall output of the envelope",
            min_value=0.0,
            max_value=1.0,
            default_value=0.001,
        )
        self.amount_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("amount", self.amount_knob.get_value())
        )
        amount_row.addWidget(self.amount_knob)

        self.accent_knob = Knob(
            label="Accent",
            description="Scales the accent input's effect on the envelope",
            min_value=0.0,
            max_value=1.0,
            default_value=0.001,
        )
        self.accent_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "accent_amount", self.accent_knob.get_value()
            )
        )
        amount_row.addWidget(self.accent_knob)
        layout.addLayout(amount_row)

        self.trigger_button = QPushButton("trig")
        self.trigger_button.clicked.connect(self._trigger)
        layout.addWidget(self.trigger_button)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.register_parameter("attack_duration", self.attack_knob)
        self.register_parameter("decay_duration", self.decay_knob)
        self.register_parameter("amount", self.amount_knob)
        self.register_parameter("accent_amount", self.accent_knob)

    def get_required_inputs(self) -> list[str]:
        return []

    def get_cv_output_range(self) -> tuple[float, float]:
        return 0.0, 1.0

    def _trigger(self) -> None:
        self.component.trigger_note_on()

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        self.component.attack_duration = float_parameter(
            parameters, "attack_duration", self.attack_knob.get_value
        )
        self.component.decay_duration = float_parameter(
            parameters, "decay_duration", self.decay_knob.get_value
        )
        self.component.amount = float_parameter(
            parameters, "amount", self.amount_knob.get_value
        )

        gate_signal = (
            read_samples(self.gate_input, num_samples)
            if self.gate_input.is_connected
            else np.zeros(num_samples, dtype=np.float32)
        )
        samples = self._render_gate_triggered_decay(gate_signal, num_samples)
        if self.accent_input.is_connected:
            accent_signal = read_samples(self.accent_input, num_samples)
            accent_amount = float_parameter(
                parameters, "accent_amount", self.accent_knob.get_value
            )
            samples = np.clip(samples * (1.0 + accent_signal * accent_amount), 0.0, 1.0)

        self.out_port.write(samples)

    def _render_gate_triggered_decay(
        self, gate_signal: np.ndarray, num_samples: int
    ) -> np.ndarray:
        output = np.zeros(num_samples, dtype=np.float32)
        start = 0

        for index, value in enumerate(gate_signal[:num_samples]):
            current_gate = float(value)
            note_on = self._previous_gate < 0.3 and current_gate > 0.7
            if note_on:
                if index > 0:
                    output[start:index] = self.component.get_samples(index - start)
                self.component.trigger_note_on()
                start = index
            self._previous_gate = current_gate

        if start < num_samples:
            output[start:] = self.component.get_samples(num_samples - start)
        return output
