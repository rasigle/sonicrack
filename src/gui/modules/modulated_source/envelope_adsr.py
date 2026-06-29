import logging
from typing import Any

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QPushButton

from src.engine.modulator import ADSREnvelope, GateTriggeredADSR
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter
from src.gui.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


@register_module()
class ADSRModule(ModuleWidget):
    """ADSR envelope module with optional gate input.

    Can be triggered by:
    - External gate signal (e.g., from MIDI Input)
    - Manual trigger button
    """

    runtime_kind = "adsr"

    metadata = ModuleMetadata(
        title="ADSR Envelope",
        category=ModuleCategory.MODULATED_SOURCE,  # Receives gate input
        description="ADSR envelope generator with gate input for MIDI triggering",
    )

    def __init__(self):
        """Initialize ADSR module."""
        super().__init__(
            width=220,
            height=275,
            color=QColor(120, 180, 80),
        )

        # Add input port for gate signal (optional)
        self.gate_input = self.add_input("Gate")

        # Add output port
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # ADSR controls
        knobs_layout = QHBoxLayout()

        self.attack_knob = Knob("Attack", 0.005, 5.0, 0.01)  # Min 5ms, default 10ms
        self.attack_knob.setToolTip(
            "Attack time (seconds)\nRange: 0.005-5.0s\nLower values may cause clicks"
        )
        self.attack_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "attack_duration", self.attack_knob.get_value()
            )
        )
        knobs_layout.addWidget(self.attack_knob)

        self.decay_knob = Knob("Decay", 0.001, 5.0, 0.2)
        self.decay_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "decay_duration", self.decay_knob.get_value()
            )
        )
        knobs_layout.addWidget(self.decay_knob)

        layout.addLayout(knobs_layout)

        knobs_layout2 = QHBoxLayout()

        self.sustain_knob = Knob("Sustain", 0.0, 1.0, 0.7)
        self.sustain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "sustain_level", self.sustain_knob.get_value()
            )
        )
        knobs_layout2.addWidget(self.sustain_knob)

        self.release_knob = Knob("Release", 0.001, 5.0, 0.3)
        self.release_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "release_duration", self.release_knob.get_value()
            )
        )
        knobs_layout2.addWidget(self.release_knob)

        layout.addLayout(knobs_layout2)

        # Manual trigger button
        trigger_layout = QHBoxLayout()
        self.trigger_button = QPushButton("Gate")
        self.trigger_button.setCheckable(True)
        self.trigger_button.setMinimumHeight(35)
        self.trigger_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: 2px solid #45a049;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:pressed {
                background-color: #45a049;
                border: 2px solid #3d8b40;
            }
            QPushButton:hover {
                background-color: #5cbf60;
            }
            QPushButton:checked {
                background-color: #2f8f46;
                border: 2px solid #8ee0a0;
            }
        """)
        self.trigger_button.setToolTip(
            "Manual Gate\n"
            "On: Start attack and hold sustain\n"
            "Off: Trigger release phase"
        )
        self.trigger_button.toggled.connect(self._on_trigger_toggled)
        trigger_layout.addWidget(self.trigger_button)
        layout.addLayout(trigger_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("attack_duration", self.attack_knob)
        self.register_parameter("decay_duration", self.decay_knob)
        self.register_parameter("sustain_level", self.sustain_knob)
        self.register_parameter("release_duration", self.release_knob)

        # Track ADSR component for manual triggering
        self._adsr_component: ADSREnvelope | GateTriggeredADSR | None = None
        self._previous_gate = 0.0

        self.component = self.create_engine_component()

    def _trigger_adsr(self, note_on: bool) -> None:
        """Trigger ADSR note on/off if available."""
        if self._adsr_component is None:
            return

        component = self._adsr_component
        adsr = component.adsr if isinstance(component, GateTriggeredADSR) else component

        if note_on:
            adsr.trigger_note_on()
            logging.debug("ADSR manually triggered (note on)")
        else:
            adsr.trigger_note_off()
            logging.debug("ADSR manually released (note off)")

    def _on_trigger_toggled(self, checked: bool) -> None:
        """Handle manual gate toggles."""
        self.trigger_button.setText("Gate On" if checked else "Gate")
        self._trigger_adsr(note_on=checked)

    def get_required_inputs(self) -> list[str]:
        """Gate input is optional - ADSR works without gate triggering."""
        return []  # No required inputs - Gate is optional

    def get_cv_output_range(self) -> tuple[float, float]:
        """ADSR envelope outputs unipolar signal [0, 1].

        Returns:
            (0.0, 1.0) - unipolar output range
        """
        return 0.0, 1.0

    # AudioModuleInterface implementation
    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the ADSR component.

        If a gate signal is connected, wraps the ADSR in a GateTriggeredADSR
        that automatically triggers on gate transitions.
        """
        # Create base ADSR envelope
        adsr = ADSREnvelope(
            attack_duration=self.attack_knob.get_value(),
            decay_duration=self.decay_knob.get_value(),
            sustain_level=self.sustain_knob.get_value(),
            release_duration=self.release_knob.get_value(),
        )

        # If gate input is connected, wrap with gate-triggered version
        if input_components and len(input_components) > 0:
            gate_source = input_components[0]
            self._adsr_component = GateTriggeredADSR(adsr, gate_source)
        else:
            # No gate input, return plain ADSR (can be manually triggered)
            self._adsr_component = adsr

        return self._adsr_component

    def _current_adsr(self) -> ADSREnvelope:
        """Return the active ADSR envelope, unwrapping legacy gate wrappers."""
        if self._adsr_component is None:
            self._adsr_component = self.create_engine_component()

        component = self._adsr_component
        if isinstance(component, GateTriggeredADSR):
            return component.adsr
        return component

    def _apply_runtime_parameters(
        self, adsr: ADSREnvelope, parameters: RuntimeParameters
    ) -> None:
        adsr.attack_duration = float_parameter(
            parameters, "attack_duration", self.attack_knob.get_value
        )
        adsr.decay_duration = float_parameter(
            parameters, "decay_duration", self.decay_knob.get_value
        )
        adsr.sustain_level = float_parameter(
            parameters, "sustain_level", self.sustain_knob.get_value
        )
        adsr.release_duration = float_parameter(
            parameters, "release_duration", self.release_knob.get_value
        )

    def _render_gate_triggered_adsr(
        self, adsr: ADSREnvelope, gate_signal: np.ndarray, num_samples: int
    ) -> np.ndarray:
        """Render ADSR output while applying gate transitions inside the buffer."""
        output = np.zeros(num_samples, dtype=np.float32)
        start = 0
        previous_gate = self._previous_gate

        for index, value in enumerate(gate_signal):
            current_gate = float(value)
            note_on = previous_gate < 0.3 and current_gate > 0.7
            note_off = previous_gate > 0.7 and current_gate < 0.3
            if not note_on and not note_off:
                previous_gate = current_gate
                continue

            if index > start:
                output[start:index] = adsr.get_samples(index - start)

            if note_on:
                adsr.trigger_note_on()
            else:
                adsr.trigger_note_off()

            start = index
            previous_gate = current_gate

        if start < num_samples:
            output[start:] = adsr.get_samples(num_samples - start)

        self._previous_gate = previous_gate
        return output

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Render the ADSR envelope for the current engine cycle."""
        adsr = self._current_adsr()
        self._apply_runtime_parameters(adsr, parameters)

        if self.gate_input.is_connected:
            gate_signal = np.asarray(
                self.gate_input.read(num_samples), dtype=np.float32
            ).reshape(-1)
            if len(gate_signal) < num_samples:
                gate_signal = np.pad(gate_signal, (0, num_samples - len(gate_signal)))
            elif len(gate_signal) > num_samples:
                gate_signal = gate_signal[:num_samples]
            samples = self._render_gate_triggered_adsr(
                adsr, gate_signal, num_samples
            )
        else:
            self._previous_gate = 0.0
            samples = adsr.get_samples(num_samples)

        self.out_port.write(samples)
