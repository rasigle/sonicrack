import logging
from typing import Any, Literal, cast

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from soniclab.dsp.modulators import ADSREnvelope, GateTriggeredADSR

from sonicrack.gui.core.module import ModuleCategory, ModuleMetadata
from sonicrack.gui.core.port import PortSignal
from sonicrack.gui.core.runtime import RuntimeParameters
from sonicrack.gui.core.runtime_helpers import float_parameter
from sonicrack.gui.module_registry import register_module
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget


@register_module()
class ADSRModule(ModuleWidget):
    """ADSR envelope module with optional gate input.

    Can be triggered by:
    - External gate signal (e.g., from MIDI Input)
    - Manual trigger button

    Retrigger mode defines note-on behavior while the envelope is still active:
    - Punch: short click-safe reset to zero, then attack from zero. This gives
      rhythmic gate patterns a stronger percussive chop.
    - Legato: attack starts from the current envelope level. This is smoother and
      avoids amplitude dips when gates overlap.
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
            height=325,
            color=QColor(120, 180, 80),
        )

        # Add input port for gate signal (optional)
        self.gate_input = self.add_input("Gate", signal=PortSignal.GATE)

        # Add output port
        self.out_port = self.add_output("Out", signal=PortSignal.CONTROL_CV)

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout(spacing=6)

        # ADSR controls
        knobs_layout = QHBoxLayout()
        knobs_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.attack_knob = Knob(
            label="Attack",
            description="Sets the attack time of the envelope",
            min_value=0.0,
            max_value=5.0,
            default_value=0.01,
        )
        self.attack_knob.setToolTip(
            "Attack time (seconds)\nRange: 0.005-5.0s\nLower values may cause clicks"
        )
        self.attack_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "attack_duration", self.attack_knob.get_value()
            )
        )
        knobs_layout.addWidget(self.attack_knob)

        self.decay_knob = Knob(
            label="Decay",
            description="Sets the decay time of the envelope",
            min_value=0.0,
            max_value=5.0,
            default_value=0.2,
        )
        self.decay_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "decay_duration", self.decay_knob.get_value()
            )
        )
        knobs_layout.addWidget(self.decay_knob)

        layout.addLayout(knobs_layout)

        knobs_layout2 = QHBoxLayout()
        knobs_layout2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.sustain_knob = Knob(
            label="Sustain",
            description="Sets the sustain level of the envelope",
            min_value=0.0,
            max_value=1.0,
            default_value=0.7,
        )
        self.sustain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "sustain_level", self.sustain_knob.get_value()
            )
        )
        knobs_layout2.addWidget(self.sustain_knob)

        self.release_knob = Knob(
            label="Release",
            description="Sets the release time of the envelope",
            min_value=0.0,
            max_value=5.0,
            default_value=0.3,
        )
        self.release_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "release_duration", self.release_knob.get_value()
            )
        )
        knobs_layout2.addWidget(self.release_knob)

        layout.addLayout(knobs_layout2)

        retrigger_layout = QVBoxLayout()
        retrigger_label = QLabel("Retrigger")
        self.retrigger_combo = QComboBox()
        self.retrigger_combo.addItems(["Punch", "Legato"])
        self.retrigger_combo.setToolTip(
            "Punch: briefly ramps to zero, then attacks for a stronger rhythmic "
            "chop without clicks.\n"
            "Legato: attacks from the current envelope level for smoother overlap."
        )
        self.retrigger_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("retrigger_mode", value)
        )
        retrigger_layout.addWidget(retrigger_label)
        retrigger_layout.addWidget(self.retrigger_combo)
        layout.addLayout(retrigger_layout)

        trigger_mode_layout = QVBoxLayout()
        trigger_mode_label = QLabel("Trig Mode")
        self.trigger_mode_combo = QComboBox()
        self.trigger_mode_combo.addItems(["Latched", "On/Off"])
        self.trigger_mode_combo.setToolTip(
            "Latched: click trig once to hold the envelope gate on, click again "
            "to release.\n"
            "On/Off: hold trig down to gate on, release it to gate off."
        )
        self.trigger_mode_combo.currentTextChanged.connect(
            self._on_trigger_mode_changed
        )
        trigger_mode_layout.addWidget(trigger_mode_label)
        trigger_mode_layout.addWidget(self.trigger_mode_combo)
        layout.addLayout(trigger_mode_layout)

        # Manual trigger button
        trigger_layout = QHBoxLayout()
        trigger_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trigger_button = QPushButton("trig")
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
            "Manual trig\n"
            "Latched: click to toggle note on/off\n"
            "On/Off: press to start attack, release for release phase"
        )
        self.trigger_button.toggled.connect(self._on_trigger_toggled)
        self.trigger_button.pressed.connect(self._on_trigger_pressed)
        self.trigger_button.released.connect(self._on_trigger_released)
        trigger_layout.addWidget(self.trigger_button)
        layout.addLayout(trigger_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("attack_duration", self.attack_knob)
        self.register_parameter("decay_duration", self.decay_knob)
        self.register_parameter("sustain_level", self.sustain_knob)
        self.register_parameter("release_duration", self.release_knob)
        self.register_parameter(
            "retrigger_mode",
            self.retrigger_combo,
            getter="currentText",
            setter="setCurrentText",
        )
        self.register_parameter(
            "trigger_mode",
            self.trigger_mode_combo,
            getter="currentText",
            setter="setCurrentText",
        )

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
        """Handle latched manual trig toggles."""
        if self._trigger_mode() != "latched":
            return

        self.trigger_button.setText("trig on" if checked else "trig")
        self._trigger_adsr(note_on=checked)

    def _on_trigger_pressed(self) -> None:
        """Start the manual gate in On/Off mode."""
        if self._trigger_mode() != "on/off":
            return

        self.trigger_button.setText("trig on")
        self._trigger_adsr(note_on=True)

    def _on_trigger_released(self) -> None:
        """Release the manual gate in On/Off mode."""
        if self._trigger_mode() != "on/off":
            return

        self.trigger_button.setText("trig")
        self._trigger_adsr(note_on=False)

    def _on_trigger_mode_changed(self, value: str) -> None:
        """Apply manual trig button behavior for the selected mode."""
        mode = self._normalize_trigger_mode(value)
        if mode == "latched":
            self.trigger_button.setCheckable(True)
        else:
            if self.trigger_button.isChecked():
                self.trigger_button.setChecked(False)
                self._trigger_adsr(note_on=False)
            self.trigger_button.setCheckable(False)
            self.trigger_button.setText("trig")

        self.parameter_changed.emit("trigger_mode", value)

    def _trigger_mode(self) -> Literal["latched", "on/off"]:
        return self._normalize_trigger_mode(self.trigger_mode_combo.currentText())

    @staticmethod
    def _normalize_trigger_mode(value: str) -> Literal["latched", "on/off"]:
        return "on/off" if value.lower() == "on/off" else "latched"

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
            retrigger_mode=self._normalize_retrigger_mode(
                self.retrigger_combo.currentText()
            ),
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
        assert component is not None
        if isinstance(component, GateTriggeredADSR):
            return cast(ADSREnvelope, component.adsr)
        return cast(ADSREnvelope, component)

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
        mode_value = parameters.get(
            "retrigger_mode", self.retrigger_combo.currentText()
        )
        adsr.retrigger_mode = self._normalize_retrigger_mode(str(mode_value))

    @staticmethod
    def _normalize_retrigger_mode(value: str) -> Literal["legato", "punch"]:
        return "legato" if value.lower() == "legato" else "punch"

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
            samples = self._render_gate_triggered_adsr(adsr, gate_signal, num_samples)
        else:
            self._previous_gate = 0.0
            samples = adsr.get_samples(num_samples)

        self.out_port.write(samples)
