import logging
from typing import Any, Literal, cast

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QPushButton
from soniclab.dsp.modulators import ADSREnvelope, GateTriggeredADSR

from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.envelope_shape_widget import EnvelopeShapeWidget
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.config.audio_config import audio_config
from sonicrack.runtime.helpers import (
    ensure_min_pulse_width,
    float_parameter,
    gate_transition_indices,
    min_trigger_samples,
    read_samples,
)
from sonicrack.runtime.specs import RuntimeParameters


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
        category=ModuleCategory.ENVELOPE,
        description="ADSR envelope generator with gate input for MIDI triggering",
    )

    def __init__(self):
        """Initialize ADSR module."""
        super().__init__(
            width=220,
            height=440,
            color=QColor(120, 180, 80),
        )

        # Add input port for gate signal (optional)
        self.gate_input = self.add_input("Gate", signal=PortSignal.GATE)
        # Optional velocity CV (0..1) scaled by Vel Depth into output level
        self.vel_input = self.add_input("Vel", signal=PortSignal.CONTROL_CV)

        # Add output port
        self.out_port = self.add_output("Out", signal=PortSignal.CONTROL_CV)

        # Use helper methods for UI construction
        layout = self._begin_controls(spacing=6)

        # Live ADSR shape preview with position playhead
        self.shape_widget = EnvelopeShapeWidget()
        layout.addWidget(self.shape_widget)

        # Poll DSP state for the shape playhead (UI thread only).
        self._viz_timer = QTimer(self)
        self._viz_timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._viz_timer.setInterval(33)
        self._viz_timer.timeout.connect(self._update_live_position)
        self._viz_timer.start()

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
            lambda *_: self._on_envelope_knob_changed(
                "attack_duration", self.attack_knob
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
            lambda *_: self._on_envelope_knob_changed("decay_duration", self.decay_knob)
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
            lambda *_: self._on_envelope_knob_changed(
                "sustain_level", self.sustain_knob
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
            lambda *_: self._on_envelope_knob_changed(
                "release_duration", self.release_knob
            )
        )
        knobs_layout2.addWidget(self.release_knob)

        layout.addLayout(knobs_layout2)

        self.vel_depth_knob = Knob(
            label="Vel Depth",
            description="How much the Vel input scales envelope level "
            "(0 = ignore velocity, 1 = full velocity scaling)",
            min_value=0.0,
            max_value=1.0,
            default_value=0.0,
        )
        self.vel_depth_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "velocity_depth", self.vel_depth_knob.get_value()
            )
        )
        layout.addWidget(self.vel_depth_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        # Manual trigger button (play control; mode is in the context menu)
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
            "On/Off: press to start attack, release for release phase\n"
            "Right-click module header to change Retrigger / Trig Mode"
        )
        self.trigger_button.toggled.connect(self._on_trigger_toggled)
        self.trigger_button.pressed.connect(self._on_trigger_pressed)
        self.trigger_button.released.connect(self._on_trigger_released)
        trigger_layout.addWidget(self.trigger_button)
        layout.addLayout(trigger_layout)

        self._finish_controls(layout)

        # Setup options: right-click module header → Retrigger / Trig Mode
        self.retrigger_param = self.register_menu_choice(
            "retrigger_mode",
            "Retrigger",
            ["Punch", "Legato"],
            "Punch",
            tooltip=(
                "Punch: briefly ramps to zero, then attacks for a stronger rhythmic "
                "chop without clicks.\n"
                "Legato: attacks from the current envelope level for smoother overlap."
            ),
        )
        self.trigger_mode_param = self.register_menu_choice(
            "trigger_mode",
            "Trig Mode",
            ["Latched", "On/Off"],
            "Latched",
            on_changed=self._on_trigger_mode_changed,
            tooltip=(
                "Latched: click trig once to hold the envelope gate on, click again "
                "to release.\n"
                "On/Off: hold trig down to gate on, release it to gate off."
            ),
        )

        # Register parameters for automatic get/set
        self.register_parameter("attack_duration", self.attack_knob)
        self.register_parameter("decay_duration", self.decay_knob)
        self.register_parameter("sustain_level", self.sustain_knob)
        self.register_parameter("release_duration", self.release_knob)
        self.register_parameter("velocity_depth", self.vel_depth_knob)

        # Track ADSR component for manual triggering
        self._adsr_component: ADSREnvelope | GateTriggeredADSR | None = None
        self._previous_gate = 0.0  # last sample of stretched gate (for edge events)
        self._previous_raw_gate = 0.0  # last sample of connected source
        # Extends trigger-like micro-pulses so Gate consumers get a usable high time.
        self._gate_hold = 0

        self.component = self.create_engine_component()
        self._update_shape_display()

    def _on_envelope_knob_changed(self, param_name: str, knob: Knob) -> None:
        """Refresh the shape preview and emit the changed parameter signal."""
        self._update_shape_display()
        self.parameter_changed.emit(param_name, knob.get_value())

    def _update_shape_display(self) -> None:
        """Sync the shape widget with the current ADSR knobs."""
        self.shape_widget.set_envelope(
            self.attack_knob.get_value(),
            self.decay_knob.get_value(),
            self.sustain_knob.get_value(),
            self.release_knob.get_value(),
        )

    def _update_live_position(self) -> None:
        """Refresh the shape playhead from the active ADSR component state."""
        if self._adsr_component is None:
            self.shape_widget.clear_live_state()
            return

        adsr = self._current_adsr()
        phase = getattr(adsr, "_phase", "idle")
        phase_name = (
            str(phase.value) if hasattr(phase, "value") else str(phase or "idle")
        ).lower()
        position = int(getattr(adsr, "_phase_position", 0) or 0)
        level = float(getattr(adsr, "val", 0.0) or 0.0)
        ended = bool(getattr(adsr, "ended", True))

        if ended or phase_name in {"idle", "ended"}:
            self.shape_widget.clear_live_state()
            return

        if phase_name == "attack":
            total = max(1, int(getattr(adsr, "_attack_samples", 1) or 1))
        elif phase_name == "decay":
            total = max(1, int(getattr(adsr, "_decay_samples", 1) or 1))
        elif phase_name == "release":
            total = max(1, int(getattr(adsr, "_release_samples", 1) or 1))
        elif phase_name == "retrigger_reset":
            total = max(1, int(getattr(adsr, "_retrigger_reset_samples", 1) or 1))
        else:
            total = 1

        progress = min(1.0, max(0.0, position / total))
        self.shape_widget.set_live_state(phase_name, progress, level)

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Restore parameters and refresh the envelope shape preview."""
        super().set_parameters(params)
        self._update_shape_display()


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

    def _trigger_mode(self) -> Literal["latched", "on/off"]:
        return self._normalize_trigger_mode(self.trigger_mode_param.get_value())

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
                self.retrigger_param.get_value()
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
            "retrigger_mode", self.retrigger_param.get_value()
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
        note_ons, note_offs, final_gate = gate_transition_indices(
            gate_signal, self._previous_gate
        )

        # Merge transition indices; Schmitt thresholds prevent both on one sample.
        events: list[tuple[int, bool]] = [
            *((int(index), True) for index in note_ons),
            *((int(index), False) for index in note_offs),
        ]
        events.sort(key=lambda item: item[0])

        start = 0
        for index, note_on in events:
            if index > start:
                output[start:index] = adsr.get_samples(index - start)
            if note_on:
                adsr.trigger_note_on()
            else:
                adsr.trigger_note_off()
            start = index

        if start < num_samples:
            output[start:] = adsr.get_samples(num_samples - start)

        self._previous_gate = final_gate
        return output

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Render the ADSR envelope for the current engine cycle."""
        adsr = self._current_adsr()
        self._apply_runtime_parameters(adsr, parameters)

        if self.gate_input.is_connected:
            # Accept Gate or Trigger sources. Short triggers (Clock, MIDI Trig)
            # are stretched to a minimum high time so attack can develop.
            raw_gate = read_samples(self.gate_input, num_samples)
            gate_signal, self._gate_hold = ensure_min_pulse_width(
                raw_gate,
                self._previous_raw_gate,
                min_trigger_samples(audio_config.sample_rate),
                self._gate_hold,
            )
            if num_samples > 0:
                self._previous_raw_gate = float(raw_gate[-1])
            samples = self._render_gate_triggered_adsr(adsr, gate_signal, num_samples)
        else:
            self._previous_gate = 0.0
            self._previous_raw_gate = 0.0
            self._gate_hold = 0
            samples = adsr.get_samples(num_samples)

        samples = np.asarray(samples, dtype=np.float32)
        vel_depth = float_parameter(
            parameters, "velocity_depth", self.vel_depth_knob.get_value
        )
        if vel_depth > 0.0 and self.vel_input.is_connected:
            vel = np.clip(read_samples(self.vel_input, num_samples), 0.0, 1.0)
            # Blend: full envelope when depth=0; at depth=1 scale by velocity.
            scale = (1.0 - vel_depth) + vel_depth * vel
            samples = samples * scale

        self.out_port.write(samples)
