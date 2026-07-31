"""MIDI Input module for the modular synthesizer GUI.

Real-time MIDI from connected controllers, converted to modular CV via
``soniclab.midi_io.MIDIToCV`` (monophonic note-stack with last/high/low
priority, sustain pedal, pitch bend, mod wheel, and expression).

Outputs:
    - 1V/Oct: pitch CV (1V/oct, 0V = MIDI 60 / C4) including pitch bend
    - Gate: held high while any note is active (incl. sustain)
    - Trig: one-sample pulse on each note-on (including legato)
    - Vel: velocity 0–1 of the priority note
    - Mod: CC#1 mod wheel 0–1
    - Expr: CC#11 expression 0–1
    - Bend: pitch bend as bipolar CV [-1, 1]

Polyphonic chords are accepted on the input stream; monophonic priority
selects which held note drives pitch/gate/velocity. For independent
per-voice CV, use the **MIDI Poly CV** module.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
)
from soniclab.midi_io import (
    CVFrequencyOutput,
    CVGateOutput,
    CVVelocityOutput,
    MIDIMessage,
    MIDIToCV,
    MIDITriggerOutput,
    NoteOffMessage,
    NoteOnMessage,
    midi_to_note_name,
)

from sonicrack.gui.modules.input.midi_cv_helpers import (
    make_expression_output,
    make_mod_wheel_output,
    make_pitch_bend_output,
    priority_from_label,
)
from sonicrack.gui.modules.input.midi_worker_thread import MIDIWorkerThread
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)


@register_module()
class MIDIInputModule(ModuleWidget):
    """Hardware MIDI input → monophonic CV with full soniclab MIDIToCV API."""

    runtime_kind = "midi"

    metadata = ModuleMetadata(
        title="MIDI Input",
        category=ModuleCategory.MIDI,
        description=(
            "Real-time MIDI input with mono note-stack priority, "
            "sustain, mod wheel, expression, and pitch bend"
        ),
        version="1.1.0",
        author="SonicRack",
    )

    midi_message_received = pyqtSignal(object)  # MIDIMessage
    device_status_changed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__(
            width=260,
            height=230,
            color=QColor(200, 100, 150),
        )

        self.freq_port = self.add_output("1V/Oct", signal=PortSignal.PITCH_CV)
        self.gate_port = self.add_output("Gate", signal=PortSignal.GATE)
        self.trigger_port = self.add_output("Trig", signal=PortSignal.TRIGGER)
        self.vel_port = self.add_output("Vel", signal=PortSignal.CONTROL_CV)
        self.mod_port = self.add_output("Mod", signal=PortSignal.CONTROL_CV)
        self.expr_port = self.add_output("Expr", signal=PortSignal.CONTROL_CV)
        self.bend_port = self.add_output("Bend", signal=PortSignal.CONTROL_CV)

        self.midi_worker: MIDIWorkerThread | None = None
        self.cv_converter = MIDIToCV(note_priority="last")
        self.freq_output = CVFrequencyOutput(self.cv_converter)
        self.gate_output = CVGateOutput(self.cv_converter)
        self.trigger_output = MIDITriggerOutput()
        self.vel_output = CVVelocityOutput(self.cv_converter)
        self.mod_output = make_mod_wheel_output(self.cv_converter)
        self.expr_output = make_expression_output(self.cv_converter)
        self.bend_output = make_pitch_bend_output(self.cv_converter)
        self._is_running = False
        self._last_note_display = "--"

        layout = self._begin_controls()

        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("(No Device)")
        self.device_combo.setToolTip(
            "MIDI input device (active while selected). "
            "Right-click module to refresh the list."
        )
        self.device_combo.currentTextChanged.connect(self._on_device_changed)
        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)

        bend_layout = QHBoxLayout()
        self.bend_range_knob = Knob(
            label="PB ±",
            description="Pitch bend range in semitones",
            min_value=1,
            max_value=24,
            default_value=2,
        )
        self.bend_range_knob.value_changed.connect(self._on_bend_range_changed)
        bend_layout.addWidget(self.bend_range_knob)
        layout.addLayout(bend_layout)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)

        self.note_label = QLabel("--")
        self.note_label.setStyleSheet(
            "color: white; font-size: 14px; font-weight: bold;"
        )
        layout.addWidget(self.note_label)

        self._finish_controls(layout)

        self.midi_message_received.connect(self._on_midi_message)
        self.device_status_changed.connect(self._on_status_changed)

        # Setup option: right-click module header → Priority
        self.priority_param = self.register_menu_choice(
            "priority",
            "Priority",
            ["Last", "High", "Low"],
            "Last",
            on_changed=self._on_priority_changed,
            tooltip=(
                "Which held note drives pitch/gate when several keys are down "
                "(soniclab MIDIToCV note-stack)"
            ),
        )

        self.register_parameter(
            "device", self.device_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter("pitch_bend_range", self.bend_range_knob)

        self._refresh_devices()
        logger.info("MIDI Input module initialized")

    def _on_priority_changed(self, label: str) -> None:
        self.cv_converter.note_priority = priority_from_label(label)
        # Re-evaluate active note under the new priority rule.
        if self.cv_converter.held_notes:
            self.cv_converter._update_from_stack()  # noqa: SLF001 — public stack API
            self._refresh_note_display()

    def _on_bend_range_changed(self) -> None:
        self.cv_converter.pitch_bend_range = float(self.bend_range_knob.get_value())
        if self.cv_converter.current_note is not None:
            self.cv_converter._update_pitch_cv()  # noqa: SLF001
        self.parameter_changed.emit(
            "pitch_bend_range", int(self.bend_range_knob.get_value())
        )

    def _refresh_note_display(self) -> None:
        note = self.cv_converter.current_note
        if note is not None and self.cv_converter.gate > 0.0:
            name = midi_to_note_name(note)
            held = len(self.cv_converter.held_notes)
            suffix = f" +{held - 1}" if held > 1 else ""
            text = f"{name} ({note}){suffix}"
            self.note_label.setText(text)
            self.note_label.setStyleSheet(
                "color: #00ff00; font-size: 14px; font-weight: bold;"
            )
            self._last_note_display = text
        else:
            self.note_label.setText("--")
            self.note_label.setStyleSheet(
                "color: white; font-size: 14px; font-weight: bold;"
            )
            self._last_note_display = "--"

    def _populate_context_menu(self, menu) -> None:
        menu.addSeparator()
        refresh_action = menu.addAction("Refresh MIDI Devices")
        refresh_action.triggered.connect(self._refresh_devices)

    def _refresh_devices(self) -> None:
        try:
            from soniclab.midi_io import MIDIInput
            from soniclab.midi_io.input import MIDO_AVAILABLE

            if not MIDO_AVAILABLE:
                self.device_status_changed.emit("MIDI library not installed")
                logger.warning("mido library not available")
                return

            devices = MIDIInput.list_devices()
            current = self.device_combo.currentText()
            self.device_combo.blockSignals(True)
            self.device_combo.clear()
            self.device_combo.addItem("(No Device)")
            self.device_combo.addItems(devices)
            idx = self.device_combo.findText(current)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
            else:
                self.device_combo.setCurrentIndex(0)
            self.device_combo.blockSignals(False)

            if devices:
                self.device_status_changed.emit(f"Found {len(devices)} device(s)")
            else:
                self.device_status_changed.emit("No MIDI devices found")
            logger.info("Refreshed MIDI devices: %s", devices)
        except ImportError:
            self.device_status_changed.emit("MIDI library not installed")
            logger.warning("Could not import MIDI modules")
        except Exception as e:
            self.device_status_changed.emit(f"Error: {e}")
            logger.error("Failed to list MIDI devices: %s", e, exc_info=True)
        finally:
            # Re-sync connection if the selected device disappeared / changed.
            self._sync_midi_device()

    def _on_device_changed(self, device: str) -> None:
        self.parameter_changed.emit("device", device)
        self._sync_midi_device(device)

    def _sync_midi_device(self, device: str | None = None) -> None:
        """Keep the MIDI worker active exactly while a real device is selected."""
        if device is None:
            device = self.device_combo.currentText()

        if not device or device == "(No Device)":
            if self._is_running:
                self._stop_midi()
            elif self.status_label.text() in ("", "Connecting..."):
                self.device_status_changed.emit("Idle")
            return

        if (
            self._is_running
            and self.midi_worker is not None
            and self.midi_worker.device_name == device
        ):
            return

        if self._is_running:
            self._stop_midi()
        self._start_midi(device)

    def _start_midi(self, device: str | None = None) -> None:
        if device is None:
            device = self.device_combo.currentText()
        if not device or device == "(No Device)":
            return

        try:
            self.midi_worker = MIDIWorkerThread(device)
            self.midi_worker.message_received.connect(self._on_midi_message)
            self.midi_worker.status_changed.connect(self._on_worker_status_changed)
            self.midi_worker.error_occurred.connect(self._on_worker_error)
            self.midi_worker.start()
            self._is_running = True
            self.device_status_changed.emit("Connecting...")
        except Exception as e:
            self._is_running = False
            self.midi_worker = None
            self.device_status_changed.emit(f"Error: {e}")
            logger.error("Failed to start MIDI worker: %s", e, exc_info=True)

    def _stop_midi(self) -> None:
        if self.midi_worker:
            try:
                self.midi_worker.stop()
                self.midi_worker.wait(2000)
            except Exception as e:
                logger.error("Error stopping MIDI worker: %s", e)
            finally:
                self.midi_worker = None

        self._is_running = False
        self.cv_converter.reset()
        self.trigger_output.reset()
        self._refresh_note_display()
        self.device_status_changed.emit("Idle")
        logger.info("Stopped MIDI input")

    def shutdown(self, graceful: bool = True) -> None:
        del graceful
        self._stop_midi()

    def _on_worker_status_changed(self, status: str) -> None:
        self.device_status_changed.emit(status)

    def _on_worker_error(self, error: str) -> None:
        logger.error("Worker error: %s", error)
        self._stop_midi()
        self.device_status_changed.emit(f"Error: {error}")

    def _on_midi_message(self, msg: MIDIMessage) -> None:
        """Handle received MIDI (UI thread); arm Trig on note-on including legato."""
        prior_note = self.cv_converter.current_note
        prior_gate = self.cv_converter.gate

        self.cv_converter.process_message(msg)

        if isinstance(msg, NoteOnMessage) and msg.velocity > 0:
            # Fire on new notes and priority changes while other keys stay held.
            if (
                self.cv_converter.gate > 0.0
                and self.cv_converter.current_note is not None
                and (prior_gate == 0.0 or self.cv_converter.current_note != prior_note)
            ):
                self.trigger_output.arm()
            self._refresh_note_display()
        elif isinstance(msg, NoteOffMessage):
            self._refresh_note_display()
        else:
            # CC / pitch bend — keep note label, no trig
            pass

    def _on_status_changed(self, status: str) -> None:
        self.status_label.setText(status)

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> MIDIToCV:
        del input_components, modulation_components
        return self.cv_converter

    def get_output_component(self, port_name: str) -> Any:
        outputs = {
            "1V/Oct": self.freq_output,
            "Freq": self.freq_output,
            "Gate": self.gate_output,
            "Trig": self.trigger_output,
            "Vel": self.vel_output,
            "Mod": self.mod_output,
            "Expr": self.expr_output,
            "Bend": self.bend_output,
        }
        return outputs.get(port_name, self.freq_output)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        priority = parameters.get("priority")
        if isinstance(priority, str):
            wanted = priority_from_label(priority)
            if self.cv_converter.note_priority != wanted:
                self.cv_converter.note_priority = wanted
        bend_range = parameters.get("pitch_bend_range")
        if isinstance(bend_range, (int, float, str)):
            self.cv_converter.pitch_bend_range = float(bend_range)

        self.freq_port.write(self.freq_output.get_samples(num_samples))
        self.gate_port.write(self.gate_output.get_samples(num_samples))
        self.trigger_port.write(self.trigger_output.get_samples(num_samples))
        self.vel_port.write(self.vel_output.get_samples(num_samples))
        self.mod_port.write(self.mod_output.get_samples(num_samples))
        self.expr_port.write(self.expr_output.get_samples(num_samples))
        self.bend_port.write(self.bend_output.get_samples(num_samples))

    def __del__(self) -> None:
        with suppress(Exception):
            self._stop_midi()
