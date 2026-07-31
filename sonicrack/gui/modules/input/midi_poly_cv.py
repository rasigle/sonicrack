"""Polyphonic MIDI → multi-voice CV module.

Uses ``soniclab.midi_io.PolyphonicMIDIToCV`` so chords produce independent
pitch / gate / velocity streams per voice slot (instead of collapsing to a
single monophonic priority note).
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
    QPushButton,
)
from soniclab.midi_io import (
    MIDIMessage,
    MIDITriggerOutput,
    NoteOffMessage,
    NoteOnMessage,
    PolyphonicMIDIToCV,
    VoiceCVOutput,
    midi_to_note_name,
)

from sonicrack.gui.modules.input.midi_cv_helpers import ScalarCVOutput
from sonicrack.gui.modules.input.midi_worker_thread import MIDIWorkerThread
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import Port, PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)

# Fixed voice count for stable port layout / patch compatibility.
DEFAULT_VOICES = 4
MAX_VOICES = 4


@register_module()
class MIDIPolyCVModule(ModuleWidget):
    """Hardware MIDI → independent pitch/gate/velocity CV per voice.

    Patch each voice into a separate VCO + ADSR + VCA path for true polyphony
    in a modular graph. Global Mod (CC#1) and a shared any-note Trig are also
    provided. Sustain pedal (CC64) and pitch bend follow soniclab's
    ``PolyphonicMIDIToCV`` behaviour.
    """

    runtime_kind = "midi_poly"

    metadata = ModuleMetadata(
        title="MIDI Poly CV",
        category=ModuleCategory.MIDI,
        description=(
            f"Polyphonic MIDI→CV ({DEFAULT_VOICES} voices): independent "
            "1V/oct, gate, and velocity per simultaneous note"
        ),
        version="1.0.0",
        author="SonicRack",
    )

    device_status_changed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__(
            width=300,
            height=360,
            color=QColor(170, 80, 140),
        )

        self.poly_cv = PolyphonicMIDIToCV(
            max_voices=MAX_VOICES,
            pitch_bend_range=2.0,
        )
        self.midi_worker: MIDIWorkerThread | None = None
        self._is_running = False
        self._global_trigger = MIDITriggerOutput()

        # Per-voice ports and adapters
        self.voice_pitch_ports: list[Port] = []
        self.voice_gate_ports: list[Port] = []
        self.voice_vel_ports: list[Port] = []
        self.voice_pitch_outputs: list[VoiceCVOutput] = []
        self.voice_gate_outputs: list[VoiceCVOutput] = []
        self.voice_vel_outputs: list[VoiceCVOutput] = []

        for index in range(MAX_VOICES):
            n = index + 1
            pitch, gate, vel = self.poly_cv.voice_outputs(index)
            self.voice_pitch_outputs.append(pitch)
            self.voice_gate_outputs.append(gate)
            self.voice_vel_outputs.append(vel)
            self.voice_pitch_ports.append(
                self.add_output(f"V{n} 1V/Oct", signal=PortSignal.PITCH_CV)
            )
            self.voice_gate_ports.append(
                self.add_output(f"V{n} Gate", signal=PortSignal.GATE)
            )
            self.voice_vel_ports.append(
                self.add_output(f"V{n} Vel", signal=PortSignal.CONTROL_CV)
            )

        self.mod_port = self.add_output("Mod", signal=PortSignal.CONTROL_CV)
        self.trig_port = self.add_output("Trig", signal=PortSignal.TRIGGER)
        self.mod_output = ScalarCVOutput(lambda: float(self.poly_cv.mod_wheel))

        layout = self._begin_controls(spacing=6)

        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("(No Device)")
        device_layout.addWidget(self.device_combo)
        layout.addLayout(device_layout)

        bend_row = QHBoxLayout()
        self.bend_range_knob = Knob(
            label="PB ±",
            description="Pitch bend range in semitones (all voices)",
            min_value=1,
            max_value=24,
            default_value=2,
        )
        self.bend_range_knob.value_changed.connect(self._on_bend_range_changed)
        bend_row.addWidget(self.bend_range_knob)
        layout.addLayout(bend_row)

        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        btn_layout.addWidget(self.refresh_btn)
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._toggle_midi)
        btn_layout.addWidget(self.start_btn)
        layout.addLayout(btn_layout)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)

        self.voices_label = QLabel("Voices: 0 / 4")
        self.voices_label.setStyleSheet("color: #ccc; font-size: 11px;")
        layout.addWidget(self.voices_label)

        self.notes_label = QLabel("--")
        self.notes_label.setStyleSheet(
            "color: white; font-size: 12px; font-weight: bold;"
        )
        self.notes_label.setWordWrap(True)
        layout.addWidget(self.notes_label)

        self._finish_controls(layout)

        self.device_status_changed.connect(self._on_status_changed)
        self.register_parameter(
            "device", self.device_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter("pitch_bend_range", self.bend_range_knob)

        self._refresh_devices()
        logger.info("MIDI Poly CV module initialized (%s voices)", MAX_VOICES)

    def _on_bend_range_changed(self) -> None:
        self.poly_cv.pitch_bend_range = float(self.bend_range_knob.get_value())
        # Retune active voices with the new span applied to current bend.
        for voice in self.poly_cv.voices:
            if voice.note is not None:
                self.poly_cv._update_voice_pitch(voice)  # noqa: SLF001
        self.parameter_changed.emit(
            "pitch_bend_range", int(self.bend_range_knob.get_value())
        )

    def _refresh_devices(self) -> None:
        try:
            from soniclab.midi_io import MIDIInput
            from soniclab.midi_io.input import MIDO_AVAILABLE

            if not MIDO_AVAILABLE:
                self.device_status_changed.emit("MIDI library not installed")
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
            self.device_combo.blockSignals(False)

            if devices:
                self.device_status_changed.emit(f"Found {len(devices)} device(s)")
            else:
                self.device_status_changed.emit("No MIDI devices found")
        except Exception as e:
            self.device_status_changed.emit(f"Error: {e}")
            logger.error("MIDI Poly CV device list failed: %s", e, exc_info=True)

    def _toggle_midi(self) -> None:
        if self._is_running:
            self._stop_midi()
        else:
            self._start_midi()

    def _start_midi(self) -> None:
        device = self.device_combo.currentText()
        if device == "(No Device)":
            self.device_status_changed.emit("Please select a device")
            return
        try:
            self.midi_worker = MIDIWorkerThread(device)
            self.midi_worker.message_received.connect(self._on_midi_message)
            self.midi_worker.status_changed.connect(self.device_status_changed.emit)
            self.midi_worker.error_occurred.connect(self._on_worker_error)
            self.midi_worker.start()
            self._is_running = True
            self.start_btn.setText("Stop")
            self.device_status_changed.emit("Connecting...")
        except Exception as e:
            self.device_status_changed.emit(f"Error: {e}")
            logger.error("Failed to start MIDI Poly CV: %s", e, exc_info=True)

    def _stop_midi(self) -> None:
        if self.midi_worker is not None:
            try:
                self.midi_worker.stop()
                self.midi_worker.wait(2000)
            except Exception as e:
                logger.error("Error stopping MIDI Poly CV worker: %s", e)
            finally:
                self.midi_worker = None
        self._is_running = False
        self.start_btn.setText("Start")
        self.poly_cv.reset()
        self._global_trigger.reset()
        self._refresh_voice_display()

    def shutdown(self, graceful: bool = True) -> None:
        del graceful
        self._stop_midi()

    def _on_worker_error(self, error: str) -> None:
        logger.error("MIDI Poly CV worker error: %s", error)
        self.device_status_changed.emit(f"Error: {error}")
        self._stop_midi()

    def _on_status_changed(self, status: str) -> None:
        self.status_label.setText(status)

    def _on_midi_message(self, msg: MIDIMessage) -> None:
        prior_active = self.poly_cv.active_voice_count
        prior_notes = set(self.poly_cv.get_playing_notes())

        self.poly_cv.process_message(msg)

        if isinstance(msg, NoteOnMessage) and msg.velocity > 0:
            # Arm shared Trig when a new note is allocated or re-triggered.
            if msg.note not in prior_notes or self.poly_cv.active_voice_count > prior_active:
                self._global_trigger.arm()
            # Also arm on re-press of an already-held note (voice re-trigger).
            elif msg.note in prior_notes:
                self._global_trigger.arm()
        elif isinstance(msg, NoteOffMessage):
            pass

        self._refresh_voice_display()

    def _refresh_voice_display(self) -> None:
        active = self.poly_cv.active_voice_count
        self.voices_label.setText(f"Voices: {active} / {MAX_VOICES}")
        notes = self.poly_cv.get_playing_notes()
        if notes:
            names = ", ".join(f"{midi_to_note_name(n)}({n})" for n in notes)
            self.notes_label.setText(names)
            self.notes_label.setStyleSheet(
                "color: #00ff00; font-size: 12px; font-weight: bold;"
            )
        else:
            self.notes_label.setText("--")
            self.notes_label.setStyleSheet(
                "color: white; font-size: 12px; font-weight: bold;"
            )

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> PolyphonicMIDIToCV:
        del input_components, modulation_components
        return self.poly_cv

    def get_output_component(self, port_name: str) -> Any:
        if port_name == "Mod":
            return self.mod_output
        if port_name == "Trig":
            return self._global_trigger
        for index in range(MAX_VOICES):
            n = index + 1
            if port_name == f"V{n} 1V/Oct":
                return self.voice_pitch_outputs[index]
            if port_name == f"V{n} Gate":
                return self.voice_gate_outputs[index]
            if port_name == f"V{n} Vel":
                return self.voice_vel_outputs[index]
        return self.voice_pitch_outputs[0]

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        bend_range = parameters.get("pitch_bend_range") if parameters else None
        if bend_range is not None:
            self.poly_cv.pitch_bend_range = float(bend_range)

        for index in range(MAX_VOICES):
            self.voice_pitch_ports[index].write(
                self.voice_pitch_outputs[index].get_samples(num_samples)
            )
            self.voice_gate_ports[index].write(
                self.voice_gate_outputs[index].get_samples(num_samples)
            )
            self.voice_vel_ports[index].write(
                self.voice_vel_outputs[index].get_samples(num_samples)
            )
        self.mod_port.write(self.mod_output.get_samples(num_samples))
        self.trig_port.write(self._global_trigger.get_samples(num_samples))

    def note_on(self, note: int, velocity: int = 100) -> None:
        """Public helper for tests / local note injection without hardware."""
        prior_notes = set(self.poly_cv.get_playing_notes())
        self.poly_cv.note_on(note, velocity)
        if velocity > 0:
            self._global_trigger.arm()
            if note in prior_notes:
                pass  # re-trigger still armed above
        self._refresh_voice_display()

    def note_off(self, note: int) -> None:
        """Public helper for tests / local note injection without hardware."""
        self.poly_cv.note_off(note)
        self._refresh_voice_display()

    def set_active(self, active: bool) -> None:
        super().set_active(active)
        if not active:
            self.poly_cv.reset()
            self._global_trigger.reset()
            self._refresh_voice_display()

    def __del__(self) -> None:
        with suppress(Exception):
            self._stop_midi()
