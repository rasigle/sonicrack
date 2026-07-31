"""Polyphonic subtractive voice with on-screen keyboard."""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeyEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QGraphicsItem,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)
from soniclab import ADSREnvelope, Chain, SawtoothOscillator, SineOscillator
from soniclab.dsp.modifiers import ModulatedVolume
from soniclab.generators.oscillators.oscillator import SquareOscillator, TriangleOscillator
from soniclab.midi_io import (
    NoteOffMessage,
    NoteOnMessage,
    PolyphonicSynth,
    midi_to_note_name,
)

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.input.midi_worker_thread import MIDIWorkerThread
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, str_parameter
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)

WHITE_KEYS = (0, 2, 4, 5, 7, 9, 11)
BLACK_KEYS = (1, 3, 6, 8, 10)
NOTE_LABELS = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}
BLACK_KEY_COLUMNS = {1: 1, 3: 3, 6: 7, 8: 9, 10: 11}
COMPUTER_KEY_OFFSETS: dict[int, int] = {
    Qt.Key.Key_A.value: 0,
    Qt.Key.Key_W.value: 1,
    Qt.Key.Key_S.value: 2,
    Qt.Key.Key_E.value: 3,
    Qt.Key.Key_D.value: 4,
    Qt.Key.Key_F.value: 5,
    Qt.Key.Key_T.value: 6,
    Qt.Key.Key_G.value: 7,
    Qt.Key.Key_Y.value: 8,
    Qt.Key.Key_H.value: 9,
    Qt.Key.Key_U.value: 10,
    Qt.Key.Key_J.value: 11,
}


@register_module()
class PolyVoiceModule(ModuleWidget):
    """Polyphonic keyboard instrument with voice allocation.

    Uses ``soniclab.PolyphonicSynth`` for free → releasing → oldest-active
    voice stealing. Play via the on-screen keys, computer keyboard
    (``A W S E D F T G Y H U J``), or an optional MIDI input device for
    true multi-note controller routing.
    """

    runtime_kind = "poly_voice"

    # Thread-safe UI updates from the MIDI worker
    midi_status_changed = pyqtSignal(str)

    metadata = ModuleMetadata(
        title="Poly Voice",
        category=ModuleCategory.MODULATED_SOURCE,
        description="Polyphonic subtractive voice with keyboard, MIDI, and ADSR",
    )

    def __init__(self) -> None:
        super().__init__(width=400, height=480, color=QColor(90, 140, 160))

        self.out_port = self.add_output("Out", signal=PortSignal.AUDIO)
        self._lock = threading.RLock()
        self._pressed_note_offsets: dict[int, int] = {}
        self._midi_held_notes: set[int] = set()
        self._voice_params = {
            "waveform": "Sine",
            "attack": 0.01,
            "decay": 0.15,
            "sustain": 0.7,
            "release": 0.25,
            "gain": 0.35,
        }
        self._max_voices = 8
        self.synth = self._build_synth()
        self.midi_worker: MIDIWorkerThread | None = None
        self._midi_running = False

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = self._begin_controls(spacing=6)

        # Optional MIDI device for polyphonic controller input
        midi_row = QHBoxLayout()
        midi_row.addWidget(QLabel("MIDI:"))
        self.midi_device_combo = QComboBox()
        self.midi_device_combo.addItem("(No Device)")
        self.midi_device_combo.setToolTip(
            "Select a MIDI keyboard/controller for polyphonic note input"
        )
        midi_row.addWidget(self.midi_device_combo)
        self.midi_refresh_btn = QPushButton("↻")
        self.midi_refresh_btn.setMaximumWidth(28)
        self.midi_refresh_btn.setToolTip("Refresh MIDI device list")
        self.midi_refresh_btn.clicked.connect(self._refresh_midi_devices)
        midi_row.addWidget(self.midi_refresh_btn)
        self.midi_start_btn = QPushButton("Start")
        self.midi_start_btn.setMaximumWidth(52)
        self.midi_start_btn.clicked.connect(self._toggle_midi)
        midi_row.addWidget(self.midi_start_btn)
        layout.addLayout(midi_row)

        self.midi_status_label = QLabel("MIDI idle")
        self.midi_status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.midi_status_label)
        self.midi_status_changed.connect(self._on_midi_status)

        settings = QHBoxLayout()
        settings.addWidget(QLabel("Oct:"))
        self.octave_combo = QComboBox()
        self.octave_combo.addItems([str(o) for o in range(1, 8)])
        self.octave_combo.setCurrentText("4")
        self.octave_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("octave", value)
        )
        settings.addWidget(self.octave_combo)

        settings.addWidget(QLabel("Wave:"))
        self.wave_combo = QComboBox()
        self.wave_combo.addItems(["Sine", "Triangle", "Sawtooth", "Square"])
        self.wave_combo.currentTextChanged.connect(self._on_waveform_changed)
        settings.addWidget(self.wave_combo)

        self.voices_knob = Knob(
            label="Voices",
            description="Maximum simultaneous voices",
            min_value=1,
            max_value=16,
            default_value=8,
        )
        self.voices_knob.value_changed.connect(self._on_voices_changed)
        settings.addWidget(self.voices_knob)
        layout.addLayout(settings)

        env_row = QHBoxLayout()
        self.attack_knob = Knob(
            label="A",
            description="Attack time (s)",
            min_value=0.001,
            max_value=2.0,
            default_value=0.01,
            logarithmic=True,
        )
        self.decay_knob = Knob(
            label="D",
            description="Decay time (s)",
            min_value=0.001,
            max_value=2.0,
            default_value=0.15,
            logarithmic=True,
        )
        self.sustain_knob = Knob(
            label="S",
            description="Sustain level",
            min_value=0.0,
            max_value=1.0,
            default_value=0.7,
        )
        self.release_knob = Knob(
            label="R",
            description="Release time (s)",
            min_value=0.01,
            max_value=3.0,
            default_value=0.25,
            logarithmic=True,
        )
        for knob, name in (
            (self.attack_knob, "attack"),
            (self.decay_knob, "decay"),
            (self.sustain_knob, "sustain"),
            (self.release_knob, "release"),
        ):
            knob.value_changed.connect(
                lambda _=None, n=name, k=knob: self._on_env_param(n, k)
            )
            env_row.addWidget(knob)
        layout.addLayout(env_row)

        self.velocity_knob = Knob(
            label="Vel",
            description="Keyboard velocity",
            min_value=1,
            max_value=127,
            default_value=100,
        )
        self.velocity_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "velocity", int(self.velocity_knob.get_value())
            )
        )
        layout.addWidget(self.velocity_knob)

        self.note_label = QLabel("--")
        self.note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note_label.setStyleSheet(
            "color: white; font-size: 13px; font-weight: bold;"
        )
        layout.addWidget(self.note_label)

        keyboard_layout = QGridLayout()
        keyboard_layout.setHorizontalSpacing(2)
        keyboard_layout.setVerticalSpacing(2)
        self.key_buttons: dict[int, QPushButton] = {}
        self._add_keyboard_buttons(keyboard_layout)
        layout.addLayout(keyboard_layout)

        self._finish_controls(layout)

        self.register_parameter(
            "octave", self.octave_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter(
            "waveform", self.wave_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter("voices", self.voices_knob)
        self.register_parameter("attack", self.attack_knob)
        self.register_parameter("decay", self.decay_knob)
        self.register_parameter("sustain", self.sustain_knob)
        self.register_parameter("release", self.release_knob)
        self.register_parameter("velocity", self.velocity_knob)
        self.register_parameter(
            "midi_device",
            self.midi_device_combo,
            getter="currentText",
            setter="setCurrentText",
        )
        self._install_sample_rate_listener()
        self._refresh_midi_devices()

    def _refresh_midi_devices(self) -> None:
        """Refresh available MIDI input devices."""
        try:
            from soniclab.midi_io import MIDIInput
            from soniclab.midi_io.input import MIDO_AVAILABLE

            if not MIDO_AVAILABLE:
                self.midi_status_changed.emit("MIDI library not installed")
                return
            devices = MIDIInput.list_devices()
            current = self.midi_device_combo.currentText()
            self.midi_device_combo.blockSignals(True)
            self.midi_device_combo.clear()
            self.midi_device_combo.addItem("(No Device)")
            self.midi_device_combo.addItems(devices)
            idx = self.midi_device_combo.findText(current)
            if idx >= 0:
                self.midi_device_combo.setCurrentIndex(idx)
            self.midi_device_combo.blockSignals(False)
            if devices:
                self.midi_status_changed.emit(f"Found {len(devices)} MIDI device(s)")
            else:
                self.midi_status_changed.emit("No MIDI devices found")
        except Exception as exc:  # pragma: no cover - hardware dependent
            logger.warning("Poly Voice MIDI device list failed: %s", exc)
            self.midi_status_changed.emit(f"MIDI error: {exc}")

    def _toggle_midi(self) -> None:
        if self._midi_running:
            self._stop_midi()
        else:
            self._start_midi()

    def _start_midi(self) -> None:
        device = self.midi_device_combo.currentText()
        if device == "(No Device)":
            self.midi_status_changed.emit("Select a MIDI device first")
            return
        try:
            self.midi_worker = MIDIWorkerThread(device)
            self.midi_worker.message_received.connect(self._on_midi_message)
            self.midi_worker.status_changed.connect(self.midi_status_changed.emit)
            self.midi_worker.error_occurred.connect(self._on_midi_error)
            self.midi_worker.start()
            self._midi_running = True
            self.midi_start_btn.setText("Stop")
            self.midi_status_changed.emit(f"Connecting: {device}")
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to start Poly Voice MIDI: %s", exc, exc_info=True)
            self.midi_status_changed.emit(f"Error: {exc}")

    def _stop_midi(self) -> None:
        if self.midi_worker is not None:
            with suppress(Exception):
                self.midi_worker.stop()
                self.midi_worker.wait(2000)
            self.midi_worker = None
        self._midi_running = False
        self.midi_start_btn.setText("Start")
        with self._lock:
            for note in list(self._midi_held_notes):
                self.synth.note_off(note)
            self._midi_held_notes.clear()
        self.midi_status_changed.emit("MIDI idle")

    def _on_midi_status(self, status: str) -> None:
        self.midi_status_label.setText(status)

    def _on_midi_error(self, error: str) -> None:
        logger.error("Poly Voice MIDI error: %s", error)
        self.midi_status_changed.emit(f"Error: {error}")
        self._stop_midi()

    def _on_midi_message(self, msg: Any) -> None:
        """Route polyphonic note on/off from the MIDI worker into the voice pool."""
        if isinstance(msg, NoteOnMessage):
            if msg.velocity > 0:
                with self._lock:
                    self.synth.note_on(int(msg.note), int(msg.velocity))
                    self._midi_held_notes.add(int(msg.note))
                self.note_label.setText(f"{midi_to_note_name(msg.note)} ({msg.note})")
                self.note_label.setStyleSheet(
                    "color: #00ff00; font-size: 13px; font-weight: bold;"
                )
            else:
                # Note-on with velocity 0 = note off
                with self._lock:
                    self.synth.note_off(int(msg.note))
                    self._midi_held_notes.discard(int(msg.note))
        elif isinstance(msg, NoteOffMessage):
            with self._lock:
                self.synth.note_off(int(msg.note))
                self._midi_held_notes.discard(int(msg.note))
            if not self._midi_held_notes and not self._pressed_note_offsets:
                self.note_label.setText("--")
                self.note_label.setStyleSheet(
                    "color: white; font-size: 13px; font-weight: bold;"
                )

    def shutdown(self, graceful: bool = True) -> None:
        """Release MIDI resources on app/module teardown."""
        del graceful
        self._stop_midi()

    def _add_keyboard_buttons(self, layout: QGridLayout) -> None:
        for index, note_offset in enumerate(WHITE_KEYS):
            button = self._create_key_button(note_offset, is_black=False)
            layout.addWidget(button, 1, index * 2, 1, 2)
        for note_offset in BLACK_KEYS:
            button = self._create_key_button(note_offset, is_black=True)
            layout.addWidget(button, 0, BLACK_KEY_COLUMNS[note_offset], 1, 2)

    def _create_key_button(self, note_offset: int, *, is_black: bool) -> QPushButton:
        button = QPushButton(NOTE_LABELS[note_offset])
        button.setMinimumHeight(40 if is_black else 54)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setStyleSheet(
            """
            QPushButton {
                background-color: #111418;
                color: #f4f6f8;
                border: 1px solid #050608;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:pressed { background-color: #2f85c8; }
            """
            if is_black
            else """
            QPushButton {
                background-color: #f1f3f5;
                color: #171a1f;
                border: 1px solid #9aa3ad;
                border-radius: 3px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:pressed { background-color: #6fc7ff; }
            """
        )
        button.pressed.connect(lambda offset=note_offset: self._note_on(offset))
        button.released.connect(lambda offset=note_offset: self._note_off(offset))
        self.key_buttons[note_offset] = button
        return button

    def _build_synth(self) -> PolyphonicSynth:
        return PolyphonicSynth(
            self._voice_factory,
            max_voices=self._max_voices,
            sample_rate=audio_config.sample_rate,
        )

    def _voice_factory(self) -> Any:
        params = self._voice_params
        wave = params["waveform"]
        freq = 440.0
        sr = audio_config.sample_rate
        if wave == "Triangle":
            osc: Any = TriangleOscillator(freq, sample_rate=sr, mode="pure")
        elif wave == "Sawtooth":
            osc = SawtoothOscillator(freq, sample_rate=sr, mode="vcv")
        elif wave == "Square":
            osc = SquareOscillator(freq, sample_rate=sr, mode="vcv")
        else:
            osc = SineOscillator(freq, sample_rate=sr, mode="pure")

        env = ADSREnvelope(
            attack_duration=float(params["attack"]),
            decay_duration=float(params["decay"]),
            sustain_level=float(params["sustain"]),
            release_duration=float(params["release"]),
            sample_rate=sr,
        )
        chain = Chain(osc, ModulatedVolume(env))
        # Soft per-voice level so chords stay manageable before RMS mix.
        if hasattr(osc, "amplitude"):
            osc.amplitude = float(params["gain"])
        return chain

    def _sync_voice_params_from_ui(self) -> None:
        self._voice_params = {
            "waveform": self.wave_combo.currentText(),
            "attack": self.attack_knob.get_value(),
            "decay": self.decay_knob.get_value(),
            "sustain": self.sustain_knob.get_value(),
            "release": self.release_knob.get_value(),
            "gain": 0.35,
        }

    def _on_waveform_changed(self, value: str) -> None:
        self._sync_voice_params_from_ui()
        self.parameter_changed.emit("waveform", value)

    def _on_env_param(self, name: str, knob: Knob) -> None:
        self._sync_voice_params_from_ui()
        self.parameter_changed.emit(name, knob.get_value())

    def _on_voices_changed(self) -> None:
        voices = max(1, min(16, int(round(self.voices_knob.get_value()))))
        self.parameter_changed.emit("voices", float(voices))
        with self._lock:
            if voices != self._max_voices:
                self._max_voices = voices
                self.synth = self._build_synth()

    def _base_note(self) -> int:
        return (int(self.octave_combo.currentText()) + 1) * 12

    def _midi_note_for_offset(self, note_offset: int) -> int:
        return max(0, min(127, self._base_note() + note_offset))

    def _note_on(self, note_offset: int) -> None:
        note = self._midi_note_for_offset(note_offset)
        velocity = int(self.velocity_knob.get_value())
        self._pressed_note_offsets[note_offset] = note
        with self._lock:
            self.synth.note_on(note, velocity)
        self.key_buttons[note_offset].setDown(True)
        self.note_label.setText(f"{midi_to_note_name(note)} ({note})")
        self.note_label.setStyleSheet(
            "color: #00ff00; font-size: 13px; font-weight: bold;"
        )

    def _note_off(self, note_offset: int) -> None:
        note = self._pressed_note_offsets.pop(
            note_offset, self._midi_note_for_offset(note_offset)
        )
        with self._lock:
            self.synth.note_off(note)
        self.key_buttons[note_offset].setDown(False)
        if self._pressed_note_offsets:
            fallback = next(reversed(self._pressed_note_offsets.values()))
            self.note_label.setText(f"{midi_to_note_name(fallback)} ({fallback})")
        else:
            self.note_label.setText("--")
            self.note_label.setStyleSheet(
                "color: white; font-size: 13px; font-weight: bold;"
            )

    def note_on_midi(self, note: int, velocity: int = 100) -> None:
        """Trigger a MIDI note from tests or external helpers."""
        with self._lock:
            self.synth.note_on(note, velocity)

    def note_off_midi(self, note: int) -> None:
        """Release a MIDI note from tests or external helpers."""
        with self._lock:
            self.synth.note_off(note)

    def mousePressEvent(self, event: Any) -> None:
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            return
        note_offset = COMPUTER_KEY_OFFSETS.get(event.key())
        if note_offset is None:
            super().keyPressEvent(event)
            return
        if not event.isAutoRepeat() and note_offset not in self._pressed_note_offsets:
            self._note_on(note_offset)
        event.accept()

    def keyReleaseEvent(self, event: QKeyEvent | None) -> None:
        if event is None:
            return
        note_offset = COMPUTER_KEY_OFFSETS.get(event.key())
        if note_offset is None:
            super().keyReleaseEvent(event)
            return
        if not event.isAutoRepeat():
            self._note_off(note_offset)
        event.accept()

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        del new_sample_rate
        with self._lock:
            self.synth = self._build_synth()

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        self._voice_params = {
            "waveform": str_parameter(
                parameters, "waveform", self.wave_combo.currentText
            ),
            "attack": float_parameter(parameters, "attack", self.attack_knob.get_value),
            "decay": float_parameter(parameters, "decay", self.decay_knob.get_value),
            "sustain": float_parameter(
                parameters, "sustain", self.sustain_knob.get_value
            ),
            "release": float_parameter(
                parameters, "release", self.release_knob.get_value
            ),
            "gain": 0.35,
        }
        voices = max(
            1,
            min(16, int(round(float_parameter(parameters, "voices", self.voices_knob.get_value)))),
        )
        with self._lock:
            if voices != self._max_voices:
                self._max_voices = voices
                self.synth = self._build_synth()
            samples = self.synth.get_samples(num_samples)
        self.out_port.write(np.asarray(samples, dtype=np.float32))

    def set_active(self, active: bool) -> None:
        super().set_active(active)
        if not active:
            with self._lock:
                self.synth.reset()
                self._midi_held_notes.clear()
            for offset in list(self._pressed_note_offsets):
                self.key_buttons[offset].setDown(False)
            self._pressed_note_offsets.clear()
            self.note_label.setText("--")
