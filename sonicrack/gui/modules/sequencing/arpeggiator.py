"""Arpeggiator module wrapping soniclab's note-driven monophonic arpeggiator.

Holds a chord (or free-form MIDI note list), expands it across octaves with a
chosen pattern, and emits sample-accurate pitch CV, gate, trigger, and velocity
driven by an internal or external step clock. Built for experimenting with
``soniclab.sequencing.Arpeggiator`` without wiring a full MIDI stack first.
"""

from __future__ import annotations

from typing import cast

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)
from soniclab.midi_io import midi_to_note_name
from soniclab.sequencing import Arpeggiator, ArpPattern

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import Knob, LedIndicator, LedStyle, ProceduralKnobStyle
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import (
    float_parameter,
    read_samples,
    rising_edge_pulses,
    str_parameter,
)
from sonicrack.runtime.specs import RuntimeParameters

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_DEFAULT_NOTES = "60,64,67"  # C major triad at C4
_PATTERN_LABELS = (
    ("Up", "up"),
    ("Down", "down"),
    ("Up-Down", "up_down"),
    ("Down-Up", "down_up"),
    ("As Played", "as_played"),
    ("Random", "random"),
)
_LABEL_TO_PATTERN = {label: value for label, value in _PATTERN_LABELS}
_PATTERN_TO_LABEL = {value: label for label, value in _PATTERN_LABELS}

# Chord qualities relative to the selected root (semitone offsets).
_CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "Maj": (0, 4, 7),
    "Min": (0, 3, 7),
    "Maj7": (0, 4, 7, 11),
    "Min7": (0, 3, 7, 10),
    "Dom7": (0, 4, 7, 10),
    "Sus2": (0, 2, 7),
    "Sus4": (0, 5, 7),
    "Dim": (0, 3, 6),
    "Aug": (0, 4, 8),
    "Power": (0, 7),
    "Maj9": (0, 4, 7, 11, 14),
    "Min9": (0, 3, 7, 10, 14),
    "Add9": (0, 4, 7, 14),
}


def _midi_label(note: int) -> str:
    """Format MIDI note as name+octave (C4 = 60)."""
    try:
        return midi_to_note_name(note)
    except Exception:
        return f"{_NOTE_NAMES[note % 12]}{note // 12 - 1}"


def _parse_notes(text: str) -> list[int]:
    """Parse comma-separated MIDI note numbers into a clean list."""
    notes: list[int] = []
    for item in text.split(","):
        token = item.strip()
        if not token or token in {"-", "r", "R", "rest", "Rest"}:
            continue
        try:
            notes.append(max(0, min(127, int(token))))
        except ValueError:
            continue
    return notes


@register_module()
class ArpeggiatorModule(ModuleWidget):
    """Chord-held monophonic arpeggiator for soniclab engine experiments."""

    runtime_kind = "arpeggiator"

    # Emitted from the audio path; Qt queues delivery to the GUI thread so
    # labels/LEDs are never touched inside process_runtime.
    playhead_changed = pyqtSignal(int, str)  # step index, note label

    metadata = ModuleMetadata(
        title="Arpeggiator",
        category=ModuleCategory.SEQUENCER,
        description=(
            "Note-driven monophonic arpeggiator (soniclab): patterns, "
            "octaves, latch, internal/external clock"
        ),
        version="1.0.0",
        author="SonicRack",
    )

    def __init__(self) -> None:
        super().__init__(width=320, height=340, color=QColor(70, 130, 170))

        self.clock_input = self.add_input("Clock", signal=PortSignal.TRIGGER)
        self.reset_input = self.add_input("Reset", signal=PortSignal.TRIGGER)
        self.freq_port = self.add_output("Freq", signal=PortSignal.PITCH_CV)
        self.gate_port = self.add_output("Gate", signal=PortSignal.GATE)
        self.trigger_port = self.add_output("Trig", signal=PortSignal.TRIGGER)
        self.vel_port = self.add_output("Vel", signal=PortSignal.CONTROL_CV)

        self.component = Arpeggiator(
            bpm=120.0,
            division="1/16",
            pattern=ArpPattern.UP,
            octaves=1,
            gate_length=0.5,
            sample_rate=audio_config.sample_rate,
        )
        self.component.set_notes(_parse_notes(_DEFAULT_NOTES))

        self._previous_notes = _DEFAULT_NOTES
        self._previous_reset = 0.0
        self._previous_clock = 0.0
        self._displayed_step = -1
        self._displayed_note = ""

        self.playhead_changed.connect(self._on_playhead_changed)

        layout = self._begin_controls(spacing=5)
        compact = ProceduralKnobStyle.small()

        # --- Chord builder (Apply writes into Notes; free-form edits stay free) ---
        chord_row = QHBoxLayout()
        chord_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chord_row.addWidget(QLabel("Root:"))
        self.root_combo = QComboBox()
        self.root_combo.addItems(list(_NOTE_NAMES))
        self.root_combo.setCurrentText("C")
        self.root_combo.setToolTip("Root note for the chord builder")
        self.root_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("root", value)
        )
        chord_row.addWidget(self.root_combo)

        chord_row.addWidget(QLabel("Oct:"))
        self.root_octave_combo = QComboBox()
        self.root_octave_combo.addItems([str(o) for o in range(1, 7)])
        self.root_octave_combo.setCurrentText("4")
        self.root_octave_combo.setToolTip("Root octave for the chord builder")
        self.root_octave_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("root_octave", value)
        )
        chord_row.addWidget(self.root_octave_combo)

        chord_row.addWidget(QLabel("Chord:"))
        self.chord_combo = QComboBox()
        self.chord_combo.addItems(list(_CHORD_INTERVALS.keys()))
        self.chord_combo.setCurrentText("Maj")
        self.chord_combo.setToolTip("Chord quality for the chord builder")
        self.chord_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("chord", value)
        )
        chord_row.addWidget(self.chord_combo)

        self.apply_chord_button = QPushButton("Apply")
        self.apply_chord_button.setToolTip(
            "Write the selected root/chord into the Notes field"
        )
        self.apply_chord_button.setMaximumWidth(52)
        self.apply_chord_button.clicked.connect(self._on_apply_chord_clicked)
        chord_row.addWidget(self.apply_chord_button)
        layout.addLayout(chord_row)

        notes_row = QHBoxLayout()
        notes_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        notes_row.addWidget(QLabel("Notes:"))
        self.notes_edit = QLineEdit(_DEFAULT_NOTES)
        self.notes_edit.setToolTip(
            "Comma-separated MIDI notes held by the arpeggiator "
            "(e.g. 60,64,67 for a C major triad)"
        )
        self.notes_edit.textChanged.connect(self._on_notes_changed)
        notes_row.addWidget(self.notes_edit)
        layout.addLayout(notes_row)

        # --- Pattern + latch ---
        pattern_row = QHBoxLayout()
        pattern_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pattern_row.addWidget(QLabel("Pattern:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems([label for label, _ in _PATTERN_LABELS])
        self.pattern_combo.setCurrentText("Up")
        self.pattern_combo.setToolTip("Note order through the held chord")
        self.pattern_combo.currentTextChanged.connect(self._on_pattern_changed)
        pattern_row.addWidget(self.pattern_combo)

        self.latch_checkbox = QCheckBox("Latch")
        self.latch_checkbox.setToolTip(
            "Keep the last held chord sounding after notes are cleared"
        )
        self.latch_checkbox.toggled.connect(self._on_latch_toggled)
        pattern_row.addWidget(self.latch_checkbox)

        self.run_checkbox = QCheckBox("Run")
        self.run_checkbox.setChecked(True)
        self.run_checkbox.setToolTip("Enable internal clock when no Clock input")
        self.run_checkbox.toggled.connect(
            lambda value: self.parameter_changed.emit("running", value)
        )
        pattern_row.addWidget(self.run_checkbox)
        layout.addLayout(pattern_row)

        # --- Knobs: octaves / gate / transpose / swing ---
        knob_row = QHBoxLayout()
        knob_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.octaves_knob = Knob(
            label="Octaves",
            description="How many octaves the pattern spans (1–4)",
            min_value=1.0,
            max_value=4.0,
            default_value=1.0,
            style=compact,
        )
        self.octaves_knob.value_changed.connect(self._on_octaves_changed)
        knob_row.addWidget(self.octaves_knob)

        self.gate_length_knob = Knob(
            label="Gate",
            description="Gate length as a fraction of each step",
            min_value=0.05,
            max_value=1.0,
            default_value=0.5,
            style=compact,
        )
        self.gate_length_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "gate_length", self.gate_length_knob.get_value()
            )
        )
        knob_row.addWidget(self.gate_length_knob)

        self.transpose_knob = Knob(
            label="Trans",
            description="Transpose the sequence in semitones",
            min_value=-24.0,
            max_value=24.0,
            default_value=0.0,
            style=compact,
        )
        self.transpose_knob.value_changed.connect(self._on_transpose_changed)
        knob_row.addWidget(self.transpose_knob)

        self.swing_knob = Knob(
            label="Swing",
            description="Internal clock swing amount",
            min_value=0.0,
            max_value=0.75,
            default_value=0.0,
            style=compact,
        )
        self.swing_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("swing", self.swing_knob.get_value())
        )
        knob_row.addWidget(self.swing_knob)
        layout.addLayout(knob_row)

        # --- Clock ---
        clock_row = QHBoxLayout()
        clock_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bpm_knob = Knob(
            label="BPM",
            description="Internal clock tempo (ignored when Clock is patched)",
            min_value=30.0,
            max_value=300.0,
            default_value=120.0,
            style=compact,
        )
        self.bpm_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("bpm", self.bpm_knob.get_value())
        )
        clock_row.addWidget(self.bpm_knob)

        clock_row.addWidget(QLabel("Div:"))
        self.division_combo = QComboBox()
        self.division_combo.addItems(["1/4", "1/8", "1/16", "1/32"])
        self.division_combo.setCurrentText("1/16")
        self.division_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("division", value)
        )
        clock_row.addWidget(self.division_combo)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Reset pattern position and clock phase")
        self.reset_button.setMaximumWidth(56)
        self.reset_button.clicked.connect(self._on_reset_clicked)
        clock_row.addWidget(self.reset_button)
        layout.addLayout(clock_row)

        # --- Playhead display ---
        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(6)
        playhead_style = LedStyle(
            size=10,
            off_color=QColor(40, 50, 60),
            on_color=QColor(80, 220, 255),
            border_color=QColor(20, 30, 40),
        )
        self.play_led = LedIndicator(style=playhead_style)
        self.play_led.set_on(False)
        status_grid.addWidget(
            self.play_led, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.note_label = QLabel("--")
        self.note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note_label.setStyleSheet(
            "color: #9fe8ff; font-size: 13px; font-weight: bold;"
        )
        status_grid.addWidget(self.note_label, 0, 1)

        self.sequence_label = QLabel(self._format_sequence_preview())
        self.sequence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sequence_label.setWordWrap(True)
        self.sequence_label.setStyleSheet("color: #a0b0c0; font-size: 10px;")
        self.sequence_label.setToolTip("Expanded pattern notes after octaves/transpose")
        status_grid.addWidget(self.sequence_label, 1, 0, 1, 2)
        layout.addLayout(status_grid)

        self._finish_controls(layout)

        self.register_parameter(
            "notes", self.notes_edit, getter="text", setter="setText"
        )
        self.register_parameter(
            "root", self.root_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter(
            "root_octave",
            self.root_octave_combo,
            getter="currentText",
            setter="setCurrentText",
        )
        self.register_parameter(
            "chord", self.chord_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter(
            "pattern", self.pattern_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter(
            "octaves", self, getter="get_octaves", setter="set_octaves"
        )
        self.register_parameter("gate_length", self.gate_length_knob)
        self.register_parameter(
            "transpose", self, getter="get_transpose", setter="set_transpose"
        )
        self.register_parameter("swing", self.swing_knob)
        self.register_parameter("bpm", self.bpm_knob)
        self.register_parameter(
            "division",
            self.division_combo,
            getter="currentText",
            setter="setCurrentText",
        )
        self.register_parameter(
            "latch", self.latch_checkbox, getter="isChecked", setter="setChecked"
        )
        self.register_parameter(
            "running", self.run_checkbox, getter="isChecked", setter="setChecked"
        )

        self._install_sample_rate_listener()
        self._refresh_sequence_preview()

    # ------------------------------------------------------------------ params

    def get_octaves(self) -> int:
        return int(round(self.octaves_knob.get_value()))

    def set_octaves(self, value: int | float) -> None:
        self.octaves_knob.set_value(float(max(1, min(4, int(round(float(value)))))))

    def get_transpose(self) -> int:
        return int(round(self.transpose_knob.get_value()))

    def set_transpose(self, value: int | float) -> None:
        self.transpose_knob.set_value(
            float(max(-24, min(24, int(round(float(value))))))
        )

    def _on_notes_changed(self, value: str) -> None:
        self.parameter_changed.emit("notes", value)
        self._refresh_sequence_preview()

    def _on_pattern_changed(self, value: str) -> None:
        self.parameter_changed.emit("pattern", value)
        self._refresh_sequence_preview()

    def _on_latch_toggled(self, value: bool) -> None:
        self.parameter_changed.emit("latch", value)
        self._refresh_sequence_preview()

    def _on_octaves_changed(self) -> None:
        self.parameter_changed.emit("octaves", self.get_octaves())
        self._refresh_sequence_preview()

    def _on_transpose_changed(self) -> None:
        self.parameter_changed.emit("transpose", self.get_transpose())
        self._refresh_sequence_preview()

    def _on_apply_chord_clicked(self) -> None:
        self.apply_chord()

    def apply_chord(
        self,
        root: str | None = None,
        octave: int | None = None,
        quality: str | None = None,
    ) -> str:
        """Build MIDI notes from root/octave/quality and write them into Notes.

        Returns the notes text that was applied.
        """
        root_name = root if root is not None else self.root_combo.currentText()
        if octave is None:
            try:
                octave = int(self.root_octave_combo.currentText())
            except ValueError:
                octave = 4
        chord_quality = (
            quality if quality is not None else self.chord_combo.currentText()
        )
        intervals = _CHORD_INTERVALS.get(chord_quality, (0, 4, 7))
        root_pc = _NOTE_NAMES.index(root_name) if root_name in _NOTE_NAMES else 0
        root_midi = max(0, min(127, (int(octave) + 1) * 12 + root_pc))
        notes = [max(0, min(127, root_midi + interval)) for interval in intervals]
        text = ",".join(str(n) for n in notes)
        self.notes_edit.setText(text)
        return text

    def _on_reset_clicked(self) -> None:
        self.component.reset()
        self._displayed_step = -1
        self.play_led.set_on(False)
        self.note_label.setText("--")

    def _format_sequence_preview(self) -> str:
        notes = self.component.sequence_notes
        if not notes:
            return "(no notes)"
        labels = [_midi_label(n) for n in notes[:16]]
        suffix = "…" if len(notes) > 16 else ""
        return " → ".join(labels) + suffix

    def _refresh_sequence_preview(self) -> None:
        # Keep component in sync so the preview reflects current knobs/notes.
        notes = _parse_notes(self.notes_edit.text())
        self.component.set_notes(notes)
        self.component.pattern = self._pattern_value(self.pattern_combo.currentText())
        self.component.octaves = self.get_octaves()
        self.component.transpose = self.get_transpose()
        self.component.latch = self.latch_checkbox.isChecked()
        self.sequence_label.setText(self._format_sequence_preview())

    def _on_playhead_changed(self, step: int, label: str) -> None:
        del step  # already tracked on the audio path for emit dedupe
        self.play_led.set_on(bool(label) and label != "--")
        self.note_label.setText(label if label else "--")

    @staticmethod
    def _pattern_value(label: str) -> str:
        return _LABEL_TO_PATTERN.get(label, label.strip().lower().replace(" ", "_"))

    @staticmethod
    def _pattern_label(value: str) -> str:
        key = value.strip().lower().replace(" ", "_").replace("-", "_")
        return _PATTERN_TO_LABEL.get(key, "Up")

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component.sample_rate = new_sample_rate
        self.component.clock.sample_rate = new_sample_rate
        self.component.reset()
        self._previous_clock = 0.0
        self._previous_reset = 0.0

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        notes_text = str_parameter(parameters, "notes", self.notes_edit.text)
        if notes_text != self._previous_notes:
            self.component.set_notes(_parse_notes(notes_text))
            self._previous_notes = notes_text

        pattern_label = str_parameter(
            parameters, "pattern", self.pattern_combo.currentText
        )
        self.component.pattern = self._pattern_value(pattern_label)

        octaves_raw = parameters.get("octaves", self.get_octaves())
        try:
            self.component.octaves = max(
                1, min(4, int(round(float(cast(float | int | str, octaves_raw)))))
            )
        except (TypeError, ValueError):
            self.component.octaves = self.get_octaves()

        self.component.gate_length = float_parameter(
            parameters, "gate_length", self.gate_length_knob.get_value
        )

        transpose_raw = parameters.get("transpose", self.get_transpose())
        try:
            self.component.transpose = int(
                round(float(cast(float | int | str, transpose_raw)))
            )
        except (TypeError, ValueError):
            self.component.transpose = self.get_transpose()

        latch = bool(parameters.get("latch", self.latch_checkbox.isChecked()))
        self.component.latch = latch

        self.component.configure_clock(
            bpm=float_parameter(parameters, "bpm", self.bpm_knob.get_value),
            division=str_parameter(
                parameters, "division", self.division_combo.currentText
            ),
            swing=float_parameter(parameters, "swing", self.swing_knob.get_value),
        )

        if self.reset_input.is_connected:
            reset_signal = read_samples(self.reset_input, num_samples)
            current_reset = float(reset_signal[0]) if len(reset_signal) else 0.0
            if self._previous_reset < 0.3 and current_reset > 0.7:
                self.component.reset()
            self._previous_reset = current_reset

        running = bool(parameters.get("running", self.run_checkbox.isChecked()))
        # Collapse multi-sample Clock triggers to one edge per step.
        if self.clock_input.is_connected:
            clock_pulses, self._previous_clock = rising_edge_pulses(
                read_samples(self.clock_input, num_samples),
                self._previous_clock,
            )
        else:
            self._previous_clock = 0.0
            clock_pulses = None
        frame = self.component.process(num_samples, clock_pulses, running=running)

        # Pitch CV (1V/oct, 0V = C4) — same convention as MIDI modules / VCO V/Oct.
        self.freq_port.write(frame.pitch_cv)
        self.gate_port.write(frame.gate)
        self.trigger_port.write(frame.trigger)
        self.vel_port.write(frame.velocity)

        # Defer UI updates off the audio path; mark displayed state before emit
        # so we do not re-queue the same step every buffer.
        current_note = self.component._current_note  # noqa: SLF001 — display only
        step_index = int(self.component._step_index)  # noqa: SLF001
        label = "--" if current_note is None else _midi_label(int(current_note))
        if step_index != self._displayed_step or label != self._displayed_note:
            self._displayed_step = step_index
            self._displayed_note = label
            self.playhead_changed.emit(step_index, label)
