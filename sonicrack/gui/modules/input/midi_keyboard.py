"""On-screen MIDI keyboard module for local note input."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QKeyEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QGraphicsItem,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)
from soniclab.midi_io import (
    CVFrequencyOutput,
    CVGateOutput,
    CVVelocityOutput,
    MIDIToCV,
    MIDITriggerOutput,
    NoteOffMessage,
    NoteOnMessage,
    midi_to_note_name,
)

from sonicrack.gui.modules.input.midi_cv_helpers import priority_from_label
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.specs import RuntimeParameters

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
BLACK_KEY_COLUMNS = {
    1: 1,
    3: 3,
    6: 7,
    8: 9,
    10: 11,
}
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
class MIDIKeyboardModule(ModuleWidget):
    """On-screen monophonic MIDI keyboard with full note-stack priority.

    Multi-key (polyphonic) input is accepted: all held notes stay in the
    soniclab ``MIDIToCV`` stack, and the selected **Priority** (Last / High /
    Low) chooses which note drives pitch, gate, and velocity. Use
    **MIDI Poly CV** when independent per-voice CV is required.
    """

    runtime_kind = "midi"

    metadata = ModuleMetadata(
        title="MIDI Keyboard",
        category=ModuleCategory.MIDI,
        description=(
            "On-screen keyboard with mono note-stack priority "
            "(accepts multi-key / polyphonic input)"
        ),
        version="1.1.0",
        author="SonicRack",
    )

    def __init__(self) -> None:
        super().__init__(
            width=385,
            height=285,
            color=QColor(190, 120, 80),
        )

        self.freq_port = self.add_output("1V/Oct", signal=PortSignal.PITCH_CV)
        self.gate_port = self.add_output("Gate", signal=PortSignal.GATE)
        self.trigger_port = self.add_output("Trig", signal=PortSignal.TRIGGER)
        self.vel_port = self.add_output("Vel", signal=PortSignal.CONTROL_CV)

        self.cv_converter = MIDIToCV(note_priority="last")
        self.freq_output = CVFrequencyOutput(self.cv_converter)
        self.gate_output = CVGateOutput(self.cv_converter)
        self.trigger_output = MIDITriggerOutput()
        self.vel_output = CVVelocityOutput(self.cv_converter)
        # offset → MIDI note for UI key state (supports multi-key hold)
        self._pressed_note_offsets: dict[int, int] = {}

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout = self._begin_controls(spacing=8)

        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Octave:"))
        self.octave_combo = QComboBox()
        self.octave_combo.addItems([str(octave) for octave in range(1, 8)])
        self.octave_combo.setCurrentText("4")
        self.octave_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("octave", value)
        )
        settings_layout.addWidget(self.octave_combo)

        self.velocity_knob = Knob(
            label="Velocity",
            description="Sets the MIDI velocity for notes played on the keyboard",
            min_value=0,
            max_value=127,
            default_value=100,
        )
        self.velocity_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "velocity", int(self.velocity_knob.get_value())
            )
        )
        settings_layout.addWidget(self.velocity_knob)
        layout.addLayout(settings_layout)

        self.note_label = QLabel("--")
        self.note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note_label.setStyleSheet(
            "color: white; font-size: 14px; font-weight: bold;"
        )
        layout.addWidget(self.note_label)

        keyboard_layout = QGridLayout()
        keyboard_layout.setHorizontalSpacing(2)
        keyboard_layout.setVerticalSpacing(2)
        self.key_buttons: dict[int, QPushButton] = {}
        self._add_keyboard_buttons(keyboard_layout)
        layout.addLayout(keyboard_layout)

        self._finish_controls(layout)

        # Setup option: right-click module header → Priority
        self.priority_param = self.register_menu_choice(
            "priority",
            "Priority",
            ["Last", "High", "Low"],
            "Last",
            on_changed=self._on_priority_changed,
            tooltip="Which held key drives pitch/gate when several are down",
        )

        self.register_parameter(
            "octave",
            self.octave_combo,
            getter="currentText",
            setter="setCurrentText",
        )
        self.register_parameter("velocity", self.velocity_knob)

    def _on_priority_changed(self, label: str) -> None:
        self.cv_converter.note_priority = priority_from_label(label)
        if self.cv_converter.held_notes:
            self.cv_converter._update_from_stack()  # noqa: SLF001
            self._refresh_note_display()

    def _add_keyboard_buttons(self, layout: QGridLayout) -> None:
        for index, note_offset in enumerate(WHITE_KEYS):
            button = self._create_key_button(note_offset, is_black=False)
            layout.addWidget(button, 1, index * 2, 1, 2)

        for note_offset in BLACK_KEYS:
            button = self._create_key_button(note_offset, is_black=True)
            layout.addWidget(button, 0, BLACK_KEY_COLUMNS[note_offset], 1, 2)

    def _create_key_button(self, note_offset: int, *, is_black: bool) -> QPushButton:
        button = QPushButton(NOTE_LABELS[note_offset])
        button.setMinimumHeight(42 if is_black else 58)
        button.setCheckable(False)
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
            QPushButton:pressed {
                background-color: #2f85c8;
            }
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
            QPushButton:pressed {
                background-color: #6fc7ff;
            }
            """
        )
        button.pressed.connect(lambda offset=note_offset: self._note_on(offset))
        button.released.connect(lambda offset=note_offset: self._note_off(offset))
        self.key_buttons[note_offset] = button
        return button

    def _base_note(self) -> int:
        octave = int(self.octave_combo.currentText())
        return (octave + 1) * 12

    def _midi_note_for_offset(self, note_offset: int) -> int:
        return max(0, min(127, self._base_note() + note_offset))

    def _refresh_note_display(self) -> None:
        note = self.cv_converter.current_note
        if note is not None and self.cv_converter.gate > 0.0:
            held = len(self.cv_converter.held_notes)
            suffix = f" +{held - 1}" if held > 1 else ""
            self.note_label.setText(f"{midi_to_note_name(note)} ({note}){suffix}")
            self.note_label.setStyleSheet(
                "color: #00ff00; font-size: 14px; font-weight: bold;"
            )
        else:
            self.note_label.setText("--")
            self.note_label.setStyleSheet(
                "color: white; font-size: 14px; font-weight: bold;"
            )

    def _note_on(self, note_offset: int) -> None:
        note = self._midi_note_for_offset(note_offset)
        velocity = int(self.velocity_knob.get_value())
        prior_note = self.cv_converter.current_note
        prior_gate = self.cv_converter.gate

        self._pressed_note_offsets[note_offset] = note
        self.cv_converter.process_message(
            NoteOnMessage(timestamp=0.0, channel=0, note=note, velocity=velocity)
        )

        # Arm Trig when priority note becomes active or changes (legato / re-press).
        if (
            self.cv_converter.gate > 0.0
            and self.cv_converter.current_note is not None
            and (prior_gate == 0.0 or self.cv_converter.current_note != prior_note)
        ):
            self.trigger_output.arm()

        self.key_buttons[note_offset].setDown(True)
        self._refresh_note_display()

    def _note_off(self, note_offset: int) -> None:
        note = self._pressed_note_offsets.pop(
            note_offset, self._midi_note_for_offset(note_offset)
        )
        # Let MIDIToCV's note stack select the next priority note — do not
        # re-inject NoteOn (that would break high/low priority and stack order).
        self.cv_converter.process_message(
            NoteOffMessage(timestamp=0.0, channel=0, note=note)
        )
        self.key_buttons[note_offset].setDown(False)
        self._refresh_note_display()

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
        }
        return outputs.get(port_name, self.freq_output)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        priority = parameters.get("priority") if parameters else None
        if isinstance(priority, str):
            wanted = priority_from_label(priority)
            if self.cv_converter.note_priority != wanted:
                self.cv_converter.note_priority = wanted

        self.freq_port.write(self.freq_output.get_samples(num_samples))
        self.gate_port.write(self.gate_output.get_samples(num_samples))
        self.trigger_port.write(self.trigger_output.get_samples(num_samples))
        self.vel_port.write(self.vel_output.get_samples(num_samples))

    def set_active(self, active: bool) -> None:
        super().set_active(active)
        if not active:
            for note_offset in self._pressed_note_offsets:
                self.key_buttons[note_offset].setDown(False)
            self._pressed_note_offsets.clear()
            self.trigger_output.reset()
            self.cv_converter.reset()
            self.note_label.setText("--")
