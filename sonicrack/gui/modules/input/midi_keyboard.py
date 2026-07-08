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
    NoteOffMessage,
    NoteOnMessage,
    midi_to_note_name,
)

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
    """On-screen MIDI keyboard with frequency, gate, and velocity CV outputs.

    Use this module as a local monophonic note source when no external MIDI
    hardware is needed. Patch ``Freq`` into a VCO frequency input, ``Gate`` into
    an ADSR gate input, and optionally ``Vel`` into a VCA CV input for
        velocity-sensitive level control. The generated oscillator signal can be
        routed to audio outputs, passive visualizers, or both in parallel. When
        the module has focus, the computer keyboard layout
        ``A W S E D F T G Y H U J`` plays one chromatic octave from C to B.
    """

    runtime_kind = "midi"

    metadata = ModuleMetadata(
        title="MIDI Keyboard",
        category=ModuleCategory.SOURCE,
        description="On-screen MIDI keyboard source for local patch playing",
        version="1.0.0",
        author="SonicRack",
    )

    def __init__(self) -> None:
        super().__init__(
            width=360,
            height=260,
            color=QColor(190, 120, 80),
        )

        self.freq_port = self.add_output("1V/Oct", signal=PortSignal.PITCH_CV)
        self.gate_port = self.add_output("Gate", signal=PortSignal.GATE)
        self.vel_port = self.add_output("Vel", signal=PortSignal.CONTROL_CV)

        self.cv_converter = MIDIToCV()
        self.freq_output = CVFrequencyOutput(self.cv_converter)
        self.gate_output = CVGateOutput(self.cv_converter)
        self.vel_output = CVVelocityOutput(self.cv_converter)
        self._pressed_note_offsets: dict[int, int] = {}

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout(spacing=8)

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

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.register_parameter(
            "octave",
            self.octave_combo,
            getter="currentText",
            setter="setCurrentText",
        )
        self.register_parameter("velocity", self.velocity_knob)

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

    def _note_on(self, note_offset: int) -> None:
        note = self._midi_note_for_offset(note_offset)
        velocity = int(self.velocity_knob.get_value())
        self._pressed_note_offsets.pop(note_offset, None)
        self._pressed_note_offsets[note_offset] = note
        self.cv_converter.process_message(
            NoteOnMessage(timestamp=0.0, channel=0, note=note, velocity=velocity)
        )
        self.key_buttons[note_offset].setDown(True)
        self.note_label.setText(f"{midi_to_note_name(note)} ({note})")
        self.note_label.setStyleSheet(
            "color: #00ff00; font-size: 14px; font-weight: bold;"
        )

    def _note_off(self, note_offset: int) -> None:
        note = self._pressed_note_offsets.pop(
            note_offset, self._midi_note_for_offset(note_offset)
        )
        self.cv_converter.process_message(
            NoteOffMessage(timestamp=0.0, channel=0, note=note)
        )
        self.key_buttons[note_offset].setDown(False)
        if self._pressed_note_offsets:
            fallback_note = next(reversed(self._pressed_note_offsets.values()))
            self.cv_converter.process_message(
                NoteOnMessage(
                    timestamp=0.0,
                    channel=0,
                    note=fallback_note,
                    velocity=int(self.velocity_knob.get_value()),
                )
            )
            self.note_label.setText(
                f"{midi_to_note_name(fallback_note)} ({fallback_note})"
            )
        else:
            self.note_label.setText("--")
            self.note_label.setStyleSheet(
                "color: white; font-size: 14px; font-weight: bold;"
            )

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
            "Vel": self.vel_output,
        }
        return outputs.get(port_name, self.freq_output)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        del parameters
        self.freq_port.write(self.freq_output.get_samples(num_samples))
        self.gate_port.write(self.gate_output.get_samples(num_samples))
        self.vel_port.write(self.vel_output.get_samples(num_samples))

    def set_active(self, active: bool) -> None:
        super().set_active(active)
        if not active:
            for note_offset in self._pressed_note_offsets:
                self.key_buttons[note_offset].setDown(False)
            self._pressed_note_offsets.clear()
            self.cv_converter.reset()
            self.note_label.setText("--")
