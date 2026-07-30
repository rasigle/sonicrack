"""Monophonic step sequencer module."""

from __future__ import annotations

import random

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)
from soniclab.sequencing import TB303StepEvent, TB303StepSequencer

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import (
    ImageButtonStyle,
    ImagePushButton,
    Knob,
    LedIndicator,
    LedStyle,
    ProceduralKnobStyle,
)
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import (
    float_parameter,
    read_samples,
    str_parameter,
)
from sonicrack.runtime.specs import RuntimeParameters

# Bass-range minor scale offsets used when randomizing notes (root MIDI 36 / C2).
_RANDOMIZE_ROOT = 36
_RANDOMIZE_SCALE_INTERVALS = (0, 2, 3, 5, 7, 8, 10, 12, 15)
_REST_PROBABILITY = 0.2
_ACCENT_PROBABILITY = 0.3
_SLIDE_PROBABILITY = 0.25
_DEFAULT_ACCENTS = (True, False, False, True, False, False, True, False)
_DEFAULT_SLIDES = (False, False, True, False, False, False, True, False)


@register_module()
class StepSequencerModule(ModuleWidget):
    """Compact monophonic sequencer for pitch, gate, accent, and slide CV."""

    runtime_kind = "step_sequencer"
    step_toggle_count = 8

    metadata = ModuleMetadata(
        title="Step Sequencer",
        category=ModuleCategory.SEQUENCER,
        description="Monophonic pattern sequencer with accent and slide outputs",
    )

    def __init__(self) -> None:
        super().__init__(width=340, height=340, color=QColor(120, 100, 170))

        self.clock_input = self.add_input("Clock", signal=PortSignal.TRIGGER)
        self.reset_input = self.add_input("Reset", signal=PortSignal.TRIGGER)
        self.freq_port = self.add_output("Freq", signal=PortSignal.PITCH_CV)
        self.gate_port = self.add_output("Gate", signal=PortSignal.GATE)
        self.accent_port = self.add_output("Accent", signal=PortSignal.CONTROL_CV)
        self.slide_port = self.add_output("Slide", signal=PortSignal.CONTROL_CV)

        self.component = TB303StepSequencer(sample_rate=audio_config.sample_rate)
        self._previous_pattern_key: tuple[str, str, str, str, float] | None = None
        self._previous_reset = 0.0
        self._syncing_step_toggles = False
        self.accent_buttons: list[ImagePushButton] = []
        self.accent_leds: list[LedIndicator] = []
        self.slide_buttons: list[ImagePushButton] = []
        self.slide_leds: list[LedIndicator] = []

        layout = self._begin_controls(spacing=6)

        self.notes_edit = QLineEdit("36,-,36,39,41,-,39,36")
        self.notes_edit.setToolTip("Comma-separated MIDI notes. Use '-' for rests.")
        self.notes_edit.textChanged.connect(
            lambda value: self.parameter_changed.emit("notes", value)
        )
        layout.addWidget(QLabel("Notes:"))
        layout.addWidget(self.notes_edit)
        layout.addLayout(self._create_step_toggle_grid())

        randomize_layout = QHBoxLayout()
        randomize_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.randomize_button = QPushButton("Randomize")
        self.randomize_button.setMinimumHeight(28)
        self.randomize_button.setToolTip(
            "Randomize notes, accents, and slides for a new pattern"
        )
        self.randomize_button.setStyleSheet("""
            QPushButton {
                background-color: #6b5a9a;
                color: white;
                border: 2px solid #4a3d6e;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
                padding: 2px 12px;
            }
            QPushButton:hover {
                background-color: #7d6aad;
            }
            QPushButton:pressed {
                background-color: #55487a;
            }
        """)
        self.randomize_button.clicked.connect(self._on_randomize_clicked)
        randomize_layout.addWidget(self.randomize_button)
        layout.addLayout(randomize_layout)

        clock_layout = QHBoxLayout()
        compact_knob_style = ProceduralKnobStyle.small()
        self.bpm_knob = Knob(
            label="BPM",
            description="Sets the tempo of the sequencer in beats per minute",
            min_value=20.0,
            max_value=300.0,
            default_value=120.0,
            style=compact_knob_style,
        )
        self.bpm_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("bpm", self.bpm_knob.get_value())
        )
        clock_layout.addWidget(self.bpm_knob)

        self.gate_length_knob = Knob(
            label="Gate",
            description="Adjusts the length of the gate for each step",
            min_value=0.1,
            max_value=1.0,
            default_value=0.8,
            style=compact_knob_style,
        )
        self.gate_length_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "gate_length", self.gate_length_knob.get_value()
            )
        )
        clock_layout.addWidget(self.gate_length_knob)
        layout.addLayout(clock_layout)

        division_layout = QHBoxLayout()
        division_layout.addWidget(QLabel("Division:"))
        self.division_combo = QComboBox()
        self.division_combo.addItems(["1/4", "1/8", "1/16", "1/32"])
        self.division_combo.setCurrentText("1/16")
        self.division_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("division", value)
        )
        division_layout.addWidget(self.division_combo)
        layout.addLayout(division_layout)

        self._finish_controls(layout)

        self.register_parameter(
            "notes", self.notes_edit, getter="text", setter="setText"
        )
        self.register_parameter(
            "accents", self, getter="get_accents", setter="set_accents"
        )
        self.register_parameter(
            "slides", self, getter="get_slides", setter="set_slides"
        )
        self.register_parameter("bpm", self.bpm_knob)
        self.register_parameter("gate_length", self.gate_length_knob)
        self.register_parameter(
            "division",
            self.division_combo,
            getter="currentText",
            setter="setCurrentText",
        )

        self._install_sample_rate_listener()

    def _create_step_toggle_grid(self) -> QGridLayout:
        toggle_grid = QGridLayout()
        toggle_grid.setHorizontalSpacing(3)
        toggle_grid.setVerticalSpacing(2)

        button_style = ImageButtonStyle(size=22)
        accent_led_style = LedStyle(size=8)
        slide_led_style = LedStyle(size=8)

        toggle_grid.addWidget(QLabel("Acc"), 0, 0)
        toggle_grid.addWidget(QLabel("Sld"), 1, 0)
        for index in range(self.step_toggle_count):
            accent_on = _DEFAULT_ACCENTS[index]
            slide_on = _DEFAULT_SLIDES[index]

            accent_button = ImagePushButton(
                str(index + 1),
                style=button_style,
                checkable=True,
            )
            accent_button.setChecked(accent_on)
            accent_button.setToolTip(f"Toggle accent for step {index + 1}")
            accent_button.toggled.connect(
                lambda checked, step=index: self._on_step_toggle_changed(
                    "accent", step, checked
                )
            )
            accent_led = LedIndicator(style=accent_led_style)
            accent_led.set_on(accent_on)

            slide_button = ImagePushButton(
                str(index + 1),
                style=button_style,
                checkable=True,
            )
            slide_button.setChecked(slide_on)
            slide_button.setToolTip(f"Toggle slide for step {index + 1}")
            slide_button.toggled.connect(
                lambda checked, step=index: self._on_step_toggle_changed(
                    "slide", step, checked
                )
            )
            slide_led = LedIndicator(style=slide_led_style)
            slide_led.set_on(slide_on)

            self.accent_buttons.append(accent_button)
            self.accent_leds.append(accent_led)
            self.slide_buttons.append(slide_button)
            self.slide_leds.append(slide_led)

            accent_cell = QHBoxLayout()
            accent_cell.setSpacing(1)
            accent_cell.addWidget(accent_led)
            accent_cell.addWidget(accent_button)
            toggle_grid.addLayout(accent_cell, 0, index + 1)

            slide_cell = QHBoxLayout()
            slide_cell.setSpacing(1)
            slide_cell.addWidget(slide_led)
            slide_cell.addWidget(slide_button)
            toggle_grid.addLayout(slide_cell, 1, index + 1)

        return toggle_grid

    def get_accents(self) -> str:
        """Serialize accent step toggles for parameters and presets."""
        return self._flags_to_text(
            [button.isChecked() for button in self.accent_buttons]
        )

    def set_accents(self, value: str) -> None:
        """Load accent step toggles from a comma-separated flag string."""
        self._apply_flags("accent", value)
        self.parameter_changed.emit("accents", self.get_accents())

    def get_slides(self) -> str:
        """Serialize slide step toggles for parameters and presets."""
        return self._flags_to_text(
            [button.isChecked() for button in self.slide_buttons]
        )

    def set_slides(self, value: str) -> None:
        """Load slide step toggles from a comma-separated flag string."""
        self._apply_flags("slide", value)
        self.parameter_changed.emit("slides", self.get_slides())

    def _on_randomize_clicked(self) -> None:
        self.randomize_pattern()

    def randomize_pattern(self, rng: random.Random | None = None) -> None:
        """Fill notes, accents, and slides with a new random pattern.

        Args:
            rng: Optional random generator for deterministic tests.
        """
        generator = rng if rng is not None else random.Random()
        notes: list[str] = []
        accents: list[bool] = []
        slides: list[bool] = []

        for _ in range(self.step_toggle_count):
            if generator.random() < _REST_PROBABILITY:
                notes.append("-")
                accents.append(False)
                slides.append(False)
                continue

            interval = generator.choice(_RANDOMIZE_SCALE_INTERVALS)
            notes.append(str(_RANDOMIZE_ROOT + interval))
            accents.append(generator.random() < _ACCENT_PROBABILITY)
            slides.append(generator.random() < _SLIDE_PROBABILITY)

        self.notes_edit.setText(",".join(notes))
        self.set_accents(self._flags_to_text(accents))
        self.set_slides(self._flags_to_text(slides))

    def _on_step_toggle_changed(self, kind: str, step: int, checked: bool) -> None:
        if self._syncing_step_toggles:
            return
        leds = self.accent_leds if kind == "accent" else self.slide_leds
        leds[step].set_on(checked)
        if kind == "accent":
            self.parameter_changed.emit("accents", self.get_accents())
        else:
            self.parameter_changed.emit("slides", self.get_slides())

    def _apply_flags(self, kind: str, text: str) -> None:
        flags = self._parse_flags(text, self.step_toggle_count)
        buttons = self.accent_buttons if kind == "accent" else self.slide_buttons
        leds = self.accent_leds if kind == "accent" else self.slide_leds
        self._syncing_step_toggles = True
        try:
            for index, flag in enumerate(flags):
                buttons[index].setChecked(flag)
                leds[index].set_on(flag)
        finally:
            self._syncing_step_toggles = False

    @staticmethod
    def _flags_to_text(flags: list[bool]) -> str:
        return ",".join("1" if flag else "0" for flag in flags)

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component.sample_rate = new_sample_rate
        self.component.clock.sample_rate = new_sample_rate
        self.component.reset()


    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        notes = str_parameter(parameters, "notes", self.notes_edit.text)
        accents = str_parameter(parameters, "accents", self.get_accents)
        slides = str_parameter(parameters, "slides", self.get_slides)
        gate_length = float_parameter(
            parameters, "gate_length", self.gate_length_knob.get_value
        )
        pattern_key = (notes, accents, slides, "", gate_length)
        if pattern_key != self._previous_pattern_key:
            self.component.pattern = self._parse_pattern(
                notes, accents, slides, gate_length
            )
            self.component.reset()
            self._previous_pattern_key = pattern_key

        self.component.configure_clock(
            bpm=float_parameter(parameters, "bpm", self.bpm_knob.get_value),
            division=str_parameter(
                parameters, "division", self.division_combo.currentText
            ),
            swing=0.0,
        )

        if self.reset_input.is_connected:
            reset_signal = read_samples(self.reset_input, num_samples)
            current_reset = float(reset_signal[0]) if len(reset_signal) else 0.0
            if self._previous_reset < 0.3 and current_reset > 0.7:
                self.component.reset()
            self._previous_reset = current_reset

        clock_pulses = (
            read_samples(self.clock_input, num_samples)
            if self.clock_input.is_connected
            else None
        )
        frame = self.component.process(num_samples, clock_pulses)
        self.freq_port.write(frame.frequency)
        self.gate_port.write(frame.gate)
        self.accent_port.write(frame.accent)
        self.slide_port.write(frame.slide)

    @staticmethod
    def _parse_pattern(
        notes_text: str,
        accents_text: str,
        slides_text: str,
        gate_length: float,
    ) -> list[TB303StepEvent]:
        notes = [item.strip() for item in notes_text.split(",") if item.strip()]
        accents = StepSequencerModule._parse_flags(accents_text, len(notes))
        slides = StepSequencerModule._parse_flags(slides_text, len(notes))
        pattern: list[TB303StepEvent] = []
        for index, item in enumerate(notes):
            if item in {"-", "r", "R", "rest", "Rest"}:
                note = None
                gate = False
            else:
                try:
                    note = int(item)
                    gate = True
                except ValueError:
                    note = None
                    gate = False
            pattern.append(
                TB303StepEvent(
                    note=note,
                    gate=gate,
                    accent=accents[index],
                    slide=slides[index],
                    gate_length=gate_length,
                )
            )
        return pattern or [TB303StepEvent(None, gate=False)]

    @staticmethod
    def _parse_flags(text: str, length: int) -> list[bool]:
        raw = [item.strip().lower() for item in text.split(",") if item.strip()]
        flags = [item in {"1", "true", "t", "yes", "y", "x"} for item in raw]
        if len(flags) < length:
            flags.extend([False] * (length - len(flags)))
        return flags[:length]
