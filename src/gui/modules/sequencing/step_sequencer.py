"""Monophonic step sequencer module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit

from src.engine.sequencing import TB303StepEvent, TB303StepSequencer
from src.gui.audio_config import audio_config
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, read_samples, str_parameter
from src.gui.module_registry import register_module
from src.gui.widgets import (
    ImageButtonStyle,
    ImagePushButton,
    Knob,
    LedIndicator,
    LedStyle,
    ProceduralKnobStyle,
)
from src.gui.widgets.module_widget import ModuleWidget


@register_module()
class StepSequencerModule(ModuleWidget):
    """Compact monophonic sequencer for pitch, gate, accent, and slide CV."""

    runtime_kind = "step_sequencer"
    step_toggle_count = 8

    metadata = ModuleMetadata(
        title="Step Sequencer",
        category=ModuleCategory.SOURCE,
        description="Monophonic pattern sequencer with accent and slide outputs",
    )

    def __init__(self) -> None:
        super().__init__(width=340, height=335, color=QColor(120, 100, 170))

        self.clock_input = self.add_input("Clock")
        self.reset_input = self.add_input("Reset")
        self.freq_port = self.add_output("Freq")
        self.gate_port = self.add_output("Gate")
        self.accent_port = self.add_output("Accent")
        self.slide_port = self.add_output("Slide")

        self.component = TB303StepSequencer(sample_rate=audio_config.sample_rate)
        self._previous_pattern_key: tuple[str, str, str, str, float] | None = None
        self._previous_reset = 0.0
        self._syncing_step_toggles = False
        self.accent_buttons: list[ImagePushButton] = []
        self.accent_leds: list[LedIndicator] = []
        self.slide_buttons: list[ImagePushButton] = []
        self.slide_leds: list[LedIndicator] = []

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout(spacing=6)

        self.notes_edit = QLineEdit("36,-,36,39,41,-,39,36")
        self.notes_edit.setToolTip("Comma-separated MIDI notes. Use '-' for rests.")
        self.notes_edit.textChanged.connect(
            lambda value: self.parameter_changed.emit("notes", value)
        )
        layout.addWidget(QLabel("Notes:"))
        layout.addWidget(self.notes_edit)

        self.accent_edit = QLineEdit("1,0,0,1,0,0,1,0")
        self.accent_edit.textChanged.connect(self._on_accents_text_changed)
        layout.addWidget(QLabel("Accents:"))
        layout.addWidget(self.accent_edit)

        self.slide_edit = QLineEdit("0,0,1,0,0,0,1,0")
        self.slide_edit.textChanged.connect(self._on_slides_text_changed)
        layout.addWidget(QLabel("Slides:"))
        layout.addWidget(self.slide_edit)
        layout.addLayout(self._create_step_toggle_grid())
        self._sync_step_toggles_from_text()

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

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.register_parameter(
            "notes", self.notes_edit, getter="text", setter="setText"
        )
        self.register_parameter(
            "accents", self.accent_edit, getter="text", setter="setText"
        )
        self.register_parameter(
            "slides", self.slide_edit, getter="text", setter="setText"
        )
        self.register_parameter("bpm", self.bpm_knob)
        self.register_parameter("gate_length", self.gate_length_knob)
        self.register_parameter(
            "division",
            self.division_combo,
            getter="currentText",
            setter="setCurrentText",
        )

        self._sample_rate_listener = self._on_global_sample_rate_changed
        audio_config.add_sample_rate_listener(self._sample_rate_listener)
        self.destroyed.connect(self._cleanup_audio_config_listeners)

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
            accent_button = ImagePushButton(
                str(index + 1),
                style=button_style,
                checkable=True,
            )
            accent_button.setToolTip(f"Toggle accent for step {index + 1}")
            accent_button.toggled.connect(
                lambda checked, step=index: self._on_step_toggle_changed(
                    "accent", step, checked
                )
            )
            accent_led = LedIndicator(style=accent_led_style)

            slide_button = ImagePushButton(
                str(index + 1),
                style=button_style,
                checkable=True,
            )
            slide_button.setToolTip(f"Toggle slide for step {index + 1}")
            slide_button.toggled.connect(
                lambda checked, step=index: self._on_step_toggle_changed(
                    "slide", step, checked
                )
            )
            slide_led = LedIndicator(style=slide_led_style)

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

    def _on_accents_text_changed(self, value: str) -> None:
        self.parameter_changed.emit("accents", value)
        self._sync_step_toggles_from_text()

    def _on_slides_text_changed(self, value: str) -> None:
        self.parameter_changed.emit("slides", value)
        self._sync_step_toggles_from_text()

    def _on_step_toggle_changed(self, kind: str, step: int, checked: bool) -> None:
        if self._syncing_step_toggles:
            return
        edit = self.accent_edit if kind == "accent" else self.slide_edit
        values = self._parse_flags(edit.text(), self.step_toggle_count)
        values[step] = checked
        edit.setText(self._flags_to_text(values))

    def _sync_step_toggles_from_text(self) -> None:
        self._syncing_step_toggles = True
        try:
            accents = self._parse_flags(self.accent_edit.text(), self.step_toggle_count)
            slides = self._parse_flags(self.slide_edit.text(), self.step_toggle_count)
            for index in range(self.step_toggle_count):
                accent_on = accents[index]
                slide_on = slides[index]
                self.accent_buttons[index].setChecked(accent_on)
                self.accent_leds[index].set_on(accent_on)
                self.slide_buttons[index].setChecked(slide_on)
                self.slide_leds[index].set_on(slide_on)
        finally:
            self._syncing_step_toggles = False

    @staticmethod
    def _flags_to_text(flags: list[bool]) -> str:
        return ",".join("1" if flag else "0" for flag in flags)

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component.sample_rate = new_sample_rate
        self.component.clock.sample_rate = new_sample_rate
        self.component.reset()

    def _cleanup_audio_config_listeners(self, *_args: object) -> None:
        audio_config.remove_sample_rate_listener(self._sample_rate_listener)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        notes = str_parameter(parameters, "notes", self.notes_edit.text)
        accents = str_parameter(parameters, "accents", self.accent_edit.text)
        slides = str_parameter(parameters, "slides", self.slide_edit.text)
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
