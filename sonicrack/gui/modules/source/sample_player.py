"""Triggered sample player module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton
from soniclab.generators.sample_player import SamplePlayer
from soniclab.utils.cv import pitch_cv_to_frequency

from sonicrack.config.app_settings import app_settings
from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import (
    as_mono,
    float_parameter,
    gate_transition_indices,
    read_samples,
    str_parameter,
)
from sonicrack.runtime.specs import RuntimeParameters


def _default_pluck(sample_rate: float, seconds: float = 0.55) -> np.ndarray:
    """Built-in decaying sine so the module sounds without a loaded file."""
    n = max(16, int(float(sample_rate) * seconds))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    env = np.exp(-t * 7.5).astype(np.float32)
    return (np.sin(2.0 * np.pi * 220.0 * t) * env).astype(np.float32)


class _PathParameter:
    """String path stored as a patch parameter."""

    def __init__(self, on_changed) -> None:
        self._value = ""
        self._on_changed = on_changed

    def get_value(self) -> str:
        return self._value

    def set_value(self, value: object) -> None:
        path = "" if value is None else str(value)
        if path == self._value:
            return
        self._value = path
        self._on_changed(path)


@register_module()
class SamplePlayerModule(ModuleWidget):
    """One-shot / loop sample player with gate and 1V/oct pitch."""

    runtime_kind = "sample_player"

    metadata = ModuleMetadata(
        title="Sample Player",
        category=ModuleCategory.SOURCE,
        description="Triggered WAV player with pitch, loop, and gate",
    )

    def __init__(self) -> None:
        super().__init__(width=230, height=270, color=QColor(80, 140, 150))
        self.freq_input = self.add_input("Freq", signal=PortSignal.PITCH_CV)
        self.gate_input = self.add_input("Gate", signal=PortSignal.GATE)
        self.out_port = self.add_output("Out", signal=PortSignal.AUDIO)

        self._file_path = _PathParameter(self._load_from_path)
        self._previous_gate = 0.0
        self._autoplay_when_ungated = True
        sample_rate = audio_config.sample_rate
        self.component = SamplePlayer(
            samples=_default_pluck(sample_rate),
            sample_rate=sample_rate,
            source_sample_rate=sample_rate,
            amplitude=0.7,
            frequency=440.0,
            root_frequency=440.0,
            autoplay=True,
        )

        layout = self._begin_controls(spacing=6)
        self.file_label = QLabel("Built-in pluck")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet("color: #c5d0d8; font-size: 10px;")
        layout.addWidget(self.file_label)

        load_row = QHBoxLayout()
        self.load_button = QPushButton("Load WAV")
        self.load_button.setToolTip("Load a WAV file into the player")
        self.load_button.clicked.connect(self._choose_file)
        load_row.addWidget(self.load_button)
        layout.addLayout(load_row)

        knobs = QHBoxLayout()
        self.pitch_knob = Knob(
            label="Pitch",
            description="Playback pitch in Hz at root",
            min_value=20.0,
            max_value=4000.0,
            default_value=440.0,
            logarithmic=True,
        )
        self.pitch_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "frequency", self.pitch_knob.get_value()
            )
        )
        knobs.addWidget(self.pitch_knob)

        self.level_knob = Knob(
            label="Level",
            description="Output level",
            min_value=0.0,
            max_value=1.0,
            default_value=0.7,
        )
        self.level_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("level", self.level_knob.get_value())
        )
        knobs.addWidget(self.level_knob)
        layout.addLayout(knobs)
        self._finish_controls(layout)

        self.register_parameter("frequency", self.pitch_knob)
        self.register_parameter("level", self.level_knob)
        self.register_parameter("path", self._file_path)
        self.loop_choice = self.register_menu_choice(
            "loop",
            "Loop",
            ("Off", "On"),
            "Off",
            tooltip="Loop the buffer while Gate is high",
        )
        self._install_sample_rate_listener()

    def _choose_file(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        start_dir = app_settings.file_dialog_start_dir(self._file_path.get_value())
        path, _filter = QFileDialog.getOpenFileName(
            None,
            "Load Sample",
            start_dir,
            "WAV files (*.wav);;All files (*.*)",
        )
        if not path:
            return
        app_settings.remember_file_directory(path)
        self._file_path.set_value(path)
        self.parameter_changed.emit("path", path)

    def _load_from_path(self, path: str) -> None:
        if not path:
            self._reset_default_buffer()
            return
        file_path = Path(path)
        if not file_path.is_file():
            self.file_label.setText(f"Missing: {file_path.name}")
            return
        try:
            self.component.load(file_path)
        except Exception as exc:  # noqa: BLE001 — surface load errors on the panel
            self.file_label.setText(f"Load failed: {exc}")
            return
        self.file_label.setText(file_path.name)
        self._autoplay_when_ungated = False
        if not self.gate_input.is_connected:
            self.component.trigger_note_on()

    def _reset_default_buffer(self) -> None:
        sample_rate = audio_config.sample_rate
        self.component.set_samples(
            _default_pluck(sample_rate),
            source_sample_rate=sample_rate,
        )
        self.file_label.setText("Built-in pluck")
        self._autoplay_when_ungated = True
        if not self.gate_input.is_connected:
            self.component.trigger_note_on()

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        path = self._file_path.get_value()
        loop = self.loop_choice.get_value() == "On"
        self.component = SamplePlayer(
            sample_rate=new_sample_rate,
            amplitude=self.level_knob.get_value(),
            frequency=self.pitch_knob.get_value(),
            root_frequency=440.0,
            loop=loop,
        )
        if path:
            self._load_from_path(path)
        else:
            self._reset_default_buffer()

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        loop = str_parameter(parameters, "loop", self.loop_choice.get_value) == "On"
        self.component.loop = loop
        self.component.amplitude = float_parameter(
            parameters, "level", self.level_knob.get_value
        )
        base_freq = float_parameter(
            parameters, "frequency", self.pitch_knob.get_value
        )
        if self.freq_input.is_connected:
            pitch_cv = float(as_mono(read_samples(self.freq_input, num_samples))[0])
            self.component.frequency = float(pitch_cv_to_frequency(pitch_cv))
        else:
            self.component.frequency = base_freq

        if self.gate_input.is_connected:
            gate = as_mono(read_samples(self.gate_input, num_samples))
            ons, offs, final = gate_transition_indices(gate, self._previous_gate)
            self._previous_gate = final
            if ons.size:
                self.component.trigger_note_on()
            elif offs.size:
                self.component.trigger_note_off()
        elif self._autoplay_when_ungated and not self.component.is_playing:
            if loop or not self.component.ended:
                self.component.trigger_note_on()

        self.out_port.write(self.component.get_samples(num_samples))
