"""Compact subtractive synth voice."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel
from soniclab.utils.cv import pitch_cv_to_frequency
from soniclab.voices import SubtractiveVoice
from soniclab.voices.subtractive import FilterModeName

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import (
    as_mono,
    ensure_min_pulse_width,
    float_parameter,
    min_trigger_samples,
    read_samples,
    silence,
    str_parameter,
)
from sonicrack.runtime.specs import RuntimeParameters

_WAVE_ITEMS = ("Saw", "Square", "Sine", "Triangle")
_WAVE_TO_ENGINE = {
    "Saw": "saw",
    "Square": "square",
    "Sine": "sine",
    "Triangle": "triangle",
}
_FILTER_ITEMS = ("Low", "High", "Band", "Notch")
_FILTER_TO_ENGINE: dict[str, FilterModeName] = {
    "Low": "low",
    "High": "high",
    "Band": "band",
    "Notch": "notch",
}


def _filter_mode(label: str) -> FilterModeName:
    return _FILTER_TO_ENGINE.get(label, "low")


@register_module()
class SubtractiveVoiceModule(ModuleWidget):
    """Osc → SVF → VCA voice with dual envelopes, driven by pitch and gate."""

    runtime_kind = "subtractive_voice"

    metadata = ModuleMetadata(
        title="Subtractive Voice",
        category=ModuleCategory.MODULATED_SOURCE,
        description="Classic osc → filter → amp voice with ADSR",
    )

    def __init__(self) -> None:
        super().__init__(width=300, height=430, color=QColor(70, 120, 165))
        self.freq_input = self.add_input("Freq", signal=PortSignal.PITCH_CV)
        self.gate_input = self.add_input("Gate", signal=PortSignal.GATE)
        self.out_port = self.add_output("Out", signal=PortSignal.AUDIO)
        self.component = SubtractiveVoice(sample_rate=audio_config.sample_rate)
        self._gate_hold = 0
        self._previous_raw_gate = 0.0

        layout = self._begin_controls(spacing=4)

        wave_row = QHBoxLayout()
        wave_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wave_row.addWidget(QLabel("Wave:"))
        self.wave_combo = QComboBox()
        self.wave_combo.addItems(list(_WAVE_ITEMS))
        self.wave_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("waveform", value)
        )
        wave_row.addWidget(self.wave_combo)
        wave_row.addWidget(QLabel("Filt:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(list(_FILTER_ITEMS))
        self.filter_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("filter_mode", value)
        )
        wave_row.addWidget(self.filter_combo)
        layout.addLayout(wave_row)

        filter_row = QHBoxLayout()
        filter_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cutoff_knob = Knob(
            label="Cutoff",
            description="Filter cutoff in Hz",
            min_value=40.0,
            max_value=12000.0,
            default_value=1800.0,
            logarithmic=True,
        )
        self.cutoff_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("cutoff", self.cutoff_knob.get_value())
        )
        filter_row.addWidget(self.cutoff_knob)

        self.resonance_knob = Knob(
            label="Res",
            description="Filter resonance / Q",
            min_value=0.1,
            max_value=12.0,
            default_value=1.2,
            logarithmic=True,
        )
        self.resonance_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "resonance", self.resonance_knob.get_value()
            )
        )
        filter_row.addWidget(self.resonance_knob)

        self.env_mod_knob = Knob(
            label="Env.Mod",
            description="Filter envelope amount in octaves",
            min_value=0.0,
            max_value=6.0,
            default_value=2.5,
        )
        self.env_mod_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "env_amount", self.env_mod_knob.get_value()
            )
        )
        filter_row.addWidget(self.env_mod_knob)
        layout.addLayout(filter_row)

        amp_row = QHBoxLayout()
        amp_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.attack_knob = Knob(
            label="A",
            description="Amplitude attack",
            min_value=0.001,
            max_value=2.0,
            default_value=0.01,
            logarithmic=True,
        )
        self.attack_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("attack", self.attack_knob.get_value())
        )
        amp_row.addWidget(self.attack_knob)

        self.decay_knob = Knob(
            label="D",
            description="Amplitude decay",
            min_value=0.01,
            max_value=2.0,
            default_value=0.18,
            logarithmic=True,
        )
        self.decay_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("decay", self.decay_knob.get_value())
        )
        amp_row.addWidget(self.decay_knob)

        self.sustain_knob = Knob(
            label="S",
            description="Amplitude sustain",
            min_value=0.0,
            max_value=1.0,
            default_value=0.7,
        )
        self.sustain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "sustain", self.sustain_knob.get_value()
            )
        )
        amp_row.addWidget(self.sustain_knob)

        self.release_knob = Knob(
            label="R",
            description="Amplitude release",
            min_value=0.01,
            max_value=3.0,
            default_value=0.28,
            logarithmic=True,
        )
        self.release_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "release", self.release_knob.get_value()
            )
        )
        amp_row.addWidget(self.release_knob)
        layout.addLayout(amp_row)

        motion_row = QHBoxLayout()
        motion_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lfo_rate_knob = Knob(
            label="LFO",
            description="Filter/pitch LFO rate (0 = off)",
            min_value=0.0,
            max_value=12.0,
            default_value=0.0,
        )
        self.lfo_rate_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "lfo_rate", self.lfo_rate_knob.get_value()
            )
        )
        motion_row.addWidget(self.lfo_rate_knob)

        self.lfo_filter_knob = Knob(
            label="Flt.LFO",
            description="LFO to filter in octaves",
            min_value=0.0,
            max_value=3.0,
            default_value=0.0,
        )
        self.lfo_filter_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "lfo_filter", self.lfo_filter_knob.get_value()
            )
        )
        motion_row.addWidget(self.lfo_filter_knob)

        self.volume_knob = Knob(
            label="Level",
            description="Voice output level",
            min_value=0.0,
            max_value=1.0,
            default_value=0.7,
        )
        self.volume_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("volume", self.volume_knob.get_value())
        )
        motion_row.addWidget(self.volume_knob)
        layout.addLayout(motion_row)
        self._finish_controls(layout)

        self.register_parameter(
            "waveform", self.wave_combo, "currentText", "setCurrentText"
        )
        self.register_parameter(
            "filter_mode", self.filter_combo, "currentText", "setCurrentText"
        )
        self.register_parameter("cutoff", self.cutoff_knob)
        self.register_parameter("resonance", self.resonance_knob)
        self.register_parameter("env_amount", self.env_mod_knob)
        self.register_parameter("attack", self.attack_knob)
        self.register_parameter("decay", self.decay_knob)
        self.register_parameter("sustain", self.sustain_knob)
        self.register_parameter("release", self.release_knob)
        self.register_parameter("lfo_rate", self.lfo_rate_knob)
        self.register_parameter("lfo_filter", self.lfo_filter_knob)
        self.register_parameter("volume", self.volume_knob)
        self._install_sample_rate_listener()

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component = SubtractiveVoice(
            waveform=_WAVE_TO_ENGINE.get(self.wave_combo.currentText(), "saw"),
            cutoff=self.cutoff_knob.get_value(),
            resonance=self.resonance_knob.get_value(),
            filter_mode=_filter_mode(self.filter_combo.currentText()),
            env_amount=self.env_mod_knob.get_value(),
            attack=self.attack_knob.get_value(),
            decay=self.decay_knob.get_value(),
            sustain=self.sustain_knob.get_value(),
            release=self.release_knob.get_value(),
            lfo_rate_hz=self.lfo_rate_knob.get_value(),
            lfo_filter_depth=self.lfo_filter_knob.get_value(),
            volume=self.volume_knob.get_value(),
            sample_rate=new_sample_rate,
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if not self.freq_input.is_connected and not self.gate_input.is_connected:
            self.out_port.write(silence(num_samples))
            return

        voice = self.component
        voice.waveform = _WAVE_TO_ENGINE.get(
            str_parameter(parameters, "waveform", self.wave_combo.currentText),
            "saw",
        )
        voice.filter_mode = _filter_mode(
            str_parameter(parameters, "filter_mode", self.filter_combo.currentText)
        )
        voice.cutoff = float_parameter(parameters, "cutoff", self.cutoff_knob.get_value)
        voice.resonance = float_parameter(
            parameters, "resonance", self.resonance_knob.get_value
        )
        voice.env_amount = float_parameter(
            parameters, "env_amount", self.env_mod_knob.get_value
        )
        voice.attack = float_parameter(parameters, "attack", self.attack_knob.get_value)
        voice.decay = float_parameter(parameters, "decay", self.decay_knob.get_value)
        voice.sustain = float_parameter(
            parameters, "sustain", self.sustain_knob.get_value
        )
        voice.release = float_parameter(
            parameters, "release", self.release_knob.get_value
        )
        voice.lfo_rate_hz = float_parameter(
            parameters, "lfo_rate", self.lfo_rate_knob.get_value
        )
        voice.lfo_filter_depth = float_parameter(
            parameters, "lfo_filter", self.lfo_filter_knob.get_value
        )
        voice.volume = float_parameter(parameters, "volume", self.volume_knob.get_value)

        if self.freq_input.is_connected:
            freq = pitch_cv_to_frequency(
                as_mono(read_samples(self.freq_input, num_samples))
            )
        else:
            freq = np.full(num_samples, 220.0, dtype=np.float32)
        if self.gate_input.is_connected:
            raw_gate = as_mono(read_samples(self.gate_input, num_samples))
            # Clock/trig pulses are ~2 ms; stretch so attack can actually open.
            attack_s = max(0.02, float(voice.attack) * 0.5)
            width = min_trigger_samples(
                audio_config.sample_rate, min(0.12, max(0.02, attack_s))
            )
            gate, self._gate_hold = ensure_min_pulse_width(
                raw_gate,
                self._previous_raw_gate,
                width,
                self._gate_hold,
            )
            if num_samples > 0:
                self._previous_raw_gate = float(raw_gate[-1])
        else:
            gate = np.ones(num_samples, dtype=np.float32)
        self.out_port.write(voice.process(frequency=freq, gate=gate))
