"""TB-303 style voice module."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel
from soniclab.utils.cv import pitch_cv_to_frequency
from soniclab.voices import TB303Voice

from sonicrack.gui.audio_config import audio_config
from sonicrack.gui.core.module import ModuleCategory, ModuleMetadata
from sonicrack.gui.core.port import PortSignal
from sonicrack.gui.core.runtime import RuntimeParameters
from sonicrack.gui.core.runtime_helpers import float_parameter, read_samples, silence
from sonicrack.gui.module_registry import register_module
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget


@register_module()
class TB303VoiceModule(ModuleWidget):
    """Convenience acid-bass voice driven by pitch/gate/accent/slide CV."""

    runtime_kind = "tb303_voice"

    metadata = ModuleMetadata(
        title="TB-303 Voice",
        category=ModuleCategory.MODULATED_SOURCE,
        description="Acid bass voice with oscillator, slide, envelope, and filter",
    )

    def __init__(self) -> None:
        super().__init__(width=300, height=440, color=QColor(155, 145, 65))

        self.freq_input = self.add_input("Freq", signal=PortSignal.PITCH_CV)
        self.gate_input = self.add_input("Gate", signal=PortSignal.GATE)
        self.accent_input = self.add_input("Accent", signal=PortSignal.CONTROL_CV)
        self.slide_input = self.add_input("Slide", signal=PortSignal.CONTROL_CV)
        self.out_port = self.add_output("Out", signal=PortSignal.AUDIO)
        self.component = TB303Voice(sample_rate=audio_config.sample_rate)

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout(spacing=4)

        wave_layout = QHBoxLayout()
        wave_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wave_layout.addWidget(QLabel("Wave:"))
        self.wave_combo = QComboBox()
        self.wave_combo.addItems(["Sawtooth", "Square"])
        self.wave_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("waveform", value)
        )
        wave_layout.addWidget(self.wave_combo)
        layout.addLayout(wave_layout)

        osc_row = QHBoxLayout()
        osc_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tune_knob = Knob(
            label="Tune",
            description="Transposes the oscillator in semitones",
            min_value=-24.0,
            max_value=24.0,
            default_value=0.0,
        )
        self.tune_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("tuning", self.tune_knob.get_value())
        )
        osc_row.addWidget(self.tune_knob)

        self.pw_knob = Knob(
            label="PW",
            description="Adjusts the pulse width of the square wave",
            min_value=0.05,
            max_value=0.95,
            default_value=0.5,
        )
        self.pw_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("pulsewidth", self.pw_knob.get_value())
        )
        osc_row.addWidget(self.pw_knob)
        layout.addLayout(osc_row)

        filter_row = QHBoxLayout()
        filter_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cutoff_knob = Knob(
            label="Cutoff",
            description="Sets the cutoff frequency of the filter",
            min_value=20.0,
            max_value=12000.0,
            default_value=700.0,
            logarithmic=True,
        )
        self.cutoff_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("cutoff", self.cutoff_knob.get_value())
        )
        filter_row.addWidget(self.cutoff_knob)

        self.resonance_knob = Knob(
            label="Resonance",
            description="Adjusts the resonance of the filter",
            min_value=0.0,
            max_value=18.0,
            default_value=8.0,
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
            description="Controls the envelope modulation depth",
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

        shape_row = QHBoxLayout()
        shape_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.decay_knob = Knob(
            label="Decay",
            description="Sets the decay time of the envelope",
            min_value=0.01,
            max_value=1.0,
            default_value=0.18,
        )
        self.decay_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("decay", self.decay_knob.get_value())
        )
        shape_row.addWidget(self.decay_knob)

        self.accent_knob = Knob(
            label="Accent",
            description="Controls the accent amount",
            min_value=0.0,
            max_value=1.0,
            default_value=0.7,
        )
        self.accent_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("accent", self.accent_knob.get_value())
        )
        shape_row.addWidget(self.accent_knob)

        self.slide_knob = Knob(
            label="Slide",
            description="Controls the slide time between notes",
            min_value=0.0,
            max_value=0.5,
            default_value=0.08,
        )
        self.slide_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "slide_time", self.slide_knob.get_value()
            )
        )
        shape_row.addWidget(self.slide_knob)
        layout.addLayout(shape_row)

        gain_row = QHBoxLayout()
        gain_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drive_knob = Knob(
            label="Drive",
            description="Adjusts the drive of the filter",
            min_value=0.0,
            max_value=24.0,
            default_value=6.0,
        )
        self.drive_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("drive_db", self.drive_knob.get_value())
        )
        gain_row.addWidget(self.drive_knob)

        self.volume_knob = Knob(
            label="Volume",
            description="Adjusts the output volume",
            min_value=0.0,
            max_value=1.0,
            default_value=0.8,
        )
        self.volume_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("volume", self.volume_knob.get_value())
        )
        gain_row.addWidget(self.volume_knob)
        layout.addLayout(gain_row)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.register_parameter(
            "waveform", self.wave_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter("tuning", self.tune_knob)
        self.register_parameter("pulsewidth", self.pw_knob)
        self.register_parameter("cutoff", self.cutoff_knob)
        self.register_parameter("resonance", self.resonance_knob)
        self.register_parameter("env_amount", self.env_mod_knob)
        self.register_parameter("decay", self.decay_knob)
        self.register_parameter("accent", self.accent_knob)
        self.register_parameter("slide_time", self.slide_knob)
        self.register_parameter("drive_db", self.drive_knob)
        self.register_parameter("volume", self.volume_knob)

        self._sample_rate_listener = self._on_global_sample_rate_changed
        audio_config.add_sample_rate_listener(self._sample_rate_listener)
        self.destroyed.connect(self._cleanup_audio_config_listeners)

    def get_required_inputs(self) -> list[str]:
        return ["Freq", "Gate"]

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component = TB303Voice(sample_rate=new_sample_rate)

    def _cleanup_audio_config_listeners(self, *_args: object) -> None:
        audio_config.remove_sample_rate_listener(self._sample_rate_listener)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if not self.freq_input.is_connected or not self.gate_input.is_connected:
            self.out_port.write(silence(num_samples))
            return

        self.component.waveform = str(
            parameters.get("waveform", self.wave_combo.currentText())
        )
        self.component.tuning = float_parameter(
            parameters, "tuning", self.tune_knob.get_value
        )
        self.component.pulsewidth = float_parameter(
            parameters, "pulsewidth", self.pw_knob.get_value
        )
        self.component.cutoff = float_parameter(
            parameters, "cutoff", self.cutoff_knob.get_value
        )
        self.component.resonance = float_parameter(
            parameters, "resonance", self.resonance_knob.get_value
        )
        self.component.env_amount = float_parameter(
            parameters, "env_amount", self.env_mod_knob.get_value
        )
        self.component.decay = float_parameter(
            parameters, "decay", self.decay_knob.get_value
        )
        self.component.accent = float_parameter(
            parameters, "accent", self.accent_knob.get_value
        )
        self.component.slide_time = float_parameter(
            parameters, "slide_time", self.slide_knob.get_value
        )
        self.component.drive_db = float_parameter(
            parameters, "drive_db", self.drive_knob.get_value
        )
        self.component.volume = float_parameter(
            parameters, "volume", self.volume_knob.get_value
        )
        frequency_signal = np.asarray(
            pitch_cv_to_frequency(read_samples(self.freq_input, num_samples)),
            dtype=np.float32,
        ).reshape(-1)

        self.out_port.write(
            self.component.process(
                frequency=frequency_signal,
                gate=read_samples(self.gate_input, num_samples),
                accent=(
                    read_samples(self.accent_input, num_samples)
                    if self.accent_input.is_connected
                    else None
                ),
                slide=(
                    read_samples(self.slide_input, num_samples)
                    if self.slide_input.is_connected
                    else None
                ),
            )
        )
