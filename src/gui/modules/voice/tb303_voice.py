"""TB-303 style voice module."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel

from src.engine.voices import TB303Voice
from src.engine.utils.cv import pitch_cv_to_frequency
from src.gui.audio_config import audio_config
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, read_samples, silence
from src.gui.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


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

        self.freq_input = self.add_input("Freq")
        self.gate_input = self.add_input("Gate")
        self.accent_input = self.add_input("Accent")
        self.slide_input = self.add_input("Slide")
        self.out_port = self.add_output("Out")
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
        self.tuning_knob = Knob("Tune", 0.5, 2.0, 1.0)
        self.tuning_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("tuning", self.tuning_knob.get_value())
        )
        osc_row.addWidget(self.tuning_knob)

        self.pulsewidth_knob = Knob("PW", 0.05, 0.95, 0.5)
        self.pulsewidth_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "pulsewidth", self.pulsewidth_knob.get_value()
            )
        )
        osc_row.addWidget(self.pulsewidth_knob)
        layout.addLayout(osc_row)

        filter_row = QHBoxLayout()
        filter_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cutoff_knob = Knob("Cutoff", 20.0, 12000.0, 700.0, logarithmic=True)
        self.cutoff_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("cutoff", self.cutoff_knob.get_value())
        )
        filter_row.addWidget(self.cutoff_knob)

        self.resonance_knob = Knob("Res", 0.1, 18.0, 8.0, logarithmic=True)
        self.resonance_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "resonance", self.resonance_knob.get_value()
            )
        )
        filter_row.addWidget(self.resonance_knob)

        self.env_amount_knob = Knob("Env", 0.0, 6.0, 2.5)
        self.env_amount_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "env_amount", self.env_amount_knob.get_value()
            )
        )
        filter_row.addWidget(self.env_amount_knob)
        layout.addLayout(filter_row)

        shape_row = QHBoxLayout()
        shape_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.decay_knob = Knob("Decay", 0.01, 1.0, 0.18)
        self.decay_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("decay", self.decay_knob.get_value())
        )
        shape_row.addWidget(self.decay_knob)

        self.accent_knob = Knob("Accent", 0.0, 1.0, 0.7)
        self.accent_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("accent", self.accent_knob.get_value())
        )
        shape_row.addWidget(self.accent_knob)

        self.slide_time_knob = Knob("Slide", 0.0, 0.5, 0.08)
        self.slide_time_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "slide_time", self.slide_time_knob.get_value()
            )
        )
        shape_row.addWidget(self.slide_time_knob)
        layout.addLayout(shape_row)

        gain_row = QHBoxLayout()
        gain_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drive_knob = Knob("Drive", 0.0, 24.0, 6.0)
        self.drive_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("drive_db", self.drive_knob.get_value())
        )
        gain_row.addWidget(self.drive_knob)

        self.volume_knob = Knob("Volume", 0.0, 1.0, 0.8)
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
        self.register_parameter("tuning", self.tuning_knob)
        self.register_parameter("pulsewidth", self.pulsewidth_knob)
        self.register_parameter("cutoff", self.cutoff_knob)
        self.register_parameter("resonance", self.resonance_knob)
        self.register_parameter("env_amount", self.env_amount_knob)
        self.register_parameter("decay", self.decay_knob)
        self.register_parameter("accent", self.accent_knob)
        self.register_parameter("slide_time", self.slide_time_knob)
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
            parameters, "tuning", self.tuning_knob.get_value
        )
        self.component.pulsewidth = float_parameter(
            parameters, "pulsewidth", self.pulsewidth_knob.get_value
        )
        self.component.cutoff = float_parameter(
            parameters, "cutoff", self.cutoff_knob.get_value
        )
        self.component.resonance = float_parameter(
            parameters, "resonance", self.resonance_knob.get_value
        )
        self.component.env_amount = float_parameter(
            parameters, "env_amount", self.env_amount_knob.get_value
        )
        self.component.decay = float_parameter(
            parameters, "decay", self.decay_knob.get_value
        )
        self.component.accent = float_parameter(
            parameters, "accent", self.accent_knob.get_value
        )
        self.component.slide_time = float_parameter(
            parameters, "slide_time", self.slide_time_knob.get_value
        )
        self.component.drive_db = float_parameter(
            parameters, "drive_db", self.drive_knob.get_value
        )
        self.component.volume = float_parameter(
            parameters, "volume", self.volume_knob.get_value
        )

        self.out_port.write(
            self.component.process(
                frequency=pitch_cv_to_frequency(
                    read_samples(self.freq_input, num_samples)
                ),
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
