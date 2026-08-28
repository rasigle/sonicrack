"""Detuned multi-oscillator (supersaw-style) source."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel
from soniclab.generators.oscillators.oscillator_polyblep import (
    PolyBLEPOscillator,
    WaveShape,
)
from soniclab.utils.cv import pitch_cv_to_frequency

from sonicrack.config.audio_config import audio_config
from sonicrack.constants import AUDIO_FREQUENCY_KNOB_CURVE
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import (
    as_mono,
    constant_power_pan,
    float_parameter,
    read_samples,
    str_parameter,
)
from sonicrack.runtime.specs import RuntimeParameters

_WAVE_ITEMS = ("Saw", "Square", "Triangle", "Sine")
_WAVE_TO_SHAPE = {
    "Saw": WaveShape.SAWTOOTH_UP,
    "Square": WaveShape.SQUARE,
    "Triangle": WaveShape.TRIANGLE,
    "Sine": WaveShape.SINE,
}


@register_module()
class UnisonModule(ModuleWidget):
    """Stacked, detuned oscillators with stereo spread."""

    runtime_kind = "unison"

    metadata = ModuleMetadata(
        title="Unison",
        category=ModuleCategory.SOURCE,
        description="Detuned multi-oscillator with stereo spread",
    )

    def __init__(self) -> None:
        super().__init__(width=240, height=280, color=QColor(95, 85, 160))
        self.freq_input = self.add_input("Freq", signal=PortSignal.PITCH_CV)
        self.out_port = self.add_output("Out", signal=PortSignal.AUDIO)
        self._oscillators: list[PolyBLEPOscillator] = []
        self._voice_count = 0

        layout = self._begin_controls(spacing=6)
        wave_row = QHBoxLayout()
        wave_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wave_row.addWidget(QLabel("Wave:"))
        self.wave_combo = QComboBox()
        self.wave_combo.addItems(list(_WAVE_ITEMS))
        self.wave_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("waveform", value)
        )
        wave_row.addWidget(self.wave_combo)
        layout.addLayout(wave_row)

        row1 = QHBoxLayout()
        self.freq_knob = Knob(
            label="Freq",
            description="Base frequency in Hz",
            min_value=20.0,
            max_value=4000.0,
            default_value=110.0,
            logarithmic=True,
            curve_points=AUDIO_FREQUENCY_KNOB_CURVE,
        )
        self.freq_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("frequency", self.freq_knob.get_value())
        )
        row1.addWidget(self.freq_knob)

        self.voices_knob = Knob(
            label="Voices",
            description="Number of stacked oscillators",
            min_value=1.0,
            max_value=7.0,
            default_value=5.0,
        )
        self.voices_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("voices", self.voices_knob.get_value())
        )
        row1.addWidget(self.voices_knob)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.detune_knob = Knob(
            label="Detune",
            description="Detune spread in cents",
            min_value=0.0,
            max_value=50.0,
            default_value=14.0,
        )
        self.detune_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("detune", self.detune_knob.get_value())
        )
        row2.addWidget(self.detune_knob)

        self.spread_knob = Knob(
            label="Spread",
            description="Stereo voice spread",
            min_value=0.0,
            max_value=1.0,
            default_value=0.7,
        )
        self.spread_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("spread", self.spread_knob.get_value())
        )
        row2.addWidget(self.spread_knob)
        layout.addLayout(row2)

        self.level_knob = Knob(
            label="Level",
            description="Output level",
            min_value=0.0,
            max_value=1.0,
            default_value=0.45,
        )
        self.level_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("level", self.level_knob.get_value())
        )
        layout.addWidget(self.level_knob)
        self._finish_controls(layout)

        self.register_parameter(
            "waveform", self.wave_combo, "currentText", "setCurrentText"
        )
        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("voices", self.voices_knob)
        self.register_parameter("detune", self.detune_knob)
        self.register_parameter("spread", self.spread_knob)
        self.register_parameter("level", self.level_knob)
        self._rebuild_oscillators(5, WaveShape.SAWTOOTH_UP)
        self._install_sample_rate_listener()

    def _rebuild_oscillators(self, count: int, shape: WaveShape) -> None:
        count = max(1, min(7, int(count)))
        self._oscillators = [
            PolyBLEPOscillator(
                frequency=110.0,
                amplitude=1.0,
                gain_db=None,
                wave_shape=shape,
                sample_rate=audio_config.sample_rate,
            )
            for _ in range(count)
        ]
        self._voice_count = count

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        del new_sample_rate
        shape = _WAVE_TO_SHAPE.get(self.wave_combo.currentText(), WaveShape.SAWTOOTH_UP)
        self._rebuild_oscillators(int(round(self.voices_knob.get_value())), shape)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        voices = max(
            1,
            min(
                7,
                int(
                    round(
                        float_parameter(
                            parameters, "voices", self.voices_knob.get_value
                        )
                    )
                ),
            ),
        )
        shape = _WAVE_TO_SHAPE.get(
            str_parameter(parameters, "waveform", self.wave_combo.currentText),
            WaveShape.SAWTOOTH_UP,
        )
        if voices != self._voice_count or (
            self._oscillators and self._oscillators[0].wave_shape != shape
        ):
            self._rebuild_oscillators(voices, shape)

        base_freq = float_parameter(parameters, "frequency", self.freq_knob.get_value)
        if self.freq_input.is_connected:
            pitch_cv = float(as_mono(read_samples(self.freq_input, num_samples))[0])
            base_freq = float(pitch_cv_to_frequency(pitch_cv))
        detune = float_parameter(parameters, "detune", self.detune_knob.get_value)
        spread = float_parameter(parameters, "spread", self.spread_knob.get_value)
        level = float_parameter(parameters, "level", self.level_knob.get_value)

        left = np.zeros(num_samples, dtype=np.float32)
        right = np.zeros(num_samples, dtype=np.float32)
        if voices == 1:
            offsets = np.array([0.0], dtype=np.float32)
        else:
            offsets = np.linspace(-1.0, 1.0, voices, dtype=np.float32)

        gain = level / np.sqrt(float(voices))
        for osc, offset in zip(self._oscillators, offsets, strict=True):
            cents = float(offset) * detune
            osc.frequency = float(base_freq * (2.0 ** (cents / 1200.0)))
            osc.wave_shape = shape
            buf = np.asarray(osc.get_samples(num_samples), dtype=np.float32)
            pan_l, pan_r = constant_power_pan(float(offset) * spread)
            left += buf * pan_l
            right += buf * pan_r

        stereo = np.column_stack((left, right)) * np.float32(gain)
        self.out_port.write(stereo.astype(np.float32, copy=False))
