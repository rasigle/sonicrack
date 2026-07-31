"""Morphable wavetable oscillator module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.generators.oscillators import WavetableOscillator
from soniclab.utils.cv import pitch_cv_to_frequency

from sonicrack.config.audio_config import audio_config
from sonicrack.constants import AUDIO_FREQUENCY_KNOB_CURVE
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class WavetableModule(ModuleWidget):
    """Morphable wavetable source (sine → triangle → saw → square)."""

    runtime_kind = "wavetable"

    metadata = ModuleMetadata(
        title="Wavetable",
        category=ModuleCategory.SOURCE,
        description="Morphable wavetable oscillator with optional 1V/oct CV",
    )

    def __init__(self) -> None:
        super().__init__(width=220, height=250, color=QColor(70, 130, 160))
        self.freq_input = self.add_input("Freq", signal=PortSignal.PITCH_CV)
        self.out_port = self.add_output("Out", signal=PortSignal.AUDIO)
        self.component = WavetableOscillator(sample_rate=audio_config.sample_rate)

        layout = self._begin_controls()
        row = QHBoxLayout()
        self.freq_knob = Knob(
            label="Freq",
            description="Base frequency in Hz",
            min_value=20.0,
            max_value=4000.0,
            default_value=220.0,
            logarithmic=True,
            curve_points=AUDIO_FREQUENCY_KNOB_CURVE,
        )
        self.freq_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("frequency", self.freq_knob.get_value())
        )
        row.addWidget(self.freq_knob)

        self.morph_knob = Knob(
            label="Morph",
            description="Morph across sine / triangle / saw / square tables",
            min_value=0.0,
            max_value=1.0,
            default_value=0.0,
        )
        self.morph_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("morph", self.morph_knob.get_value())
        )
        row.addWidget(self.morph_knob)
        layout.addLayout(row)

        self.gain_knob = Knob(
            label="Level",
            description="Output level",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self.gain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("gain", self.gain_knob.get_value())
        )
        layout.addWidget(self.gain_knob)
        self._finish_controls(layout)

        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("morph", self.morph_knob)
        self.register_parameter("gain", self.gain_knob)
        self._install_sample_rate_listener()

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component = WavetableOscillator(
            frequency=self.freq_knob.get_value(),
            morph=self.morph_knob.get_value(),
            gain=self.gain_knob.get_value(),
            sample_rate=new_sample_rate,
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        base_freq = float_parameter(parameters, "frequency", self.freq_knob.get_value)
        morph = float_parameter(parameters, "morph", self.morph_knob.get_value)
        gain = float_parameter(parameters, "gain", self.gain_knob.get_value)

        if self.freq_input.is_connected:
            pitch_cv = float(read_samples(self.freq_input, num_samples)[0])
            frequency = pitch_cv_to_frequency(pitch_cv)
        else:
            frequency = base_freq

        self.component.frequency = frequency
        self.component.morph = morph
        self.component.gain = gain
        self.out_port.write(self.component.get_samples(num_samples))
