from typing import Any

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QComboBox,
)

from constants import DEFAULT_GAIN_DB
from src.engine import (
    SineOscillator,
    SquareOscillator,
    SawtoothOscillator,
    TriangleOscillator,
)
from src.gui.audio_module_interface import ModuleCategory, ModuleMetadata
from src.gui.widgets import Knob, HSlider
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.module_registry import register_module


@register_module()
class OscillatorModule(ModuleWidget):
    """Oscillator module with frequency and gain controls."""

    metadata = ModuleMetadata(
        title="Oscillator",
        category=ModuleCategory.SOURCE,
        description="Multi-waveform oscillator with frequency and gain controls",
    )

    def __init__(self):
        """Initialize oscillator module."""
        super().__init__(
            width=220,
            height=200,
            color=QColor(80, 120, 200),
        )

        # Add output port
        self.out_port = self.add_output_port("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Waveform selector
        wave_layout = QHBoxLayout()
        wave_layout.addWidget(QLabel("Wave:"))
        self.wave_combo = QComboBox()
        self.wave_combo.addItems(["Sine", "Square", "Sawtooth", "Triangle"])
        self.wave_combo.currentTextChanged.connect(self._on_wave_changed)
        wave_layout.addWidget(self.wave_combo)
        layout.addLayout(wave_layout)

        # Frequency control (knobs)
        knobs_layout = QHBoxLayout()
        self.freq_knob = Knob("Freq (Hz)", 20, 2000, 440)
        self.freq_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("frequency", self.freq_knob.get_value())
        )
        knobs_layout.addWidget(self.freq_knob)

        # Gain in dB (linear mapping of dB values, since dB is already logarithmic)
        # Range: -60 dB (very quiet) to +12 dB (boost)
        # Default: -20 dB (safe for mixing)
        self.gain_knob = Knob("Gain (dB)", -60, 12, DEFAULT_GAIN_DB, logarithmic=False)
        self.gain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("gain_db", self.gain_knob.get_value())
        )
        knobs_layout.addWidget(self.gain_knob)
        layout.addLayout(knobs_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter(
            "waveform", self.wave_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("gain_db", self.gain_knob)

        self.component = self.create_engine_component()

    # AudioModuleInterface implementation
    def _on_wave_changed(self, wave_type: str):
        """Handle waveform type change."""
        self.create_engine_component()
        self.parameter_changed.emit("waveform", wave_type)

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the oscillator component."""
        wave_type = self.wave_combo.currentText()
        freq = self.freq_knob.get_value()
        gain_db = self.gain_knob.get_value()

        if wave_type == "Sine":
            return SineOscillator(freq, gain_db=gain_db)
        elif wave_type == "Square":
            return SquareOscillator(freq, gain_db=gain_db)
        elif wave_type == "Sawtooth":
            return SawtoothOscillator(freq, gain_db=gain_db)
        elif wave_type == "Triangle":
            return TriangleOscillator(freq, gain_db=gain_db)

        raise ValueError(f"Unknown waveform type: {wave_type}")
