from typing import Any

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QComboBox,
)

from src.engine import (
    SineOscillator,
    SquareOscillator,
    SawtoothOscillator,
    TriangleOscillator,
)
from src.gui.audio_module_interface import ModuleCategory
from src.gui.widgets import Knob, HSlider
from src.gui.widgets.module_widget import ModuleWidget


class OscillatorModule(ModuleWidget):
    """Oscillator module with frequency and amplitude controls."""

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

        self.amp_knob = Knob("Amplitude", 0.0, 1.0, 0.5)
        self.amp_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("amplitude", self.amp_knob.get_value())
        )
        knobs_layout.addWidget(self.amp_knob)

        layout.addLayout(knobs_layout)

        # Phase control
        self.phase_slider = HSlider("Phase", 0, 360, 0)
        self.phase_slider.value_changed.connect(
            lambda v: self.parameter_changed.emit("phase", v)
        )
        layout.addWidget(self.phase_slider)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter(
            "waveform", self.wave_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("amplitude", self.amp_knob)
        self.register_parameter("phase", self.phase_slider)

        self.component = self.create_component()

    # AudioModuleInterface implementation
    @property
    def module_title(self) -> str:
        """Return the module title."""
        return "Oscillator"

    @property
    def module_category(self) -> ModuleCategory:
        """Return SOURCE since oscillators generate audio."""
        return ModuleCategory.SOURCE

    def _on_wave_changed(self, wave_type: str):
        """Handle waveform type change."""
        self.create_component()
        self.parameter_changed.emit("waveform", wave_type)

    def create_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the oscillator component."""
        wave_type = self.wave_combo.currentText()
        freq = self.freq_knob.get_value()
        amp = self.amp_knob.get_value()
        phase = self.phase_slider.get_value()

        if wave_type == "Sine":
            return SineOscillator(freq, amplitude=amp, phase=phase)
        elif wave_type == "Square":
            return SquareOscillator(freq, amplitude=amp, phase=phase)
        elif wave_type == "Sawtooth":
            return SawtoothOscillator(freq, amplitude=amp, phase=phase)
        elif wave_type == "Triangle":
            return TriangleOscillator(freq, amplitude=amp, phase=phase)

        raise ValueError(f"Unknown waveform type: {wave_type}")
