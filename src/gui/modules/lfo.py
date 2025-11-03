from typing import Dict, Any

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, \
    QGraphicsProxyWidget

from engine import SineOscillator, SquareOscillator, SawtoothOscillator, \
    TriangleOscillator
from gui.widgets.module_widget import ModuleWidget
from gui.widgets import Knob, HSlider

TITLE = "LFO"

class LFOModule(ModuleWidget):
    """LFO (Low Frequency Oscillator) module for modulation.

    Similar to Oscillator but optimized for modulation (0.01 Hz - 20 Hz).
    """

    def __init__(self):
        """Initialize LFO module."""
        super().__init__(TITLE, category="oscillator", width=220, height=200, color=QColor(100, 140, 200))

        # Add output port
        self.out_port = self.add_output_port("Out")

        # Create control widget
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Waveform selector
        wave_layout = QHBoxLayout()
        wave_layout.addWidget(QLabel("Wave:"))
        self.wave_combo = QComboBox()
        self.wave_combo.addItems(["Sine", "Square", "Sawtooth", "Triangle"])
        self.wave_combo.currentTextChanged.connect(self._on_wave_changed)
        wave_layout.addWidget(self.wave_combo)
        layout.addLayout(wave_layout)

        # Frequency control (knobs) - optimized for LFO range
        knobs_layout = QHBoxLayout()
        self.freq_knob = Knob("Rate (Hz)", 0.01, 20.0, 1.0)
        self.freq_knob.value_changed.connect(lambda: self.parameter_changed.emit("frequency", self.freq_knob.get_value()))
        knobs_layout.addWidget(self.freq_knob)

        self.amp_knob = Knob("Depth", 0.0, 1.0, 0.5)
        self.amp_knob.value_changed.connect(lambda: self.parameter_changed.emit("amplitude", self.amp_knob.get_value()))
        knobs_layout.addWidget(self.amp_knob)

        layout.addLayout(knobs_layout)

        # Phase control
        self.phase_slider = HSlider("Phase", 0, 360, 0)
        self.phase_slider.value_changed.connect(lambda v: self.parameter_changed.emit("phase", v))
        layout.addWidget(self.phase_slider)

        self.controls_widget.setLayout(layout)

        # Add controls as proxy widget
        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(self.controls_widget)
        self.proxy.setPos(0, 42)

        self.create_component()

    def _on_wave_changed(self, wave_type: str):
        """Handle waveform type change."""
        self.component = self.create_component()
        self.parameter_changed.emit("waveform", wave_type)

    def create_component(self):
        """Create the LFO component."""
        wave_type = self.wave_combo.currentText()
        freq = self.freq_knob.get_value()
        amp = self.amp_knob.get_value()
        phase = self.phase_slider.get_value()

        # LFO uses the same oscillators but at lower frequencies
        # and with wave_range set to modulation range (-1 to 1)
        if wave_type == "Sine":
            return SineOscillator(freq, amplitude=amp, phase=phase, wave_range=(-1, 1))
        if wave_type == "Square":
            return SquareOscillator(freq, amplitude=amp, phase=phase, wave_range=(-1, 1))
        if wave_type == "Sawtooth":
            return SawtoothOscillator(freq, amplitude=amp, phase=phase, wave_range=(-1, 1))
        if wave_type == "Triangle":
            return TriangleOscillator(freq, amplitude=amp, phase=phase, wave_range=(-1, 1))
        raise ValueError(f"Unknown waveform type: {wave_type}")


    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {
            "waveform": self.wave_combo.currentText(),
            "frequency": self.freq_knob.get_value(),
            "amplitude": self.amp_knob.get_value(),
            "phase": self.phase_slider.get_value(),
        }

    def set_parameters(self, params: Dict[str, Any]):
        """Set parameters from dictionary."""
        if "waveform" in params:
            index = self.wave_combo.findText(params["waveform"])
            if index >= 0:
                self.wave_combo.setCurrentIndex(index)
        if "frequency" in params:
            self.freq_knob.set_value(params["frequency"])
        if "amplitude" in params:
            self.amp_knob.set_value(params["amplitude"])
        if "phase" in params:
            self.phase_slider.set_value(params["phase"])
