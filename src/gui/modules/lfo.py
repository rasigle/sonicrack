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
class LFOModule(ModuleWidget):
    """LFO (Low Frequency Oscillator) module for modulation.

    Similar to Oscillator but optimized for modulation (0.01 Hz - 20 Hz).
    """

    metadata = ModuleMetadata(
        title="LFO",
        category=ModuleCategory.SOURCE,
        description="Low-frequency oscillator for modulation",
    )

    def __init__(self):
        """Initialize LFO module."""
        super().__init__(
            width=220,
            height=200,
            color=QColor(100, 140, 200),
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

        # Frequency control (knobs) - optimized for LFO range
        knobs_layout = QHBoxLayout()
        self.freq_knob = Knob("Rate (Hz)", 0.01, 20.0, 1.0)
        self.freq_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("frequency", self.freq_knob.get_value())
        )
        knobs_layout.addWidget(self.freq_knob)

        # Gain in dB (linear mapping of dB values, since dB is already logarithmic)
        # Range: -60 dB (very quiet) to +12 dB (boost)
        # Default: -20 dB (safe for mixing)
        self.depth_knob = Knob("Depth", -60, 12, DEFAULT_GAIN_DB, logarithmic=False)
        self.depth_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("gain_db", self.depth_knob.get_value())
        )
        knobs_layout.addWidget(self.depth_knob)

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
        self.register_parameter("gain_db", self.depth_knob)
        self.register_parameter("phase", self.phase_slider)

        self.component = self.create_engine_component()

    # AudioModuleInterface implementation
    def _on_wave_changed(self, wave_type: str):
        """Handle waveform type change."""
        self.component = self.create_engine_component()
        self.parameter_changed.emit("waveform", wave_type)

    def get_cv_output_range(self) -> tuple[float, float]:
        """LFO outputs bipolar signal [-1, 1].

        Returns:
            (-1.0, 1.0) - bipolar output range
        """
        return -1.0, 1.0

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the LFO component."""
        wave_type = self.wave_combo.currentText()
        freq = self.freq_knob.get_value()
        gain_db = self.depth_knob.get_value()
        phase = self.phase_slider.get_value()

        # LFO uses the same oscillators but at lower frequencies
        # and with wave_range set to modulation range (-1 to 1)
        if wave_type == "Sine":
            return SineOscillator(
                freq, gain_db=gain_db, phase=phase, wave_range=(-1, 1)
            )
        if wave_type == "Square":
            return SquareOscillator(
                freq, gain_db=gain_db, phase=phase, wave_range=(-1, 1)
            )
        if wave_type == "Sawtooth":
            return SawtoothOscillator(
                freq, gain_db=gain_db, phase=phase, wave_range=(-1, 1)
            )
        if wave_type == "Triangle":
            return TriangleOscillator(
                freq, gain_db=gain_db, phase=phase, wave_range=(-1, 1)
            )
        raise ValueError(f"Unknown waveform type: {wave_type}")
