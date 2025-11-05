"""Modulated Oscillator module with frequency modulation input.

This module provides an oscillator that can have its frequency controlled
by an external CV source (like MIDI Input). Perfect for MIDI-controlled synthesis!
"""

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
from src.engine.modulated_oscillator import ModulatedOscillator
from src.gui.audio_module_interface import ModuleCategory, ModuleMetadata
from src.gui.widgets import Knob, HSlider
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.module_registry import register_module


@register_module()
class ModulatedOscillatorModule(ModuleWidget):
    """Oscillator with frequency modulation input.

    This oscillator can have its frequency controlled by an external CV source,
    making it perfect for MIDI keyboard control via the MIDI Input module.

    Inputs:
        - Freq: Frequency CV input (e.g., from MIDI Input)

    Outputs:
        - Out: Audio output
    """

    metadata = ModuleMetadata(
        title="VCO",
        category=ModuleCategory.MODULATED_SOURCE,  # Generator with CV inputs
        description="Voltage-Controlled Oscillator with frequency modulation input",
        version="1.0.0",
        author="AudioPlayground",
    )

    def __init__(self):
        """Initialize modulated oscillator module."""
        super().__init__(
            width=220,
            height=200,
            color=QColor(100, 140, 220),
        )

        # Add ports
        self.freq_input = self.add_input_port("Freq")
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

        # Frequency control (base frequency when no modulation)
        knobs_layout = QHBoxLayout()
        self.freq_knob = Knob("Base Hz", 20, 2000, 440)
        self.freq_knob.setToolTip(
            "Base frequency (Hz)\n"
            "Active when Freq input is disconnected\n"
            "Recompile patch after connection changes"
        )
        self.freq_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("frequency", self.freq_knob.get_value())
        )
        knobs_layout.addWidget(self.freq_knob)

        # Gain in dB
        self.gain_knob = Knob("Gain (dB)", -60, 12, -20, logarithmic=False)
        self.gain_knob.setToolTip(
            "Oscillator gain (dB)\n" "Range: -60 to +12 dB\n" "Default: -20 dB"
        )
        self.gain_knob.value_changed.connect(self._on_gain_changed)
        knobs_layout.addWidget(self.gain_knob)

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
        self.register_parameter("gain_db", self.gain_knob)
        self.register_parameter("phase", self.phase_slider)

        self.component = self.create_engine_component()

    def _on_wave_changed(self, wave_type: str):
        """Handle waveform type change."""
        self.parameter_changed.emit("waveform", wave_type)

    def _on_gain_changed(self):
        """Handle gain knob change."""
        import logging

        logger = logging.getLogger(__name__)
        gain_value = self.gain_knob.get_value()
        logger.info(f"VCO Gain changed to {gain_value} dB")
        self.parameter_changed.emit("gain_db", gain_value)

    def get_required_inputs(self) -> list[str]:
        """Freq input is optional - VCO works as normal oscillator without it."""
        return []  # No required inputs - Freq is optional

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the modulated oscillator component.

        If a frequency modulator is connected, creates a ModulatedOscillator.
        Otherwise, creates a standard oscillator.
        """
        wave_type = self.wave_combo.currentText()
        base_freq = self.freq_knob.get_value()
        gain_db = self.gain_knob.get_value()
        phase = self.phase_slider.get_value()

        # Create the base oscillator
        if wave_type == "Sine":
            osc = SineOscillator(base_freq, gain_db=gain_db, phase=phase)
        elif wave_type == "Square":
            osc = SquareOscillator(base_freq, gain_db=gain_db, phase=phase)
        elif wave_type == "Sawtooth":
            osc = SawtoothOscillator(base_freq, gain_db=gain_db, phase=phase)
        elif wave_type == "Triangle":
            osc = TriangleOscillator(base_freq, gain_db=gain_db, phase=phase)
        else:
            raise ValueError(f"Unknown waveform type: {wave_type}")

        # If we have a frequency modulation input, create ModulatedOscillator
        if input_components and len(input_components) > 0:
            freq_modulator = input_components[0]

            # Frequency modulation: replace the oscillator's frequency with CV value
            # Make it vectorized to work with numpy arrays
            def freq_mod_func(base_freq, cv_freq):
                """Use CV frequency directly (MIDI sends Hz values).

                This function is vectorized - it can accept numpy arrays.
                """
                # cv_freq can be a single value or numpy array
                return cv_freq  # Just return the CV frequency, ignore base

            # Visual feedback: Disable frequency knob when CV is connected
            self.freq_knob.setEnabled(False)
            self.freq_knob.setStyleSheet("opacity: 0.5;")  # Gray it out
            self.freq_knob.setToolTip("Frequency controlled by Freq input (CV)")

            # Log for debugging
            import logging

            logger = logging.getLogger(__name__)
            logger.info("VCO: Creating ModulatedOscillator with freq modulation")
            logger.info(
                f"VCO: Base freq={base_freq}, Modulator type={type(freq_modulator)}"
            )

            return ModulatedOscillator(osc, freq_modulator, freq_mod=freq_mod_func)

        # No modulation - enable frequency knob
        self.freq_knob.setEnabled(True)
        self.freq_knob.setStyleSheet("")  # Remove grayed out style
        self.freq_knob.setToolTip("Manual frequency control (Hz)")

        # Return plain oscillator
        return osc
