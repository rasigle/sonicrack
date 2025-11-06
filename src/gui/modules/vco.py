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

from constants import DEFAULT_GAIN_DB
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
        self.gain_mod_input = self.add_input_port("Gain")
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
        self.gain_knob = Knob("Gain (dB)", -60, 12, DEFAULT_GAIN_DB, logarithmic=False)
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
        """Freq and Gain inputs are optional - VCO works as normal oscillator without them."""
        return []  # No required inputs - Freq and Gain are optional

    def get_modulation_inputs(self) -> list[str]:
        """VCO accepts modulation on Gain port."""
        return ["Gain"]

    def get_cv_range(self, port_name: str = "Gain") -> tuple[float, float]:
        """VCO Gain port expects unipolar CV range [0, 1].

        Returns:
            (0.0, 1.0) - unipolar range for amplitude modulation
        """
        return 0.0, 1.0

    def update_knob_state(self):
        """Update knob enabled state based on port connections.

        This is called when connections change to provide immediate visual feedback,
        even if compilation fails.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Check Freq port
        freq_port = None
        for port in self.input_ports:
            if port.port_name == "Freq":
                freq_port = port
                break

        has_freq_cv = freq_port and len(freq_port.cables) > 0

        logger.info(f"VCO update_knob_state: Freq port has {len(freq_port.cables) if freq_port else 0} cables, has_freq_cv={has_freq_cv}")

        if has_freq_cv:
            # Frequency controlled by CV - disable knob
            self.freq_knob.setEnabled(False)
            self.freq_knob.setStyleSheet("opacity: 0.5;")
            self.freq_knob.setToolTip("Frequency controlled by Freq input (CV)")
            logger.info("VCO: Freq knob DISABLED")
        else:
            # No frequency CV - enable knob
            self.freq_knob.setEnabled(True)
            self.freq_knob.setStyleSheet("")
            self.freq_knob.setToolTip("Manual frequency control (Hz)")
            logger.info("VCO: Freq knob ENABLED")

        # Check Gain port
        gain_port = None
        for port in self.input_ports:
            if port.port_name == "Gain":
                gain_port = port
                break

        has_gain_cv = gain_port and len(gain_port.cables) > 0

        logger.info(f"VCO update_knob_state: Gain port has {len(gain_port.cables) if gain_port else 0} cables, has_gain_cv={has_gain_cv}")

        if has_gain_cv:
            # Gain controlled by CV - disable knob
            self.gain_knob.setEnabled(False)
            self.gain_knob.setStyleSheet("opacity: 0.5;")
            self.gain_knob.setToolTip("Gain controlled by Gain input (CV)")
            logger.info("VCO: Gain knob DISABLED")
        else:
            # No gain CV - enable knob
            self.gain_knob.setEnabled(True)
            self.gain_knob.setStyleSheet("")
            self.gain_knob.setToolTip("Manual gain control (dB)")
            logger.info("VCO: Gain knob ENABLED")

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the modulated oscillator component.

        Supports both frequency and gain modulation.
        - Freq input: CV control of frequency (e.g., from MIDI Input)
        - Gain input: CV control of amplitude (e.g., from envelope)
        """
        import logging
        logger = logging.getLogger(__name__)

        wave_type = self.wave_combo.currentText()
        base_freq = self.freq_knob.get_value()
        gain_db = self.gain_knob.get_value()
        phase = self.phase_slider.get_value()

        # Check for modulation inputs
        freq_modulator = input_components[0] if input_components and len(input_components) > 0 else None
        gain_modulator = modulation_components.get("Gain") if modulation_components else None

        has_freq_mod = freq_modulator is not None
        has_gain_mod = gain_modulator is not None

        logger.debug(f"VCO: freq_mod={has_freq_mod}, gain_mod={has_gain_mod}")

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

        # Update UI state for frequency knob
        if has_freq_mod:
            self.freq_knob.setEnabled(False)
            self.freq_knob.setStyleSheet("opacity: 0.5;")
            self.freq_knob.setToolTip("Frequency controlled by Freq input (CV)")
            logger.info("VCO create_engine_component: Freq knob DISABLED (has modulation)")
        else:
            self.freq_knob.setEnabled(True)
            self.freq_knob.setStyleSheet("")
            self.freq_knob.setToolTip("Manual frequency control (Hz)")
            logger.info("VCO create_engine_component: Freq knob ENABLED (no modulation)")

        # Update UI state for gain knob
        if has_gain_mod:
            self.gain_knob.setEnabled(False)
            self.gain_knob.setStyleSheet("opacity: 0.5;")
            self.gain_knob.setToolTip("Gain controlled by Gain input (CV)")
            logger.info("VCO create_engine_component: Gain knob DISABLED (has modulation)")
        else:
            self.gain_knob.setEnabled(True)
            self.gain_knob.setStyleSheet("")
            self.gain_knob.setToolTip("Manual gain control (dB)")
            logger.info("VCO create_engine_component: Gain knob ENABLED (no modulation)")

        # If we have frequency or gain modulation, create ModulatedOscillator
        if has_freq_mod or has_gain_mod:
            # Frequency modulation function
            def freq_mod_func(base_freq, cv_freq):
                """Use CV frequency directly (MIDI sends Hz values)."""
                return cv_freq

            # Amplitude modulation function (for gain modulation)
            def amp_mod_func(base_amp, cv_amp):
                """Use CV value to scale amplitude."""
                # cv_amp is typically [0, 1] from envelope
                # Multiply base amplitude by CV value
                return base_amp * cv_amp

            logger.info(f"VCO: Creating ModulatedOscillator (freq_mod={has_freq_mod}, gain_mod={has_gain_mod})")

            # Build modulator list based on what's connected
            modulators = []
            amp_mod = None
            freq_mod = None

            if has_gain_mod:
                modulators.append(gain_modulator)
                amp_mod = amp_mod_func

            if has_freq_mod:
                modulators.append(freq_modulator)
                freq_mod = freq_mod_func

            return ModulatedOscillator(
                osc,
                *modulators,
                amp_mod=amp_mod,
                freq_mod=freq_mod
            )

        # No modulation - return plain oscillator
        return osc
