"""Modulated Oscillator module with frequency modulation input.

This module provides an oscillator that can have its frequency controlled
by an external CV source (like MIDI Input). Perfect for MIDI-controlled synthesis!
"""

from typing import Any

from PyQt6.QtGui import QColor
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QHBoxLayout, QLabel

from src.constants import DEFAULT_GAIN_DB
from src.engine import (
    SineOscillator,
    SawtoothOscillator,
    TriangleOscillator,
    SquareOscillator,
)
from src.engine.oscillator_modulated import ModulatedOscillator
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.widgets import Knob, HSlider
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.core.module_registry import register_module


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
            height=240,
            color=QColor(100, 140, 220),
        )

        # Initialize default parameters FIRST (before creating component)
        self._waveform = "Sine"
        self._mode = self._get_default_mode_for_waveform(self._waveform)
        self._base_frequency = 440.0
        self._gain_db = DEFAULT_GAIN_DB
        self._phase = 0.0

        # Create the base oscillator component FIRST
        self.component = self._create_base_oscillator()

        # Add ports with component reference
        # Note: Input ports don't have components (they receive signals)
        # Output port has the component reference
        self.freq_input = self.add_input("Freq")
        self.gain_mod_input = self.add_input("Gain")
        self.out_port = self.add_output("Out", component=self.component)

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Waveform selector
        wave_layout = QHBoxLayout()
        wave_layout.addWidget(QLabel("Wave:"))
        self.wave_combo = QtWidgets.QComboBox()
        self.wave_combo.addItems(["Sine", "Square", "Sawtooth", "Triangle"])
        self.wave_combo.setCurrentText(self._waveform)
        self.wave_combo.currentTextChanged.connect(self._on_wave_changed)
        wave_layout.addWidget(self.wave_combo)
        layout.addLayout(wave_layout)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._refresh_mode_options(self._waveform, preserve_current=False)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # Frequency control (base frequency when no modulation)
        knobs_layout = QHBoxLayout()
        self.freq_knob = Knob("Base Hz", 20, 2000, self._base_frequency)
        self.freq_knob.setToolTip(
            "Base frequency (Hz)\n"
            "Active when Freq input is disconnected"
        )
        self.freq_knob.value_changed.connect(self._on_frequency_changed)
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
        self.register_parameter(
            "mode", self.mode_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("gain_db", self.gain_knob)
        self.register_parameter("phase", self.phase_slider)

        self.component = self.create_engine_component()

    @staticmethod
    def _get_available_modes_for_waveform(waveform: str) -> list[str]:
        """Return supported engine modes for the selected waveform."""
        if waveform == "Sine":
            return SineOscillator.get_available_modes()
        if waveform == "Square":
            return SquareOscillator.get_available_modes()
        if waveform == "Sawtooth":
            return SawtoothOscillator.get_available_modes()
        if waveform == "Triangle":
            return TriangleOscillator.get_available_modes()
        raise ValueError(f"Unknown waveform type: {waveform}")

    @classmethod
    def _get_default_mode_for_waveform(cls, waveform: str) -> str:
        """Return a sensible default mode for each waveform."""
        preferred_defaults = {
            "Sine": "analog",
            "Square": "ideal",
            "Sawtooth": "analog",
            "Triangle": "analog",
        }
        available_modes = cls._get_available_modes_for_waveform(waveform)
        preferred = preferred_defaults.get(waveform, available_modes[0])
        return preferred if preferred in available_modes else available_modes[0]

    def _refresh_mode_options(self, waveform: str, preserve_current: bool = True):
        """Refresh mode choices when the selected waveform changes."""
        available_modes = self._get_available_modes_for_waveform(waveform)
        selected_mode = self._mode if preserve_current else None

        if selected_mode not in available_modes:
            selected_mode = self._get_default_mode_for_waveform(waveform)

        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItems(available_modes)
        self.mode_combo.setCurrentText(selected_mode)
        self.mode_combo.blockSignals(False)

        self._mode = selected_mode

    def _create_base_oscillator(self):
        """Create the base oscillator component based on current waveform."""
        if self._waveform == "Sine":
            return SineOscillator(
                self._base_frequency,
                gain_db=self._gain_db,
                phase=self._phase,
                mode=self._mode,
            )
        elif self._waveform == "Square":
            return SquareOscillator(
                self._base_frequency,
                gain_db=self._gain_db,
                phase=self._phase,
                mode=self._mode,
            )
        elif self._waveform == "Sawtooth":
            return SawtoothOscillator(
                self._base_frequency,
                gain_db=self._gain_db,
                phase=self._phase,
                mode=self._mode,
            )
        elif self._waveform == "Triangle":
            return TriangleOscillator(
                self._base_frequency,
                gain_db=self._gain_db,
                phase=self._phase,
                mode=self._mode,
            )
        else:
            # Default to sine
            return SineOscillator(
                self._base_frequency,
                gain_db=self._gain_db,
                phase=self._phase,
                mode=self._mode,
            )

    def _on_wave_changed(self, wave_type: str):
        """Handle waveform type change by recreating the component."""
        import logging
        logger = logging.getLogger(__name__)

        self._waveform = wave_type
        self._refresh_mode_options(wave_type)
        logger.debug(f"VCO: Waveform changed to {wave_type}")

        # Recreate the base oscillator with new waveform
        self.component = self._create_base_oscillator()
        # Update the port's component reference
        self.out_port.component = self.component

        self.parameter_changed.emit("waveform", wave_type)

    def _on_mode_changed(self, mode: str):
        """Handle oscillator mode changes by recreating the component."""
        import logging

        logger = logging.getLogger(__name__)

        if not mode:
            return

        self._mode = mode
        logger.debug(f"VCO: Mode changed to {mode}")

        self.component = self._create_base_oscillator()
        self.out_port.component = self.component

        self.parameter_changed.emit("mode", mode)

    def _on_frequency_changed(self):
        """Handle frequency knob change by updating the component."""
        import logging
        logger = logging.getLogger(__name__)

        new_freq = self.freq_knob.get_value()
        self._base_frequency = new_freq
        logger.debug(f"VCO: Frequency changed to {new_freq} Hz")

        # Hotswap: update frequency directly on the component if possible
        if hasattr(self.component, 'frequency'):
            self.component.frequency = new_freq

        self.parameter_changed.emit("frequency", new_freq)

    def _on_gain_changed(self):
        """Handle gain knob change by updating the component."""
        import logging
        logger = logging.getLogger(__name__)

        gain_value = self.gain_knob.get_value()
        self._gain_db = gain_value
        logger.debug(f"VCO: Gain changed to {gain_value} dB")

        # Hotswap: update gain_db directly on the component if possible
        if hasattr(self.component, 'gain_db'):
            self.component.gain_db = gain_value

        self.parameter_changed.emit("gain_db", gain_value)

    def get_required_inputs(self) -> list[str]:
        """Freq and Gain inputs are optional - VCO works as normal oscillator without
        them."""
        return []  # No required inputs - Freq and Gain are optional

    def process(self, num_samples: int = 1):
        """Process audio through the VCO with optional frequency and gain modulation.

        The VCO generates audio samples based on its current state and any
        connected modulation sources.

        Args:
            num_samples: Number of samples to generate
        """
        import logging
        import numpy as np

        logger = logging.getLogger(__name__)

        # Check if output is needed
        if not self.out_port.is_connected:
            return

        # Check for modulation inputs
        has_freq_mod = self.freq_input.is_connected
        has_gain_mod = self.gain_mod_input.is_connected

        # Read modulation signals if connected
        freq_signal = None
        gain_signal = None

        if has_freq_mod:
            freq_signal = self.freq_input.read(num_samples)
            logger.debug(f"VCO: Read freq modulation, shape={np.shape(freq_signal) if freq_signal is not None else None}")

        if has_gain_mod:
            gain_signal = self.gain_mod_input.read(num_samples)
            logger.debug(f"VCO: Read gain modulation, shape={np.shape(gain_signal) if gain_signal is not None else None}")

        # Generate samples based on modulation state
        if has_freq_mod or has_gain_mod:
            # Create ModulatedOscillator for this processing cycle
            modulators = []
            amp_mod = None
            freq_mod = None

            # Helper class to wrap signals for ModulatedOscillator
            class SignalGenerator:
                def __init__(self, signal):
                    self.signal = signal
                    self.idx = 0

                def get_samples(self, n):
                    if isinstance(self.signal, (int, float)):
                        return np.full(n, self.signal)
                    result = self.signal[self.idx:self.idx + n]
                    self.idx += n
                    return result

            if has_gain_mod and gain_signal is not None:
                gain_gen = SignalGenerator(gain_signal)
                modulators.append(gain_gen)
                amp_mod = lambda base_amp, cv_amp: base_amp * cv_amp

            if has_freq_mod and freq_signal is not None:
                freq_gen = SignalGenerator(freq_signal)
                modulators.append(freq_gen)
                freq_mod = lambda base_freq, cv_freq: cv_freq  # Use CV frequency directly

            # Create modulated oscillator
            modulated_osc = ModulatedOscillator(
                self.component,
                *modulators,
                amp_mod=amp_mod,
                freq_mod=freq_mod
            )

            samples = modulated_osc.get_samples(num_samples)
        else:
            # No modulation - use base oscillator
            samples = self.component.get_samples(num_samples)

        # Write to output port
        self.out_port.write(samples)

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

        This is called when connections change to provide immediate visual feedback.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Check if Freq input is connected
        has_freq_cv = self.freq_input.is_connected

        logger.debug(
            f"VCO update_knob_state: Freq port connected={has_freq_cv}"
        )

        if has_freq_cv:
            # Frequency controlled by CV - disable knob
            self.freq_knob.setEnabled(False)
            self.freq_knob.setStyleSheet("opacity: 0.5;")
            self.freq_knob.setToolTip("Frequency controlled by Freq input (CV)")
            logger.debug("VCO: Freq knob DISABLED")
        else:
            # No frequency CV - enable knob
            self.freq_knob.setEnabled(True)
            self.freq_knob.setStyleSheet("")
            self.freq_knob.setToolTip("Manual frequency control (Hz)")
            logger.debug("VCO: Freq knob ENABLED")

        # Check if Gain input is connected
        has_gain_cv = self.gain_mod_input.is_connected

        logger.debug(
            f"VCO update_knob_state: Gain port connected={has_gain_cv}"
        )

        if has_gain_cv:
            # Gain controlled by CV - disable knob
            self.gain_knob.setEnabled(False)
            self.gain_knob.setStyleSheet("opacity: 0.5;")
            self.gain_knob.setToolTip("Gain controlled by Gain input (CV)")
            logger.debug("VCO: Gain knob DISABLED")
        else:
            # No gain CV - enable knob
            self.gain_knob.setEnabled(True)
            self.gain_knob.setStyleSheet("")
            self.gain_knob.setToolTip("Manual gain control (dB)")
            logger.debug("VCO: Gain knob ENABLED")

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
        mode = self.mode_combo.currentText()
        base_freq = self.freq_knob.get_value()
        gain_db = self.gain_knob.get_value()
        phase = self.phase_slider.get_value()

        # Check for modulation inputs
        freq_modulator = (
            input_components[0]
            if input_components and len(input_components) > 0
            else None
        )
        gain_modulator = (
            modulation_components.get("Gain") if modulation_components else None
        )

        has_freq_mod = freq_modulator is not None
        has_gain_mod = gain_modulator is not None

        logger.debug(f"VCO: freq_mod={has_freq_mod}, gain_mod={has_gain_mod}")

        # Create the base oscillator
        if wave_type == "Sine":
            osc = SineOscillator(base_freq, gain_db=gain_db, phase=phase, mode=mode)
        elif wave_type == "Square":
            osc = SquareOscillator(base_freq, gain_db=gain_db, phase=phase, mode=mode)
        elif wave_type == "Sawtooth":
            osc = SawtoothOscillator(base_freq, gain_db=gain_db, phase=phase, mode=mode)
        elif wave_type == "Triangle":
            osc = TriangleOscillator(base_freq, gain_db=gain_db, phase=phase, mode=mode)
        else:
            raise ValueError(f"Unknown waveform type: {wave_type}")

        # Update UI state for frequency knob
        if has_freq_mod:
            self.freq_knob.setEnabled(False)
            self.freq_knob.setStyleSheet("opacity: 0.5;")
            self.freq_knob.setToolTip("Frequency controlled by Freq input (CV)")
            logger.debug(
                "VCO create_engine_component: Freq knob DISABLED (has modulation)"
            )
        else:
            self.freq_knob.setEnabled(True)
            self.freq_knob.setStyleSheet("")
            self.freq_knob.setToolTip("Manual frequency control (Hz)")
            logger.debug(
                "VCO create_engine_component: Freq knob ENABLED (no modulation)"
            )

        # Update UI state for gain knob
        if has_gain_mod:
            self.gain_knob.setEnabled(False)
            self.gain_knob.setStyleSheet("opacity: 0.5;")
            self.gain_knob.setToolTip("Gain controlled by Gain input (CV)")
            logger.debug(
                "VCO create_engine_component: Gain knob DISABLED (has modulation)"
            )
        else:
            self.gain_knob.setEnabled(True)
            self.gain_knob.setStyleSheet("")
            self.gain_knob.setToolTip("Manual gain control (dB)")
            logger.debug(
                "VCO create_engine_component: Gain knob ENABLED (no modulation)"
            )

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

            logger.debug(
                f"VCO: Creating ModulatedOscillator (freq_mod={has_freq_mod}, "
                f"gain_mod={has_gain_mod})"
            )

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
                osc, *modulators, amp_mod=amp_mod, freq_mod=freq_mod
            )

        # No modulation - return plain oscillator
        return osc
