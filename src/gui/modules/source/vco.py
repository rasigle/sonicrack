"""Modulated Oscillator module with pitch CV input.

This module provides an oscillator that can have its frequency controlled
by an external 1V/oct pitch CV source.
"""

from typing import Any, cast

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QLabel

from src.constants import DEFAULT_GAIN_DB
from src.engine import (
    PITCH_CV_REFERENCE_FREQUENCY,
    SawtoothOscillator,
    SineOscillator,
    SquareOscillator,
    TriangleOscillator,
)
from src.engine.generators.oscillators.oscillator_modulated import ModulatedOscillator
from src.engine.generators.oscillators.oscillator_ramp import SawtoothMode, TriangleMode
from src.engine.generators.oscillators.oscillator_sine import SineWaveMode
from src.engine.generators.oscillators.oscillator_square import SquareWaveMode
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.port import PortSignal
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, read_samples, str_parameter
from src.gui.module_registry import register_module
from src.gui.modules.source._oscillator_runtime import (
    RuntimeOscillator,
    _frequency_slew_values,
    render_with_frequency_ramp,
    smooth_control_signal,
)
from src.gui.ui_constants import (
    AUDIO_FREQUENCY_KNOB_CURVE,
    DEFAULT_PW_PERCENTAGE_VALUE,
    MAX_PW_PERCENTAGE_VALUE,
    MIN_PW_PERCENTAGE_VALUE,
)
from src.gui.widgets import HSlider, Knob
from src.gui.widgets.module_widget import ModuleWidget

VCO_PITCH_CV_SMOOTHING_MS = 5.0
VCO_DEFAULT_FM_AMOUNT_PERCENT = 0.0
VCO_MIN_FM_AMOUNT_PERCENT = -100.0
VCO_MAX_FM_AMOUNT_PERCENT = 100.0
VCO_FM_MODE_V_OCT = "1V/octave"
VCO_FM_MODE_LINEAR = "Linear"


def _as_sine_mode(mode: str) -> SineWaveMode:
    return cast(SineWaveMode, mode)


def _as_square_mode(mode: str) -> SquareWaveMode:
    return cast(SquareWaveMode, mode)


def _as_sawtooth_mode(mode: str) -> SawtoothMode:
    return cast(SawtoothMode, mode)


def _as_triangle_mode(mode: str) -> TriangleMode:
    return cast(TriangleMode, mode)


def apply_v_oct_offset(
    base_frequency: float | np.ndarray, pitch_cv: float | np.ndarray
) -> float | np.ndarray:
    """Apply a V/Oct pitch offset around the oscillator's base frequency."""
    frequencies = np.asarray(base_frequency, dtype=np.float64) * np.power(
        2.0, np.asarray(pitch_cv, dtype=np.float64)
    )
    if np.isscalar(base_frequency) and np.isscalar(pitch_cv):
        return float(frequencies)
    return frequencies.astype(np.float32)


def apply_linear_fm_offset(
    base_frequency: float | np.ndarray,
    fm_signal: float | np.ndarray,
    fm_amount_percent: float,
) -> float | np.ndarray:
    """Apply VCV-style linear FM as a C4-scaled signed Hz offset."""
    amount = float(fm_amount_percent) / 100.0
    frequencies = np.asarray(base_frequency, dtype=np.float64) + (
        np.asarray(fm_signal, dtype=np.float64) * amount * PITCH_CV_REFERENCE_FREQUENCY
    )
    frequencies = np.maximum(frequencies, 0.0)
    if np.isscalar(base_frequency) and np.isscalar(fm_signal):
        return float(frequencies)
    return frequencies.astype(np.float32)


def apply_exponential_fm_offset(
    base_frequency: float | np.ndarray,
    fm_signal: float | np.ndarray,
    fm_amount_percent: float,
) -> float | np.ndarray:
    """Apply VCV-style exponential FM as an additional 1V/oct pitch offset."""
    amount = float(fm_amount_percent) / 100.0
    frequencies = np.asarray(base_frequency, dtype=np.float64) * np.power(
        2.0, np.asarray(fm_signal, dtype=np.float64) * amount
    )
    if np.isscalar(base_frequency) and np.isscalar(fm_signal):
        return float(frequencies)
    return frequencies.astype(np.float32)


def apply_vcv_fm_offset(
    base_frequency: float | np.ndarray,
    fm_signal: float | np.ndarray,
    fm_amount_percent: float,
    fm_mode: str,
) -> float | np.ndarray:
    """Apply FM using the same mode semantics as VCV Rack Fundamental VCO."""
    if fm_mode == VCO_FM_MODE_LINEAR:
        return apply_linear_fm_offset(base_frequency, fm_signal, fm_amount_percent)
    return apply_exponential_fm_offset(base_frequency, fm_signal, fm_amount_percent)


def _as_frequency_buffer(values: float | np.ndarray) -> np.ndarray:
    """Normalize scalar-or-array frequency results to a 1D float32 buffer."""
    return np.asarray(values, dtype=np.float32).reshape(-1)


@register_module()
class ModulatedOscillatorModule(ModuleWidget):
    """Oscillator with 1V/oct pitch CV input.

    This oscillator can have its frequency controlled by an external 1V/oct CV
    source, making it usable with MIDI and sequencer pitch outputs.

    Inputs:
        - Freq: 1V/oct pitch CV input (e.g., from MIDI Input)

    Outputs:
        - Out: Audio output
    """

    runtime_kind = "vco"

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
            width=280,
            height=275,
            color=QColor(100, 140, 220),
        )

        # Initialize default parameters FIRST (before creating component)
        self._waveform = "Sine"
        self._mode = self._get_default_mode_for_waveform(self._waveform)
        self._base_frequency = 440.0
        self._gain_db = DEFAULT_GAIN_DB
        self._phase = 0.0
        self._pulsewidth = DEFAULT_PW_PERCENTAGE_VALUE / 100
        self._fm_amount = VCO_DEFAULT_FM_AMOUNT_PERCENT
        self._fm_mode = VCO_FM_MODE_V_OCT

        # Create the base oscillator component FIRST
        self.component = self._create_base_oscillator()

        # Add ports with component reference
        # Note: Input ports don't have components (they receive signals)
        # Output port has the component reference
        self.freq_input = self.add_input("V/Oct", signal=PortSignal.PITCH_CV)
        self.fm_input = self.add_input("FM", signal=PortSignal.CONTROL_CV)
        self.gain_mod_input = self.add_input("Gain", signal=PortSignal.CONTROL_CV)
        self.out_port = self.add_output(
            "Out",
            component=self.component,
            signal=PortSignal.AUDIO,
        )

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout(spacing=6)

        # Waveform selector
        wave_layout = QHBoxLayout()
        wave_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wave_layout.addWidget(QLabel("Wave:"))
        self.wave_combo = QtWidgets.QComboBox()
        self.wave_combo.addItems(["Sine", "Square", "Sawtooth", "Triangle"])
        self.wave_combo.setCurrentText(self._waveform)
        self.wave_combo.currentTextChanged.connect(self._on_wave_changed)
        wave_layout.addWidget(self.wave_combo)
        layout.addLayout(wave_layout)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._refresh_mode_options(self._waveform, preserve_current=False)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        fm_mode_layout = QHBoxLayout()
        fm_mode_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fm_mode_layout.addWidget(QLabel("FM Mode:"))
        self.fm_mode_combo = QtWidgets.QComboBox()
        self.fm_mode_combo.addItems([VCO_FM_MODE_V_OCT, VCO_FM_MODE_LINEAR])
        self.fm_mode_combo.setCurrentText(self._fm_mode)
        self.fm_mode_combo.currentTextChanged.connect(self._on_fm_mode_changed)
        fm_mode_layout.addWidget(self.fm_mode_combo)
        layout.addLayout(fm_mode_layout)

        # Frequency control (base frequency when no modulation)
        knobs_layout = QHBoxLayout()
        knobs_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.freq_knob = Knob(
            label="Pitch (Hz)",
            min_value=11,
            max_value=6000,
            default_value=self._base_frequency,
            curve_points=AUDIO_FREQUENCY_KNOB_CURVE,
        )
        self.freq_knob.setToolTip(
            "Base frequency (Hz)\nActive when Freq input is disconnected"
        )
        self.freq_knob.value_changed.connect(self._on_frequency_changed)
        knobs_layout.addWidget(self.freq_knob)

        # Gain in dB
        self.gain_knob = Knob(
            label="Gain (dB)",
            min_value=-60,
            max_value=12,
            default_value=DEFAULT_GAIN_DB,
            logarithmic=False,
        )
        self.gain_knob.setToolTip(
            "Oscillator gain (dB)\nRange: -60 to +12 dB\nDefault: -20 dB"
        )
        self.gain_knob.value_changed.connect(self._on_gain_changed)
        knobs_layout.addWidget(self.gain_knob)

        self.fm_amount_knob = Knob(
            label="FM Amt %",
            min_value=VCO_MIN_FM_AMOUNT_PERCENT,
            max_value=VCO_MAX_FM_AMOUNT_PERCENT,
            default_value=self._fm_amount,
            logarithmic=False,
        )
        self.fm_amount_knob.setToolTip(
            "Signed FM depth. In 1V/octave mode this scales pitch CV; "
            "in Linear mode this scales C4 Hz per volt."
        )
        self.fm_amount_knob.value_changed.connect(self._on_fm_amount_changed)
        knobs_layout.addWidget(self.fm_amount_knob)

        self.pulsewidth_knob = Knob(
            label="PW",
            min_value=MIN_PW_PERCENTAGE_VALUE / 100,
            max_value=MAX_PW_PERCENTAGE_VALUE / 100,
            default_value=self._pulsewidth,
        )
        self.pulsewidth_knob.setToolTip("Square pulse width")
        self.pulsewidth_knob.value_changed.connect(self._on_pulsewidth_changed)
        knobs_layout.addWidget(self.pulsewidth_knob)

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
        self.register_parameter(
            "fm_mode",
            self.fm_mode_combo,
            getter="currentText",
            setter="setCurrentText",
        )
        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("gain_db", self.gain_knob)
        self.register_parameter("fm_amount", self.fm_amount_knob)
        self.register_parameter("phase", self.phase_slider)
        self.register_parameter("pulsewidth", self.pulsewidth_knob)

        self.component = self.create_engine_component()
        self._runtime_oscillator_shape: tuple[str, str] | None = None
        self._last_runtime_frequency = self._base_frequency
        self._last_pitch_cv: float | None = None

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
            "Sine": "pure",
            "Square": "vcv",
            "Sawtooth": "vcv",
            "Triangle": "pure",
        }
        available_modes = cls._get_available_modes_for_waveform(waveform)
        preferred = preferred_defaults.get(waveform, available_modes[0])
        return preferred if preferred in available_modes else available_modes[0]

    def _refresh_mode_options(self, waveform: str, preserve_current: bool = True):
        """Refresh mode choices when the selected waveform changes."""
        available_modes = self._get_available_modes_for_waveform(waveform)
        selected_mode = self._normalize_mode_for_waveform(
            waveform, self._mode if preserve_current else None
        )

        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        self.mode_combo.addItems(available_modes)
        self.mode_combo.setCurrentText(selected_mode)
        self.mode_combo.blockSignals(False)

        self._mode = selected_mode

    @classmethod
    def _normalize_mode_for_waveform(cls, waveform: str, mode: str | None) -> str:
        """Return a valid mode for the selected waveform."""
        available_modes = cls._get_available_modes_for_waveform(waveform)
        if mode in available_modes:
            return str(mode)
        return cls._get_default_mode_for_waveform(waveform)

    def _create_base_oscillator(self):
        """Create the base oscillator component based on current waveform."""
        if self._waveform == "Sine":
            return SineOscillator(
                self._base_frequency,
                gain_db=self._gain_db,
                phase=self._phase,
                mode=_as_sine_mode(self._mode),
            )
        elif self._waveform == "Square":
            return SquareOscillator(
                self._base_frequency,
                gain_db=self._gain_db,
                phase=self._phase,
                pulsewidth=self._pulsewidth,
                mode=_as_square_mode(self._mode),
            )
        elif self._waveform == "Sawtooth":
            return SawtoothOscillator(
                self._base_frequency,
                gain_db=self._gain_db,
                phase=self._phase,
                mode=_as_sawtooth_mode(self._mode),
            )
        elif self._waveform == "Triangle":
            return TriangleOscillator(
                self._base_frequency,
                gain_db=self._gain_db,
                phase=self._phase,
                mode=_as_triangle_mode(self._mode),
            )
        else:
            # Default to sine
            return SineOscillator(
                self._base_frequency,
                gain_db=self._gain_db,
                phase=self._phase,
                mode=_as_sine_mode(self._mode),
            )

    def _on_wave_changed(self, wave_type: str):
        """Handle waveform type change by recreating the component."""
        import logging

        logger = logging.getLogger(__name__)

        previous_mode = self._mode
        self._waveform = wave_type
        self._refresh_mode_options(wave_type)
        logger.debug(f"VCO: Waveform changed to {wave_type}")

        # Recreate the base oscillator with new waveform
        self.component = self._create_base_oscillator()
        self.out_port.component = self.component

        self.parameter_changed.emit("waveform", wave_type)
        if self._mode != previous_mode:
            self.parameter_changed.emit("mode", self._mode)

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

        # Hotswap the active oscillator without rebuilding the widget.
        if self.component is not None:
            if isinstance(self.component, ModulatedOscillator):
                self.component.oscillator._freq = new_freq
                self.component.oscillator.frequency = new_freq
            else:
                self.component.frequency = new_freq

        self.parameter_changed.emit("frequency", new_freq)

    def _on_gain_changed(self):
        """Handle gain knob change by updating the component."""
        import logging

        logger = logging.getLogger(__name__)

        gain_value = self.gain_knob.get_value()
        self._gain_db = gain_value
        logger.debug(f"VCO: Gain changed to {gain_value} dB")

        # Hotswap the active oscillator without rebuilding the widget.
        if self.component is not None:
            self.component.gain_db = gain_value

        self.parameter_changed.emit("gain_db", gain_value)

    def _on_fm_amount_changed(self):
        """Handle FM depth changes."""
        fm_amount = self.fm_amount_knob.get_value()
        self._fm_amount = fm_amount
        self.parameter_changed.emit("fm_amount", fm_amount)

    def _on_fm_mode_changed(self, fm_mode: str):
        """Handle FM mode changes."""
        self._fm_mode = fm_mode
        self.parameter_changed.emit("fm_mode", fm_mode)

    def _on_pulsewidth_changed(self):
        """Handle square pulse width changes."""
        pulsewidth = self.pulsewidth_knob.get_value()
        self._pulsewidth = pulsewidth
        if isinstance(self.component, SquareOscillator):
            self.component.pulsewidth = pulsewidth
        self.parameter_changed.emit("pulsewidth", pulsewidth)

    def get_required_inputs(self) -> list[str]:
        """Freq and Gain inputs are optional - VCO works as normal oscillator without
        them."""
        return []  # No required inputs - Freq and Gain are optional

    def get_modulation_inputs(self) -> list[str]:
        """VCO accepts modulation on Gain and FM ports."""
        return ["Gain", "FM"]

    def get_cv_range(self, port_name: str = "Gain") -> tuple[float, float]:
        """Return the expected CV range for VCO modulation inputs.

        Returns:
            Gain: (0.0, 1.0) - unipolar range for amplitude modulation
            FM: (-1.0, 1.0) - bipolar range scaled by FM Amt %
        """
        if port_name == "FM":
            return -1.0, 1.0
        return 0.0, 1.0

    def update_knob_state(self):
        """Update knob enabled state based on port connections.

        This is called when connections change to provide immediate visual feedback.
        """
        import logging

        logger = logging.getLogger(__name__)

        # Check if Freq input is connected
        has_freq_cv = self.freq_input.is_connected

        logger.debug(f"VCO update_knob_state: Freq port connected={has_freq_cv}")

        self.freq_knob.setEnabled(True)
        self.freq_knob.setStyleSheet("")
        if has_freq_cv:
            self.freq_knob.setToolTip(
                "Base frequency (Hz); V/Oct input transposes this pitch"
            )
            logger.debug("VCO: Freq knob ENABLED with V/Oct pitch input")
        else:
            self.freq_knob.setToolTip("Manual frequency control (Hz)")
            logger.debug("VCO: Freq knob ENABLED")

        # Check if Gain input is connected
        has_gain_cv = self.gain_mod_input.is_connected
        has_fm = self.fm_input.is_connected

        logger.debug(f"VCO update_knob_state: Gain port connected={has_gain_cv}")
        logger.debug(f"VCO update_knob_state: FM port connected={has_fm}")

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

        if has_fm:
            self.fm_amount_knob.setToolTip("Signed FM depth for the connected FM input")
        else:
            self.fm_amount_knob.setToolTip(
                "Signed FM depth. In 1V/octave mode this scales pitch CV; "
                "in Linear mode this scales C4 Hz per volt."
            )

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
        mode = self._normalize_mode_for_waveform(
            wave_type, self.mode_combo.currentText()
        )
        base_freq = self.freq_knob.get_value()
        gain_db = self.gain_knob.get_value()
        fm_amount = self.fm_amount_knob.get_value()
        fm_mode = self.fm_mode_combo.currentText()
        phase = self.phase_slider.get_value()
        pulsewidth = self.pulsewidth_knob.get_value()

        # Check for modulation inputs
        freq_modulator = (
            input_components[0]
            if input_components and len(input_components) > 0
            else None
        )
        gain_modulator = (
            modulation_components.get("Gain") if modulation_components else None
        )
        fm_modulator = (
            modulation_components.get("FM") if modulation_components else None
        )

        has_freq_mod = freq_modulator is not None
        has_gain_mod = gain_modulator is not None
        has_fm_mod = fm_modulator is not None and fm_amount != 0.0

        logger.debug(
            f"VCO: freq_mod={has_freq_mod}, gain_mod={has_gain_mod}, "
            f"fm_mod={has_fm_mod}"
        )

        # Create the base oscillator
        if wave_type == "Sine":
            osc = SineOscillator(
                frequency=base_freq,
                gain_db=gain_db,
                phase=phase,
                mode=_as_sine_mode(mode),
            )
        elif wave_type == "Square":
            osc = SquareOscillator(
                frequency=base_freq,
                gain_db=gain_db,
                phase=phase,
                pulsewidth=pulsewidth,
                mode=_as_square_mode(mode),
            )
        elif wave_type == "Sawtooth":
            osc = SawtoothOscillator(
                frequency=base_freq,
                gain_db=gain_db,
                phase=phase,
                mode=_as_sawtooth_mode(mode),
            )
        elif wave_type == "Triangle":
            osc = TriangleOscillator(
                frequency=base_freq,
                gain_db=gain_db,
                phase=phase,
                mode=_as_triangle_mode(mode),
            )
        else:
            raise ValueError(f"Unknown waveform type: {wave_type}")

        # Update UI state for frequency knob. V/Oct transposes the knob's base pitch.
        self.freq_knob.setEnabled(True)
        self.freq_knob.setStyleSheet("")
        if has_freq_mod:
            self.freq_knob.setToolTip(
                "Base frequency (Hz); V/Oct input transposes this pitch"
            )
            logger.debug("VCO create_engine_component: Freq knob ENABLED with V/Oct")
        else:
            self.freq_knob.setToolTip("Manual frequency control (Hz)")
            logger.debug("VCO create_engine_component: Freq knob ENABLED")

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

        # If we have frequency, FM, or gain modulation, create ModulatedOscillator
        if has_freq_mod or has_gain_mod or has_fm_mod:
            # Frequency modulation function
            def freq_mod_func(base_freq, pitch_cv):
                """Apply 1V/oct pitch CV as an offset around the base frequency."""
                return apply_v_oct_offset(base_freq, pitch_cv)

            def fm_mod_func(current_freq, fm_signal):
                """Apply VCV-style FM after base pitch CV."""
                return apply_vcv_fm_offset(
                    current_freq,
                    fm_signal,
                    fm_amount,
                    fm_mode,
                )

            # Amplitude modulation function (for gain modulation)
            def amp_mod_func(base_amp, cv_amp):
                """Use CV value to scale amplitude."""
                # cv_amp is typically [0, 1] from envelope
                # Multiply base amplitude by CV value
                return base_amp * cv_amp

            logger.debug(
                f"VCO: Creating ModulatedOscillator (freq_mod={has_freq_mod}, "
                f"gain_mod={has_gain_mod}, fm_mod={has_fm_mod})"
            )

            # Build modulator list based on what's connected
            modulators = []
            amp_mod = None
            freq_mod = None
            fm_mod = None

            if has_gain_mod:
                modulators.append(gain_modulator)
                amp_mod = amp_mod_func

            if has_freq_mod:
                modulators.append(freq_modulator)
                freq_mod = freq_mod_func

            if has_fm_mod:
                modulators.append(fm_modulator)
                fm_mod = fm_mod_func

            return ModulatedOscillator(
                osc,
                *modulators,
                amp_mod=amp_mod,
                freq_mod=freq_mod,
                fm_mod=fm_mod,
            )

        # No modulation - return plain oscillator
        return osc

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Render VCO output for the current engine cycle."""
        wave_type = str_parameter(parameters, "waveform", self.wave_combo.currentText)
        mode = self._normalize_mode_for_waveform(
            wave_type,
            str_parameter(parameters, "mode", self.mode_combo.currentText),
        )
        fm_mode = str_parameter(parameters, "fm_mode", self.fm_mode_combo.currentText)
        frequency = float_parameter(parameters, "frequency", self.freq_knob.get_value)
        gain_db = float_parameter(parameters, "gain_db", self.gain_knob.get_value)
        fm_amount = float_parameter(
            parameters, "fm_amount", self.fm_amount_knob.get_value
        )
        phase = float_parameter(parameters, "phase", self.phase_slider.get_value)
        pulsewidth = float_parameter(
            parameters, "pulsewidth", self.pulsewidth_knob.get_value
        )

        oscillator_shape = (wave_type, mode)
        if self.component is None or oscillator_shape != self._runtime_oscillator_shape:
            self.component = self._create_runtime_base_oscillator(
                wave_type, mode, frequency, gain_db, phase, pulsewidth
            )
            self._runtime_oscillator_shape = oscillator_shape
            self._last_runtime_frequency = frequency
            self._last_pitch_cv = None
        else:
            self.component.gain_db = gain_db
            self.component.phase = phase
            if isinstance(self.component, SquareOscillator):
                self.component.pulsewidth = pulsewidth

        freq_signal = None
        fm_signal = None
        gain_signal = None
        if self.gain_mod_input.is_connected:
            gain_signal = read_samples(self.gain_mod_input, num_samples)
        if self.fm_input.is_connected and fm_amount != 0.0:
            fm_signal = read_samples(self.fm_input, num_samples)

        if self.freq_input.is_connected:
            freq_signal = read_samples(self.freq_input, num_samples)
            samples = self._render_frequency_signal(
                frequency,
                freq_signal,
                fm_signal=fm_signal,
                fm_amount=fm_amount,
                fm_mode=fm_mode,
            )
        elif fm_signal is not None:
            base_frequencies, rendered_frequency = self._build_base_frequency_ramp(
                frequency,
                num_samples,
            )
            samples = self._render_frequency_signal(
                base_frequencies,
                None,
                fm_signal=fm_signal,
                fm_amount=fm_amount,
                fm_mode=fm_mode,
            )
            self._last_runtime_frequency = rendered_frequency
        else:
            self._last_pitch_cv = None
            samples, rendered_frequency = render_with_frequency_ramp(
                cast(RuntimeOscillator, self.component),
                self._last_runtime_frequency,
                frequency,
                num_samples,
            )
            self._last_runtime_frequency = rendered_frequency

        if gain_signal is not None:
            samples = samples * gain_signal

        self.out_port.write(samples)

    def _render_frequency_signal(
        self,
        base_frequency: float | np.ndarray,
        pitch_cv_signal: np.ndarray | None,
        *,
        fm_signal: np.ndarray | None = None,
        fm_amount: float = 0.0,
        fm_mode: str = VCO_FM_MODE_V_OCT,
    ) -> np.ndarray:
        """Render a pitch-CV/FM buffer using bulk frequency processing.

        This method now uses the bulk frequency API which is 30-50x faster
        than the previous per-sample loop approach.
        """
        # Smooth pitch CV if present
        if pitch_cv_signal is not None:
            pitch_cv_signal, self._last_pitch_cv = smooth_control_signal(
                pitch_cv_signal,
                self._last_pitch_cv,
                float(getattr(self.component, "sample_rate", 44100.0)),
                VCO_PITCH_CV_SMOOTHING_MS,
            )
        else:
            length = len(fm_signal) if fm_signal is not None else 512
            pitch_cv_signal = np.zeros(length, dtype=np.float32)

        # Calculate final frequencies with pitch CV and FM
        frequencies = apply_v_oct_offset(base_frequency, pitch_cv_signal)
        if fm_signal is not None:
            frequencies = apply_vcv_fm_offset(
                frequencies, fm_signal, fm_amount, fm_mode
            )

        # Use bulk frequency API (30-50x faster than loop!)
        samples = self.component.process_frequency_buffer(frequencies)

        # Track last frequency for smooth transitions
        if len(frequencies) > 0:
            self._last_runtime_frequency = float(frequencies[-1])

        return samples

    def _build_base_frequency_ramp(
        self, target_frequency: float, num_samples: int
    ) -> tuple[np.ndarray, float]:
        """Build the base-frequency trajectory for FM-only rendering."""
        sample_rate = float(getattr(self.component, "sample_rate", 44100.0))
        if self._last_runtime_frequency == target_frequency:
            frequencies = np.full(num_samples, target_frequency, dtype=np.float64)
        else:
            frequencies = _frequency_slew_values(
                self._last_runtime_frequency,
                target_frequency,
                num_samples,
                sample_rate,
                35.0,
            )
        final_frequency = (
            float(frequencies[-1]) if len(frequencies) > 0 else target_frequency
        )
        return frequencies, final_frequency

    @staticmethod
    def _create_runtime_base_oscillator(
        wave_type: str,
        mode: str,
        frequency: float,
        gain_db: float,
        phase: float,
        pulsewidth: float,
    ):
        """Create the VCO base oscillator from a runtime parameter snapshot."""
        if wave_type == "Sine":
            return SineOscillator(
                frequency, gain_db=gain_db, phase=phase, mode=_as_sine_mode(mode)
            )
        if wave_type == "Square":
            return SquareOscillator(
                frequency,
                gain_db=gain_db,
                phase=phase,
                pulsewidth=pulsewidth,
                mode=_as_square_mode(mode),
            )
        if wave_type == "Sawtooth":
            return SawtoothOscillator(
                frequency,
                gain_db=gain_db,
                phase=phase,
                mode=_as_sawtooth_mode(mode),
            )
        if wave_type == "Triangle":
            return TriangleOscillator(
                frequency,
                gain_db=gain_db,
                phase=phase,
                mode=_as_triangle_mode(mode),
            )
        raise ValueError(f"Unknown waveform type: {wave_type}")
