"""Voltage-Controlled Oscillator (VCO) with V/Oct pitch and FM inputs."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QLabel
from soniclab import (
    PITCH_CV_REFERENCE_FREQUENCY,
    SawtoothOscillator,
    SineOscillator,
    SquareOscillator,
    TriangleOscillator,
)
from soniclab.generators.oscillators.oscillator_modulated import ModulatedOscillator
from soniclab.generators.oscillators.oscillator_ramp import SawtoothMode, TriangleMode
from soniclab.generators.oscillators.oscillator_sine import SineWaveMode
from soniclab.generators.oscillators.oscillator_square import SquareWaveMode

from sonicrack.config.audio_config import audio_config
from sonicrack.constants import (
    AUDIO_FREQUENCY_KNOB_CURVE,
    DEFAULT_PW_PERCENTAGE_VALUE,
    MAX_PW_PERCENTAGE_VALUE,
    MIN_PW_PERCENTAGE_VALUE,
)
from sonicrack.gui.modules.source._oscillator_runtime import (
    RuntimeOscillator,
    render_with_frequency_ramp,
    smooth_control_signal,
)
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples, str_parameter
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)

VCO_PITCH_CV_SMOOTHING_MS = 5.0
VCO_DEFAULT_GAIN_DB = 0.0
VCO_DEFAULT_FREQUENCY = 440.0
VCO_MIN_FREQUENCY = 11.0
VCO_MAX_FREQUENCY = 6000.0
VCO_DEFAULT_FM_AMOUNT_PERCENT = 0.0
VCO_MIN_FM_AMOUNT_PERCENT = -100.0
VCO_MAX_FM_AMOUNT_PERCENT = 100.0
VCO_FM_MODE_V_OCT = "1V/octave"
VCO_FM_MODE_LINEAR = "Linear"
VCO_WAVEFORMS = ("Sine", "Square", "Sawtooth", "Triangle")

# Sensible default modes per waveform (must exist in engine mode lists).
_PREFERRED_MODES = {
    "Sine": "pure",
    "Square": "vcv",
    "Sawtooth": "vcv",
    "Triangle": "pure",
}


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


def create_vco_oscillator(
    wave_type: str,
    mode: str,
    frequency: float,
    *,
    pulsewidth: float = DEFAULT_PW_PERCENTAGE_VALUE / 100,
    phase: float = 0.0,
    sample_rate: float | None = None,
) -> SineOscillator | SquareOscillator | SawtoothOscillator | TriangleOscillator:
    """Create a base oscillator for the selected waveform and mode."""
    sr = float(sample_rate if sample_rate is not None else audio_config.sample_rate)
    common = {
        "gain_db": VCO_DEFAULT_GAIN_DB,
        "phase": phase,
        "sample_rate": sr,
    }

    if wave_type == "Sine":
        return SineOscillator(frequency, mode=_as_sine_mode(mode), **common)
    if wave_type == "Square":
        return SquareOscillator(
            frequency,
            pulsewidth=pulsewidth,
            mode=_as_square_mode(mode),
            **common,
        )
    if wave_type == "Sawtooth":
        return SawtoothOscillator(frequency, mode=_as_sawtooth_mode(mode), **common)
    if wave_type == "Triangle":
        return TriangleOscillator(frequency, mode=_as_triangle_mode(mode), **common)
    raise ValueError(f"Unknown waveform type: {wave_type}")


@register_module()
class ModulatedOscillatorModule(ModuleWidget):
    """Oscillator with 1V/oct pitch CV and optional FM input.

    Inputs:
        - V/Oct: 1V/oct pitch transpose around the Pitch knob base frequency
        - FM: bipolar FM (depth via FM Amt %, mode via FM Mode)

    Outputs:
        - Out: Audio
    """

    runtime_kind = "vco"

    metadata = ModuleMetadata(
        title="VCO",
        category=ModuleCategory.SOURCE,
        description="Voltage-Controlled Oscillator with frequency modulation input",
        version="1.0.0",
        author="SonicRack",
    )

    def __init__(self) -> None:
        super().__init__(
            width=280,
            height=245,
            color=QColor(100, 140, 220),
        )

        self._waveform = "Sine"
        self._mode = self._get_default_mode_for_waveform(self._waveform)
        self._base_frequency = VCO_DEFAULT_FREQUENCY
        self._pulsewidth = DEFAULT_PW_PERCENTAGE_VALUE / 100
        self._fm_amount = VCO_DEFAULT_FM_AMOUNT_PERCENT

        self.component = create_vco_oscillator(
            self._waveform,
            self._mode,
            self._base_frequency,
            pulsewidth=self._pulsewidth,
        )

        self.freq_input = self.add_input("V/Oct", signal=PortSignal.PITCH_CV)
        self.fm_input = self.add_input("FM", signal=PortSignal.CONTROL_CV)
        self.out_port = self.add_output(
            "Out",
            component=self.component,
            signal=PortSignal.AUDIO,
        )

        # Setup option: right-click module header → FM Mode
        self.fm_mode_param = self.register_menu_choice(
            "fm_mode",
            "FM Mode",
            [VCO_FM_MODE_V_OCT, VCO_FM_MODE_LINEAR],
            VCO_FM_MODE_V_OCT,
            tooltip=(
                "1V/octave: pitch-style exponential FM depth.\n"
                "Linear: Hz offset scaled by C4 reference frequency."
            ),
        )

        layout = self._begin_controls(spacing=6)

        wave_layout = QHBoxLayout()
        wave_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wave_layout.addWidget(QLabel("Wave:"))
        self.wave_combo = QtWidgets.QComboBox()
        self.wave_combo.addItems(list(VCO_WAVEFORMS))
        self.wave_combo.setCurrentText(self._waveform)
        self.wave_combo.currentTextChanged.connect(self._on_wave_changed)
        wave_layout.addWidget(self.wave_combo)
        layout.addLayout(wave_layout)

        mode_layout = QHBoxLayout()
        mode_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._refresh_mode_options(self._waveform, preserve_current=False)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        knobs_layout = QHBoxLayout()
        knobs_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.freq_knob = Knob(
            label="Pitch (Hz)",
            min_value=VCO_MIN_FREQUENCY,
            max_value=VCO_MAX_FREQUENCY,
            default_value=self._base_frequency,
            curve_points=AUDIO_FREQUENCY_KNOB_CURVE,
        )
        self.freq_knob.setToolTip("Base frequency (Hz); V/Oct transposes this pitch")
        self.freq_knob.value_changed.connect(self._on_frequency_changed)
        knobs_layout.addWidget(self.freq_knob)

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

        self._finish_controls(layout)

        self.register_parameter(
            "waveform", self.wave_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter(
            "mode", self.mode_combo, getter="currentText", setter="setCurrentText"
        )
        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("fm_amount", self.fm_amount_knob)
        self.register_parameter("pulsewidth", self.pulsewidth_knob)

        self._runtime_oscillator_shape: tuple[str, str] | None = (
            self._waveform,
            self._mode,
        )
        self._last_runtime_frequency = self._base_frequency
        self._last_pitch_cv: float | None = None

        self._install_sample_rate_listener()

        self._update_pulsewidth_visibility()
        self.update_knob_state()

    # ------------------------------------------------------------------ modes

    @staticmethod
    def _get_available_modes_for_waveform(waveform: str) -> list[str]:
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
        available_modes = cls._get_available_modes_for_waveform(waveform)
        preferred = _PREFERRED_MODES.get(waveform, available_modes[0])
        return preferred if preferred in available_modes else available_modes[0]

    def _refresh_mode_options(self, waveform: str, preserve_current: bool = True):
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
        available_modes = cls._get_available_modes_for_waveform(waveform)
        if mode in available_modes:
            return str(mode)
        return cls._get_default_mode_for_waveform(waveform)

    # ------------------------------------------------------------------ rebuild

    def _rebuild_base_oscillator(self) -> None:
        """Recreate the plain base oscillator and attach it to Out."""
        self.component = create_vco_oscillator(
            self._waveform,
            self._mode,
            self._base_frequency,
            pulsewidth=self._pulsewidth,
        )
        self.out_port.component = self.component
        self._runtime_oscillator_shape = (self._waveform, self._mode)
        self._last_runtime_frequency = self._base_frequency
        self._last_pitch_cv = None

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        logger.debug("VCO: sample rate → %s Hz", new_sample_rate)
        if self.component is not None and hasattr(self.component, "sample_rate"):
            self.component.sample_rate = new_sample_rate
        # Force runtime shape rebuild on next process so phase/mode stay coherent.
        self._runtime_oscillator_shape = None

    def _update_pulsewidth_visibility(self) -> None:
        """Pulse width only applies to square waves."""
        is_square = self._waveform == "Square"
        self.pulsewidth_knob.setEnabled(is_square)
        self.pulsewidth_knob.setStyleSheet("" if is_square else "opacity: 0.4;")
        self.pulsewidth_knob.setToolTip(
            "Square pulse width" if is_square else "Pulse width (Square waveform only)"
        )

    # ------------------------------------------------------------------ UI handlers

    def _on_wave_changed(self, wave_type: str) -> None:
        if not wave_type:
            return
        previous_mode = self._mode
        self._waveform = wave_type
        self._refresh_mode_options(wave_type)
        logger.debug("VCO: waveform → %s (mode=%s)", wave_type, self._mode)

        self._rebuild_base_oscillator()
        self._update_pulsewidth_visibility()

        self.parameter_changed.emit("waveform", wave_type)
        if self._mode != previous_mode:
            self.parameter_changed.emit("mode", self._mode)

    def _on_mode_changed(self, mode: str) -> None:
        if not mode:
            return
        self._mode = mode
        logger.debug("VCO: mode → %s", mode)
        self._rebuild_base_oscillator()
        self.parameter_changed.emit("mode", mode)

    def _on_frequency_changed(self) -> None:
        new_freq = self.freq_knob.get_value()
        self._base_frequency = new_freq
        logger.debug("VCO: frequency → %.2f Hz", new_freq)

        if self.component is not None:
            if isinstance(self.component, ModulatedOscillator):
                self.component.oscillator._freq = new_freq
                self.component.oscillator.frequency = new_freq
            else:
                self.component.frequency = new_freq

        self.parameter_changed.emit("frequency", new_freq)

    def _on_fm_amount_changed(self) -> None:
        self._fm_amount = self.fm_amount_knob.get_value()
        self.parameter_changed.emit("fm_amount", self._fm_amount)

    def _on_pulsewidth_changed(self) -> None:
        pulsewidth = self.pulsewidth_knob.get_value()
        self._pulsewidth = pulsewidth
        if isinstance(self.component, SquareOscillator):
            self.component.pulsewidth = pulsewidth
        self.parameter_changed.emit("pulsewidth", pulsewidth)

    # ------------------------------------------------------------------ module API

    def get_required_inputs(self) -> list[str]:
        """V/Oct and FM are optional; VCO free-runs on the Pitch knob alone."""
        return []

    def get_modulation_inputs(self) -> list[str]:
        """VCO accepts modulation on the FM port."""
        return ["FM"]

    def get_cv_range(self, port_name: str = "FM") -> tuple[float, float]:
        """Expected CV ranges for VCO modulation inputs."""
        if port_name in ("V/Oct", "Freq"):
            # Relative 1V/oct offset in volts (typical keyboard/sequencer span).
            return -5.0, 5.0
        if port_name == "FM":
            return -1.0, 1.0
        return 0.0, 1.0

    def update_knob_state(self) -> None:
        """Refresh tooltips from port connection state."""
        has_freq_cv = bool(getattr(self.freq_input, "is_connected", False))
        has_fm = bool(getattr(self.fm_input, "is_connected", False))

        self.freq_knob.setEnabled(True)
        self.freq_knob.setStyleSheet("")
        if has_freq_cv:
            self.freq_knob.setToolTip(
                "Base frequency (Hz); V/Oct input transposes this pitch"
            )
        else:
            self.freq_knob.setToolTip("Manual base frequency control (Hz)")

        if has_fm:
            self.fm_amount_knob.setToolTip("Signed FM depth for the connected FM input")
        else:
            self.fm_amount_knob.setToolTip(
                "Signed FM depth. In 1V/octave mode this scales pitch CV; "
                "in Linear mode this scales C4 Hz per volt."
            )

        logger.debug("VCO update_knob_state: V/Oct=%s FM=%s", has_freq_cv, has_fm)

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create oscillator (optionally modulated) for the patch-compiler path.

        Runtime audio uses ``process_runtime``; this remains for tests and
        legacy component mapping.
        """
        wave_type = self.wave_combo.currentText()
        mode = self._normalize_mode_for_waveform(
            wave_type, self.mode_combo.currentText()
        )
        base_freq = self.freq_knob.get_value()
        fm_amount = self.fm_amount_knob.get_value()
        fm_mode = self.fm_mode_param.get_value()
        pulsewidth = self.pulsewidth_knob.get_value()

        freq_modulator = (
            input_components[0]
            if input_components and len(input_components) > 0
            else None
        )
        fm_modulator = (
            modulation_components.get("FM") if modulation_components else None
        )

        has_freq_mod = freq_modulator is not None
        has_fm_mod = fm_modulator is not None and fm_amount != 0.0

        osc = create_vco_oscillator(wave_type, mode, base_freq, pulsewidth=pulsewidth)

        if not (has_freq_mod or has_fm_mod):
            return osc

        def freq_mod_func(base_freq_val, pitch_cv):
            return apply_v_oct_offset(base_freq_val, pitch_cv)

        def fm_mod_func(current_freq, fm_signal):
            return apply_vcv_fm_offset(current_freq, fm_signal, fm_amount, fm_mode)

        modulators = []
        freq_mod = None
        fm_mod = None
        if has_freq_mod:
            modulators.append(freq_modulator)
            freq_mod = freq_mod_func
        if has_fm_mod:
            modulators.append(fm_modulator)
            fm_mod = fm_mod_func

        logger.debug(
            "VCO: ModulatedOscillator freq_mod=%s fm_mod=%s",
            has_freq_mod,
            has_fm_mod,
        )
        return ModulatedOscillator(
            osc,
            *modulators,
            amp_mod=None,
            freq_mod=freq_mod,
            fm_mod=fm_mod,
        )

    # ------------------------------------------------------------------ runtime

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Render VCO output for the current engine cycle."""
        wave_type = str_parameter(parameters, "waveform", self.wave_combo.currentText)
        mode = self._normalize_mode_for_waveform(
            wave_type,
            str_parameter(parameters, "mode", self.mode_combo.currentText),
        )
        fm_mode = str_parameter(parameters, "fm_mode", self.fm_mode_param.get_value)
        frequency = float_parameter(parameters, "frequency", self.freq_knob.get_value)
        phase = (
            float_parameter(parameters, "phase", lambda: 0.0)
            if "phase" in parameters
            else None
        )
        fm_amount = float_parameter(
            parameters, "fm_amount", self.fm_amount_knob.get_value
        )
        pulsewidth = float_parameter(
            parameters, "pulsewidth", self.pulsewidth_knob.get_value
        )

        oscillator_shape = (wave_type, mode)
        if self.component is None or oscillator_shape != self._runtime_oscillator_shape:
            self.component = create_vco_oscillator(
                wave_type,
                mode,
                frequency,
                pulsewidth=pulsewidth,
                phase=phase or 0.0,
            )
            self.out_port.component = self.component
            self._runtime_oscillator_shape = oscillator_shape
            self._last_runtime_frequency = frequency
            self._last_pitch_cv = None
        elif isinstance(self.component, SquareOscillator):
            self.component.pulsewidth = pulsewidth

        fm_signal = None
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
            samples = self._render_frequency_signal(
                frequency,
                None,
                fm_signal=fm_signal,
                fm_amount=fm_amount,
                fm_mode=fm_mode,
            )
        else:
            self._last_pitch_cv = None
            samples, rendered_frequency = render_with_frequency_ramp(
                cast(RuntimeOscillator, self.component),
                self._last_runtime_frequency,
                frequency,
                num_samples,
            )
            self._last_runtime_frequency = rendered_frequency

        self.out_port.write(samples)

        # Expose last rendered frequency for tests / UI inspection.
        # Rare rapid FM across buffer boundaries can cause tiny phase steps;
        # see tests for boundary tolerances.
        if hasattr(self.component, "frequency"):
            self.component.frequency = self._last_runtime_frequency
        if (
            phase is not None
            and not self.freq_input.is_connected
            and fm_signal is None
            and hasattr(self.component, "phase")
        ):
            self.component.phase = phase

    def _render_frequency_signal(
        self,
        base_frequency: float | np.ndarray,
        pitch_cv_signal: np.ndarray | None,
        *,
        fm_signal: np.ndarray | None = None,
        fm_amount: float = 0.0,
        fm_mode: str = VCO_FM_MODE_V_OCT,
    ) -> np.ndarray:
        """Render pitch-CV / FM using the bulk frequency API."""
        if pitch_cv_signal is not None:
            pitch_cv_signal, self._last_pitch_cv = smooth_control_signal(
                pitch_cv_signal,
                self._last_pitch_cv,
                float(getattr(self.component, "sample_rate", audio_config.sample_rate)),
                VCO_PITCH_CV_SMOOTHING_MS,
            )
        else:
            length = len(fm_signal) if fm_signal is not None else 512
            pitch_cv_signal = np.zeros(length, dtype=np.float32)

        frequencies = apply_v_oct_offset(base_frequency, pitch_cv_signal)
        if fm_signal is not None:
            frequencies = apply_vcv_fm_offset(
                frequencies, fm_signal, fm_amount, fm_mode
            )
        frequency_buffer = _as_frequency_buffer(frequencies)

        samples = self.component.process_frequency_buffer(frequency_buffer)

        if len(frequency_buffer) > 0:
            self._last_runtime_frequency = float(frequency_buffer[-1])

        return samples
