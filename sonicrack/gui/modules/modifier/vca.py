"""Voltage-Controlled Amplifier (VCA) module."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab import Volume
from soniclab.dsp.modifiers import ModulatedVolume

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import (
    apply_cv_influence,
    float_parameter,
    ramp_if_changed,
    read_samples,
)
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)

# Dezipper time for the Volume compiler path; runtime uses buffer-linear ramps.
VCA_GAIN_SLEW_TIME_MS = 10.0


@register_module()
class VCAModule(ModuleWidget):
    """Voltage-Controlled Amplifier (VCA) with amplitude modulation.

    Controls the amplitude of an audio input. Gain sets final output level.
    When CV is connected, CV Attn blends between Gain-only and full CV control:

        amplitude = ((1 - attn) + cv * attn) * gain

    Inputs:
        - In: Audio input
        - CV In: Unipolar control voltage [0, 1] (e.g. envelope, velocity)

    Outputs:
        - Out: Amplitude-controlled audio
    """

    runtime_kind = "vca"

    metadata = ModuleMetadata(
        title="VCA",
        category=ModuleCategory.MODIFIER,
        description="Voltage-Controlled Amplifier with CV modulation",
        version="1.0.0",
        author="SonicRack",
    )

    def __init__(self) -> None:
        super().__init__(
            width=220,
            height=180,
            color=QColor(100, 140, 180),
        )

        self.in_port = self.add_input("In", signal=PortSignal.AUDIO)
        self.cv_port = self.add_input("CV In", signal=PortSignal.CONTROL_CV)
        self.out_port = self.add_output("Out", signal=PortSignal.AUDIO)

        layout = self._begin_controls()

        knobs_layout = QHBoxLayout()

        self.cv_attn_knob = Knob(
            label="CV Attn",
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
            logarithmic=False,
        )
        self.cv_attn_knob.setToolTip(
            "CV influence: 0 = Gain only, 1 = full CV control of amplitude"
        )
        self.bind_parameter_knob(self.cv_attn_knob, "cv_attenuation")
        knobs_layout.addWidget(self.cv_attn_knob)

        self.gain_knob = Knob(
            label="Gain",
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
            logarithmic=False,
        )
        self.gain_knob.setToolTip("Final VCA output gain (0.0 to 1.0)")
        self.bind_parameter_knob(self.gain_knob, "amplitude")
        knobs_layout.addWidget(self.gain_knob)
        layout.addLayout(knobs_layout)

        self._finish_controls(layout)

        self.register_parameter("amplitude", self.gain_knob)
        self.register_parameter("cv_attenuation", self.cv_attn_knob)

        # None until the first runtime buffer so we do not ramp from a stale default.
        self._last_gain: float | None = None
        self.component = self.create_engine_component()
        self.update_knob_state()

    def get_required_inputs(self) -> list[str]:
        """VCA requires audio input on the In port."""
        return ["In"]

    def get_modulation_inputs(self) -> list[str]:
        """VCA accepts modulation on the CV In port."""
        return ["CV In"]

    def get_cv_range(self, port_name: str = "CV In") -> tuple[float, float]:
        """CV In expects unipolar [0, 1] amplitude control."""
        _ = port_name
        return 0.0, 1.0

    def _cv_is_connected(self) -> bool:
        """True when CV In has a live model or GUI cable connection."""
        if bool(getattr(self.cv_port, "is_connected", False)):
            return True
        for port in self.input_ports:
            if port.port_name != "CV In":
                continue
            cables = getattr(port, "cables", None)
            return bool(cables) and len(cables) > 0
        return False

    def update_knob_state(self) -> None:
        """Refresh tooltips and CV Attn enablement from connection state."""
        has_cv = self._cv_is_connected()

        self.gain_knob.setEnabled(True)
        self.gain_knob.setStyleSheet("")
        self.cv_attn_knob.setEnabled(has_cv)
        self.cv_attn_knob.setStyleSheet("" if has_cv else "opacity: 0.5;")

        if has_cv:
            self.gain_knob.setToolTip("Final VCA output gain after CV modulation")
            self.cv_attn_knob.setToolTip(
                "CV influence amount: 0 = Gain only, 1 = CV only"
            )
        else:
            self.gain_knob.setToolTip("Final VCA output gain (0.0 to 1.0)")
            self.cv_attn_knob.setToolTip("CV influence when CV input is connected")

        logger.debug("VCA update_knob_state: has_cv=%s", has_cv)

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create Volume or ModulatedVolume for the patch-compiler path.

        Runtime audio uses ``process_runtime``; this remains for tests and
        legacy component mapping.
        """
        _ = input_components
        amplitude = float(self.gain_knob.get_value())
        cv_attenuation = float(self.cv_attn_knob.get_value())
        sample_rate = float(audio_config.sample_rate)

        cv_modulator = None
        if isinstance(modulation_components, dict):
            cv_modulator = modulation_components.get("CV In")

        if cv_modulator is not None and cv_attenuation > 0.0:
            logger.debug(
                "VCA: ModulatedVolume amp=%.3f attn=%.3f",
                amplitude,
                cv_attenuation,
            )
            influenced_cv = _CVInfluenceAmplitude(
                cv_modulator,
                output_gain=amplitude,
                influence=cv_attenuation,
            )
            return ModulatedVolume(
                influenced_cv,
                modulation_target="amplitude",
            )

        logger.debug("VCA: Volume amp=%.3f (no CV influence)", amplitude)
        return Volume(
            amplitude=amplitude,
            sample_rate=sample_rate,
            smoothing_time_ms=VCA_GAIN_SLEW_TIME_MS,
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Apply manual or CV-controlled amplitude for one render cycle."""
        if self._require_input_or_silence(num_samples):
            return

        input_signal = read_samples(self.in_port, num_samples)
        manual_gain = float_parameter(
            parameters, "amplitude", self.gain_knob.get_value
        )
        manual_gain = float(np.clip(manual_gain, 0.0, 1.0))

        # Dezipper Gain knob changes so abrupt UI moves do not click.
        # Skip ramping on the first buffer (or after a cold start).
        if self._last_gain is None:
            gain_for_process: float | np.ndarray = manual_gain
        else:
            gain_curve = ramp_if_changed(self._last_gain, manual_gain, num_samples)
            gain_for_process = manual_gain if gain_curve is None else gain_curve
        self._last_gain = manual_gain

        if self.cv_port.is_connected:
            cv_attenuation = float_parameter(
                parameters, "cv_attenuation", self.cv_attn_knob.get_value
            )
            cv_amplitude = read_samples(self.cv_port, num_samples)
            amplitude = apply_cv_influence(
                cv_amplitude,
                cv_attenuation,
                output_gain=gain_for_process,
            )
        else:
            amplitude = gain_for_process

        self.out_port.write(np.asarray(input_signal, dtype=np.float32) * amplitude)


class _CVInfluenceAmplitude:
    """Iterator/generator adapter that blends manual gain with incoming CV."""

    def __init__(self, source: Any, *, output_gain: float, influence: float):
        self._source = source
        self._iterator = iter(source)
        self._output_gain = float(output_gain)
        self._influence = float(influence)

    def __iter__(self):
        self._iterator = iter(self._source)
        return self

    def __next__(self):
        return float(
            apply_cv_influence(
                next(self._iterator),
                self._influence,
                output_gain=self._output_gain,
            )
        )

    def get_samples(self, num_samples: int, **kwargs) -> np.ndarray:
        if hasattr(self._source, "get_samples_vectorized") and not kwargs:
            cv_values = self._source.get_samples_vectorized(num_samples)
        elif hasattr(self._source, "get_samples"):
            cv_values = self._source.get_samples(num_samples, **kwargs)
        else:
            cv_values = np.array(
                [next(self._iterator) for _ in range(num_samples)],
                dtype=np.float32,
            )
        return np.asarray(
            apply_cv_influence(
                cv_values,
                self._influence,
                output_gain=self._output_gain,
            ),
            dtype=np.float32,
        )

    def trigger_release(self):
        if hasattr(self._source, "trigger_release"):
            self._source.trigger_release()

    @property
    def ended(self):
        return bool(getattr(self._source, "ended", False))
