"""Audio-to-CV envelope follower."""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import as_mono, float_parameter, read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters


def _time_constant(seconds: float, sample_rate: float) -> float:
    """One-pole coefficient for a time constant in seconds."""
    samples = max(1.0, float(seconds) * float(sample_rate))
    return float(np.exp(-1.0 / samples))


class _EnvelopeFollower:
    """Peak envelope follower with independent attack and release times."""

    def __init__(self) -> None:
        self._env = 0.0

    def reset(self) -> None:
        self._env = 0.0

    def process(
        self,
        samples: np.ndarray,
        *,
        attack_s: float,
        release_s: float,
        sample_rate: float,
        gain: float,
    ) -> np.ndarray:
        x = np.abs(as_mono(samples)) * float(gain)
        n = int(x.shape[0])
        if n == 0:
            return x
        attack = _time_constant(max(0.0005, attack_s), sample_rate)
        release = _time_constant(max(0.001, release_s), sample_rate)
        env = self._env
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            target = float(x[i])
            coef = attack if target > env else release
            env = coef * env + (1.0 - coef) * target
            out[i] = env
        self._env = env
        return out


@register_module()
class EnvelopeFollowerModule(ModuleWidget):
    """Follow audio amplitude and emit unipolar CV (auto-wah, sidechain)."""

    runtime_kind = "envelope_follower"

    metadata = ModuleMetadata(
        title="Env Follower",
        category=ModuleCategory.MODIFIER,
        description="Audio envelope follower (In → CV, with audio thru)",
    )

    def __init__(self) -> None:
        super().__init__(width=200, height=210, color=QColor(110, 170, 90))
        self.in_port = self.add_input("In", signal=PortSignal.AUDIO)
        self.env_port = self.add_output("Env", signal=PortSignal.CONTROL_CV)
        self.thru_port = self.add_output("Thru", signal=PortSignal.AUDIO)
        self._follower = _EnvelopeFollower()

        layout = self._begin_controls()
        row = QHBoxLayout()
        self.attack_knob = Knob(
            label="Attack",
            description="Rise time in milliseconds",
            min_value=0.5,
            max_value=200.0,
            default_value=8.0,
            logarithmic=True,
        )
        self.bind_parameter_knob(self.attack_knob, "attack", register=True)
        row.addWidget(self.attack_knob)

        self.release_knob = Knob(
            label="Release",
            description="Fall time in milliseconds",
            min_value=5.0,
            max_value=2000.0,
            default_value=120.0,
            logarithmic=True,
        )
        self.bind_parameter_knob(self.release_knob, "release", register=True)
        row.addWidget(self.release_knob)
        layout.addLayout(row)

        self.gain_knob = Knob(
            label="Sense",
            description="Input sensitivity",
            min_value=0.25,
            max_value=8.0,
            default_value=1.5,
            logarithmic=True,
        )
        self.bind_parameter_knob(self.gain_knob, "gain", register=True)
        layout.addWidget(self.gain_knob)
        self._finish_controls(layout)

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if not self.in_port.is_connected:
            self.env_port.write(silence(num_samples))
            self.thru_port.write(silence(num_samples))
            return
        audio = read_samples(self.in_port, num_samples)
        attack_s = (
            float_parameter(parameters, "attack", self.attack_knob.get_value) / 1000.0
        )
        release_s = (
            float_parameter(parameters, "release", self.release_knob.get_value) / 1000.0
        )
        gain = float_parameter(parameters, "gain", self.gain_knob.get_value)
        env = self._follower.process(
            audio,
            attack_s=attack_s,
            release_s=release_s,
            sample_rate=audio_config.sample_rate,
            gain=gain,
        )
        self.env_port.write(np.clip(env, 0.0, 1.0).astype(np.float32, copy=False))
        self.thru_port.write(audio)
