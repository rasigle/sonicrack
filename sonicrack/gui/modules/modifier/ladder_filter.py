"""Moog-style ladder filter module."""

from __future__ import annotations

from typing import Any, Literal

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

# pylint: disable-next=no-name-in-module
from soniclab.dsp.filters import LadderFilter  # type: ignore[attr-defined]

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.modifier._filter_base import (
    FilterModuleBase,
    apply_octave_cutoff_cv,
    labeled_knob_column,
    write_modulated_filter_output,
)
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import (
    float_parameter,
    ramp_if_changed,
    read_samples,
    str_parameter,
)
from sonicrack.runtime.specs import RuntimeParameters

_POLE_LABELS: dict[str, int] = {
    "6 dB": 1,
    "12 dB": 2,
    "18 dB": 3,
    "24 dB": 4,
}
_POLE_ITEMS = tuple(_POLE_LABELS.keys())


def _poles_from_label(text: str) -> Literal[1, 2, 3, 4]:
    poles = _POLE_LABELS.get(text, 4)
    if poles == 1:
        return 1
    if poles == 2:
        return 2
    if poles == 3:
        return 3
    return 4


@register_module()
class LadderFilterModule(FilterModuleBase):
    """Four-pole resonant ladder low-pass (Moog-style character)."""

    runtime_kind = "ladder_filter"

    metadata = ModuleMetadata(
        title="Ladder Filter",
        category=ModuleCategory.FILTER,
        description="Moog-style 24 dB ladder low-pass with cutoff CV",
    )

    def __init__(self) -> None:
        super().__init__(width=260, height=280, color=QColor(190, 120, 60))
        self._setup_filter_ports(cutoff_cv=True)
        layout = self._begin_controls(spacing=6)

        cutoff_row = QHBoxLayout()
        cutoff_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cutoff_knob = Knob(
            label="Hz",
            min_value=20,
            max_value=18000,
            default_value=800,
            logarithmic=True,
        )
        cutoff_layout, self.cutoff_value_label = labeled_knob_column(
            self,
            "Cutoff",
            self.cutoff_knob,
            "cutoff",
            lambda v: f"{int(v)} Hz",
        )
        cutoff_row.addLayout(cutoff_layout)

        self.resonance_knob = Knob(
            label="Res",
            min_value=0.0,
            max_value=1.0,
            default_value=0.45,
        )
        resonance_layout, self.resonance_value_label = labeled_knob_column(
            self,
            "Resonance",
            self.resonance_knob,
            "resonance",
            lambda v: f"{v:.2f}",
        )
        cutoff_row.addLayout(resonance_layout)
        layout.addLayout(cutoff_row)

        extra_row = QHBoxLayout()
        extra_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cv_depth_knob = Knob(
            label="Oct", min_value=-5.0, max_value=5.0, default_value=1.0
        )
        cv_layout, self.cv_depth_value_label = labeled_knob_column(
            self,
            "CV Depth",
            self.cv_depth_knob,
            "cv_depth_octaves",
            lambda v: f"{v:.1f} oct",
        )
        extra_row.addLayout(cv_layout)

        self.drive_knob = Knob(
            label="x", min_value=0.1, max_value=8.0, default_value=1.2, logarithmic=True
        )
        drive_layout, self.drive_value_label = labeled_knob_column(
            self,
            "Drive",
            self.drive_knob,
            "drive",
            lambda v: f"{v:.1f}",
        )
        extra_row.addLayout(drive_layout)
        layout.addLayout(extra_row)
        self._finish_controls(layout)

        self.poles_choice = self.register_menu_choice(
            "poles",
            "Slope",
            _POLE_ITEMS,
            "24 dB",
            tooltip="Ladder output tap (6–24 dB/oct)",
        )

        self.component = self.create_engine_component()
        self._last_cutoff = self.cutoff_knob.get_value()
        self._last_resonance = self.resonance_knob.get_value()
        self._install_sample_rate_listener()

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> LadderFilter:
        del input_components, modulation_components
        return LadderFilter(
            cutoff=self.cutoff_knob.get_value(),
            resonance=self.resonance_knob.get_value(),
            poles=_poles_from_label(self.poles_choice.get_value()),
            drive=self.drive_knob.get_value(),
            sample_rate=audio_config.sample_rate,
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if self._require_input_or_silence(num_samples):
            return

        cutoff = float_parameter(parameters, "cutoff", self.cutoff_knob.get_value)
        resonance = float_parameter(
            parameters, "resonance", self.resonance_knob.get_value
        )
        drive = float_parameter(parameters, "drive", self.drive_knob.get_value)
        poles = _poles_from_label(
            str_parameter(parameters, "poles", self.poles_choice.get_value)
        )

        if self.component is None:
            self.component = self.create_engine_component()

        self.component.cutoff = cutoff
        self.component.resonance = resonance
        self.component.drive = drive
        self.component.poles = poles

        input_signal = read_samples(self.in_port, num_samples)
        cutoff_values = ramp_if_changed(self._last_cutoff, cutoff, num_samples)
        resonance_values = ramp_if_changed(self._last_resonance, resonance, num_samples)
        cutoff_values = apply_octave_cutoff_cv(
            cutoff,
            cutoff_values,
            self.cutoff_cv_port,
            float_parameter(
                parameters, "cv_depth_octaves", self.cv_depth_knob.get_value
            ),
            num_samples,
        )
        write_modulated_filter_output(
            self.out_port,
            self.component,
            input_signal,
            cutoff_values,
            resonance_values,
        )
        self._last_cutoff = cutoff
        self._last_resonance = resonance
