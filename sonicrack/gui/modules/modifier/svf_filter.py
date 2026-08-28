"""Multi-mode state-variable filter module."""

from __future__ import annotations

from typing import Any, Literal

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

# pylint: disable-next=no-name-in-module
from soniclab.dsp.filters import StateVariableFilter  # type: ignore[attr-defined]

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.modifier._filter_base import (
    RESONANT_TYPE_ITEMS,
    FilterModuleBase,
    apply_octave_cutoff_cv,
    create_filter_type_combo,
    labeled_knob_column,
    normalize_filter_type,
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


@register_module()
class SVFFilterModule(FilterModuleBase):
    """Musical multi-mode SVF (low / high / band / notch)."""

    runtime_kind = "svf_filter"

    metadata = ModuleMetadata(
        title="SVF Filter",
        category=ModuleCategory.FILTER,
        description="State-variable filter with LP/HP/BP/Notch and cutoff CV",
    )

    def __init__(self) -> None:
        super().__init__(width=260, height=310, color=QColor(70, 150, 170))
        self._setup_filter_ports(cutoff_cv=True)
        layout = self._begin_controls(spacing=6)

        type_layout, self.type_combo = create_filter_type_combo(
            RESONANT_TYPE_ITEMS,
            centered=True,
        )
        layout.addLayout(type_layout)

        cutoff_row = QHBoxLayout()
        cutoff_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cutoff_knob = Knob(
            label="Hz",
            min_value=20,
            max_value=18000,
            default_value=1200,
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
            label="Q",
            min_value=0.1,
            max_value=20.0,
            default_value=0.9,
            logarithmic=True,
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

        cv_row = QHBoxLayout()
        cv_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        cv_row.addLayout(cv_layout)
        layout.addLayout(cv_row)
        self._finish_controls(layout)

        self._register_filter_type_parameter(self.type_combo)

        self.component = self.create_engine_component()
        self._last_cutoff = self.cutoff_knob.get_value()
        self._last_resonance = self.resonance_knob.get_value()
        self._install_sample_rate_listener()

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> StateVariableFilter:
        del input_components, modulation_components
        return StateVariableFilter(
            cutoff=self.cutoff_knob.get_value(),
            resonance=self.resonance_knob.get_value(),
            mode=normalize_filter_type(self.type_combo.currentText(), allow_notch=True),
            sample_rate=audio_config.sample_rate,
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if self._require_input_or_silence(num_samples):
            return

        mode: Literal["low", "high", "band", "notch"] = normalize_filter_type(
            str_parameter(parameters, "filter_type", self.type_combo.currentText),
            allow_notch=True,
        )
        cutoff = float_parameter(parameters, "cutoff", self.cutoff_knob.get_value)
        resonance = float_parameter(
            parameters, "resonance", self.resonance_knob.get_value
        )

        if self.component is None:
            self.component = self.create_engine_component()

        self.component.cutoff = cutoff
        self.component.resonance = resonance
        self.component.mode = mode

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
