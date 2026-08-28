"""Multi-mode state-variable filter module."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout
from soniclab.dsp.filters import StateVariableFilter

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.modifier._filter_base import (
    RESONANT_TYPE_ITEMS,
    FilterModuleBase,
    create_filter_type_combo,
    normalize_filter_type,
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
        cutoff_layout = QVBoxLayout()
        cutoff_label = QLabel("Cutoff")
        cutoff_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cutoff_knob = Knob(
            label="Hz",
            min_value=20,
            max_value=18000,
            default_value=1200,
            logarithmic=True,
        )
        self.cutoff_value_label = QLabel("1200 Hz")
        self.cutoff_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bind_parameter_knob(
            self.cutoff_knob,
            "cutoff",
            value_label=self.cutoff_value_label,
            format_value=lambda v: f"{int(v)} Hz",
        )
        cutoff_layout.addWidget(cutoff_label)
        cutoff_layout.addWidget(self.cutoff_knob)
        cutoff_layout.addWidget(self.cutoff_value_label)
        cutoff_row.addLayout(cutoff_layout)

        resonance_layout = QVBoxLayout()
        resonance_label = QLabel("Resonance")
        resonance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.resonance_knob = Knob(
            label="Q",
            min_value=0.1,
            max_value=20.0,
            default_value=0.9,
            logarithmic=True,
        )
        self.resonance_value_label = QLabel("0.90")
        self.resonance_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bind_parameter_knob(
            self.resonance_knob,
            "resonance",
            value_label=self.resonance_value_label,
            format_value=lambda v: f"{v:.2f}",
        )
        resonance_layout.addWidget(resonance_label)
        resonance_layout.addWidget(self.resonance_knob)
        resonance_layout.addWidget(self.resonance_value_label)
        cutoff_row.addLayout(resonance_layout)
        layout.addLayout(cutoff_row)

        cv_row = QHBoxLayout()
        cv_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cv_layout = QVBoxLayout()
        cv_label = QLabel("CV Depth")
        cv_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cv_depth_knob = Knob(
            label="Oct", min_value=-5.0, max_value=5.0, default_value=1.0
        )
        self.cv_depth_value_label = QLabel("1.0 oct")
        self.cv_depth_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bind_parameter_knob(
            self.cv_depth_knob,
            "cv_depth_octaves",
            value_label=self.cv_depth_value_label,
            format_value=lambda v: f"{v:.1f} oct",
        )
        cv_layout.addWidget(cv_label)
        cv_layout.addWidget(self.cv_depth_knob)
        cv_layout.addWidget(self.cv_depth_value_label)
        cv_row.addLayout(cv_layout)
        layout.addLayout(cv_row)
        self._finish_controls(layout)

        self.register_parameter("cutoff", self.cutoff_knob)
        self.register_parameter("resonance", self.resonance_knob)
        self.register_parameter("cv_depth_octaves", self.cv_depth_knob)
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
            mode=normalize_filter_type(
                self.type_combo.currentText(), allow_notch=True
            ),
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

        if self.cutoff_cv_port.is_connected:
            cv_signal = read_samples(self.cutoff_cv_port, num_samples)
            cv_depth = float_parameter(
                parameters, "cv_depth_octaves", self.cv_depth_knob.get_value
            )
            base_cutoff = (
                cutoff_values
                if cutoff_values is not None
                else np.full(num_samples, cutoff, dtype=np.float32)
            )
            cutoff_values = np.clip(
                base_cutoff * np.power(2.0, cv_signal * cv_depth),
                20.0,
                audio_config.sample_rate * 0.45,
            )

        if cutoff_values is None and resonance_values is None:
            self.out_port.write(self.component.process(input_signal))
        else:
            self.out_port.write(
                self.component.process_modulated(
                    input_signal,
                    cutoff_values=cutoff_values,
                    resonance_values=resonance_values,
                )
            )
        self._last_cutoff = cutoff
        self._last_resonance = resonance
