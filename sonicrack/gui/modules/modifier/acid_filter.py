"""Acid-style resonant filter module."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.filters.acid_303 import AcidResonantFilter

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.modifier._filter_base import (
    FilterModuleBase,
    bind_parameter_knob,
    read_optional_port,
)
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class AcidFilterModule(FilterModuleBase):
    """303-oriented resonant low-pass filter with env and accent CV."""

    runtime_kind = "acid_filter"

    metadata = ModuleMetadata(
        title="Acid Filter",
        category=ModuleCategory.MODIFIER,
        description="Acid-style resonant low-pass filter with drive",
    )

    def __init__(self) -> None:
        super().__init__(width=260, height=325, color=QColor(85, 165, 120))

        self._setup_filter_ports(cutoff_cv=True, env_cv=True, accent_cv=True)
        self.component = self.create_engine_component()

        layout = self._begin_controls(spacing=6)

        tone_row = QHBoxLayout()
        tone_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cutoff_knob = Knob(
            label="Cutoff",
            description="Sets the cutoff frequency of the filter",
            min_value=20.0,
            max_value=12000.0,
            default_value=700.0,
            logarithmic=True,
        )
        bind_parameter_knob(self, self.cutoff_knob, "cutoff")
        tone_row.addWidget(self.cutoff_knob)

        self.resonance_knob = Knob(
            label="Resonance",
            description="Adjusts the resonance of the filter",
            min_value=0.0,
            max_value=18.0,
            default_value=8.0,
            logarithmic=True,
        )
        bind_parameter_knob(self, self.resonance_knob, "resonance")
        tone_row.addWidget(self.resonance_knob)
        layout.addLayout(tone_row)

        mod_row = QHBoxLayout()
        mod_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.env_mod_knob = Knob(
            label="Env.Mod",
            description="Controls the envelope modulation depth",
            min_value=0.0,
            max_value=6.0,
            default_value=2.5,
        )
        bind_parameter_knob(self, self.env_mod_knob, "env_amount")
        mod_row.addWidget(self.env_mod_knob)

        self.accent_knob = Knob(
            label="Accent",
            description="Controls the accent amount",
            min_value=0.0,
            max_value=4.0,
            default_value=1.0,
        )
        bind_parameter_knob(self, self.accent_knob, "accent_amount")
        mod_row.addWidget(self.accent_knob)
        layout.addLayout(mod_row)

        gain_row = QHBoxLayout()
        gain_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drive_knob = Knob(
            label="Drive",
            description="Adjusts the drive of the filter",
            min_value=0.0,
            max_value=24.0,
            default_value=6.0,
        )
        bind_parameter_knob(self, self.drive_knob, "drive_db")
        gain_row.addWidget(self.drive_knob)

        self.output_knob = Knob(
            label="Output",
            description="Adjusts the output level of the filter",
            min_value=-24.0,
            max_value=12.0,
            default_value=-6.0,
        )
        bind_parameter_knob(self, self.output_knob, "output_gain_db")
        gain_row.addWidget(self.output_knob)
        layout.addLayout(gain_row)

        self._finish_controls(layout)

        self.register_parameter("cutoff", self.cutoff_knob)
        self.register_parameter("resonance", self.resonance_knob)
        self.register_parameter("env_amount", self.env_mod_knob)
        self.register_parameter("accent_amount", self.accent_knob)
        self.register_parameter("drive_db", self.drive_knob)
        self.register_parameter("output_gain_db", self.output_knob)

        self._install_sample_rate_listener()

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> AcidResonantFilter:
        del input_components, modulation_components
        return AcidResonantFilter(sample_rate=audio_config.sample_rate)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if not self.in_port.is_connected:
            self._write_silence(num_samples)
            return

        self.component.cutoff = float_parameter(
            parameters, "cutoff", self.cutoff_knob.get_value
        )
        self.component.resonance = float_parameter(
            parameters, "resonance", self.resonance_knob.get_value
        )
        self.component.env_amount = float_parameter(
            parameters, "env_amount", self.env_mod_knob.get_value
        )
        self.component.accent_amount = float_parameter(
            parameters, "accent_amount", self.accent_knob.get_value
        )
        self.component.drive_db = float_parameter(
            parameters, "drive_db", self.drive_knob.get_value
        )
        self.component.output_gain_db = float_parameter(
            parameters, "output_gain_db", self.output_knob.get_value
        )

        self.out_port.write(
            self.component.process_modulated(
                read_samples(self.in_port, num_samples),
                cutoff_cv=read_optional_port(self.cutoff_cv_port, num_samples),
                env_cv=read_optional_port(self.env_cv_port, num_samples),
                accent_cv=read_optional_port(self.accent_cv_port, num_samples),
            )
        )
