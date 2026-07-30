"""Acid-style resonant filter module."""

from __future__ import annotations

from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.filters.acid_303 import AcidResonantFilter

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.modifier._filter_base import FilterModuleBase
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import (
    float_parameter,
    ramp_if_changed,
    read_optional_port,
    read_samples,
)
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class AcidFilterModule(FilterModuleBase):
    """303-oriented resonant low-pass filter with env and accent CV."""

    runtime_kind = "acid_filter"

    metadata = ModuleMetadata(
        title="Acid Filter",
        category=ModuleCategory.FILTER,
        description="Acid-style resonant low-pass filter with drive",
    )

    def __init__(self) -> None:
        super().__init__(width=260, height=325, color=QColor(85, 165, 120))

        self._setup_filter_ports(cutoff_cv=True, env_cv=True, accent_cv=True)
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
        self.bind_parameter_knob(self.cutoff_knob, "cutoff")
        tone_row.addWidget(self.cutoff_knob)

        self.resonance_knob = Knob(
            label="Resonance",
            description="Adjusts the resonance of the filter",
            min_value=0.0,
            max_value=18.0,
            default_value=8.0,
            logarithmic=True,
        )
        self.bind_parameter_knob(self.resonance_knob, "resonance")
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
        self.bind_parameter_knob(self.env_mod_knob, "env_amount")
        mod_row.addWidget(self.env_mod_knob)

        self.accent_knob = Knob(
            label="Accent",
            description="Controls the accent amount",
            min_value=0.0,
            max_value=4.0,
            default_value=1.0,
        )
        self.bind_parameter_knob(self.accent_knob, "accent_amount")
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
        self.bind_parameter_knob(self.drive_knob, "drive_db")
        gain_row.addWidget(self.drive_knob)

        self.output_knob = Knob(
            label="Output",
            description="Adjusts the output level of the filter",
            min_value=-24.0,
            max_value=12.0,
            default_value=-6.0,
        )
        self.bind_parameter_knob(self.output_knob, "output_gain_db")
        gain_row.addWidget(self.output_knob)
        layout.addLayout(gain_row)

        self._finish_controls(layout)

        self.register_parameter("cutoff", self.cutoff_knob)
        self.register_parameter("resonance", self.resonance_knob)
        self.register_parameter("env_amount", self.env_mod_knob)
        self.register_parameter("accent_amount", self.accent_knob)
        self.register_parameter("drive_db", self.drive_knob)
        self.register_parameter("output_gain_db", self.output_knob)

        self.component = self.create_engine_component()
        self._last_cutoff = self.cutoff_knob.get_value()
        self._last_resonance = self.resonance_knob.get_value()
        self._last_drive_db = self.drive_knob.get_value()
        self._last_output_gain_db = self.output_knob.get_value()
        self._install_sample_rate_listener()

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> AcidResonantFilter:
        """Create the acid filter from current knob values (preserves params)."""
        del input_components, modulation_components
        return AcidResonantFilter(
            cutoff=self.cutoff_knob.get_value(),
            resonance=self.resonance_knob.get_value(),
            env_amount=self.env_mod_knob.get_value(),
            accent_amount=self.accent_knob.get_value(),
            drive_db=self.drive_knob.get_value(),
            output_gain_db=self.output_knob.get_value(),
            sample_rate=audio_config.sample_rate,
        )

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        """Rebuild with current knobs and resync ramp history."""
        super()._on_global_sample_rate_changed(new_sample_rate)
        self._last_cutoff = self.cutoff_knob.get_value()
        self._last_resonance = self.resonance_knob.get_value()
        self._last_drive_db = self.drive_knob.get_value()
        self._last_output_gain_db = self.output_knob.get_value()

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if self._require_input_or_silence(num_samples):
            return

        cutoff = float_parameter(parameters, "cutoff", self.cutoff_knob.get_value)
        resonance = float_parameter(
            parameters, "resonance", self.resonance_knob.get_value
        )
        env_amount = float_parameter(
            parameters, "env_amount", self.env_mod_knob.get_value
        )
        accent_amount = float_parameter(
            parameters, "accent_amount", self.accent_knob.get_value
        )
        drive_db = float_parameter(parameters, "drive_db", self.drive_knob.get_value)
        output_gain_db = float_parameter(
            parameters, "output_gain_db", self.output_knob.get_value
        )

        # Keep public component attrs in sync for tests and inspection.
        self.component.cutoff = cutoff
        self.component.resonance = resonance
        self.component.env_amount = env_amount
        self.component.accent_amount = accent_amount
        self.component.drive_db = drive_db
        self.component.output_gain_db = output_gain_db

        input_signal = read_samples(self.in_port, num_samples)

        cutoff_values = ramp_if_changed(self._last_cutoff, cutoff, num_samples)
        if cutoff_values is None:
            cutoff_values = np.full(num_samples, cutoff, dtype=np.float32)

        # Match AcidResonantFilter.process_modulated octave-CV math, then apply
        # knob ramps for resonance/drive/output via the inner biquad.
        cutoff_cv = read_optional_port(self.cutoff_cv_port, num_samples)
        if cutoff_cv is not None:
            cutoff_values = cutoff_values * np.power(2.0, cutoff_cv)
        env_cv = read_optional_port(self.env_cv_port, num_samples)
        if env_cv is not None:
            cutoff_values = cutoff_values * np.power(2.0, env_cv * env_amount)
        accent_cv = read_optional_port(self.accent_cv_port, num_samples)
        if accent_cv is not None:
            cutoff_values = cutoff_values * np.power(2.0, accent_cv * accent_amount)

        cutoff_values = np.clip(
            cutoff_values, 20.0, audio_config.sample_rate * 0.45
        )

        resonance_values = ramp_if_changed(
            self._last_resonance, resonance, num_samples
        )
        drive_db_values = ramp_if_changed(self._last_drive_db, drive_db, num_samples)
        output_gain_db_values = ramp_if_changed(
            self._last_output_gain_db, output_gain_db, num_samples
        )

        inner = getattr(self.component, "_filter", None)
        if inner is not None and hasattr(inner, "process_modulated"):
            inner.configure(
                cutoff=cutoff,
                resonance=resonance,
                filter_type="low",
                drive_db=drive_db,
                output_gain_db=output_gain_db,
            )
            output = inner.process_modulated(
                input_signal,
                cutoff_values,
                resonance_values,
                drive_db_values,
                output_gain_db_values,
            )
        else:
            # Public-API fallback (no per-sample drive/resonance ramps).
            output = self.component.process_modulated(
                input_signal,
                cutoff_cv=cutoff_cv,
                env_cv=env_cv,
                accent_cv=accent_cv,
            )

        self.out_port.write(output)
        self._last_cutoff = cutoff
        self._last_resonance = resonance
        self._last_drive_db = drive_db
        self._last_output_gain_db = output_gain_db
