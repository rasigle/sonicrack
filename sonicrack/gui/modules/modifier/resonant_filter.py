"""Resonant synth filter module for the modular synthesizer GUI."""

from __future__ import annotations

from typing import Any, Literal

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.filters.butterworth import BiquadResonantFilter

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.modifier._filter_base import (
    RESONANT_TYPE_ITEMS,
    FilterModuleBase,
    apply_octave_cutoff_cv,
    create_filter_type_combo,
    labeled_knob_column,
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
class ResonantFilterModule(FilterModuleBase):
    """Synth-style resonant biquad filter with cutoff CV and drive."""

    runtime_kind = "resonant_filter"

    metadata = ModuleMetadata(
        title="Resonant Filter",
        category=ModuleCategory.FILTER,
        description="Resonant synth filter with cutoff CV, drive, and output gain",
    )

    def __init__(self) -> None:
        super().__init__(
            width=300,
            height=345,
            color=QColor(80, 170, 150),
        )

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
            max_value=20000,
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
            max_value=12.0,
            default_value=0.707,
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

        character_row = QHBoxLayout()
        character_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cv_depth_knob = Knob(
            label="Oct", min_value=-5.0, max_value=5.0, default_value=0.0
        )
        cv_layout, self.cv_depth_value_label = labeled_knob_column(
            self,
            "CV Depth",
            self.cv_depth_knob,
            "cv_depth_octaves",
            lambda v: f"{v:.1f} oct",
        )
        character_row.addLayout(cv_layout)

        self.drive_knob = Knob(
            label="dB", min_value=0.0, max_value=24.0, default_value=0.0
        )
        drive_layout, self.drive_value_label = labeled_knob_column(
            self,
            "Drive",
            self.drive_knob,
            "drive_db",
            lambda v: f"{v:.0f} dB",
        )
        self.output_gain_knob = Knob(
            label="dB", min_value=-24.0, max_value=12.0, default_value=-6.0
        )
        output_layout, self.output_gain_value_label = labeled_knob_column(
            self,
            "Output",
            self.output_gain_knob,
            "output_gain_db",
            lambda v: f"{v:.0f} dB",
        )
        character_row.addLayout(drive_layout)
        character_row.addLayout(output_layout)
        layout.addLayout(character_row)

        self._finish_controls(layout)

        self._register_filter_type_parameter(self.type_combo)

        self.component = self.create_engine_component()
        self._runtime_filter_params: (
            tuple[float, float, Literal["low", "high", "band", "notch"], float, float]
            | None
        ) = None
        self._last_cutoff = self.cutoff_knob.get_value()
        self._last_resonance = self.resonance_knob.get_value()
        self._last_drive_db = self.drive_knob.get_value()
        self._last_output_gain_db = self.output_gain_knob.get_value()
        self._install_sample_rate_listener()

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> BiquadResonantFilter:
        del input_components, modulation_components
        return BiquadResonantFilter(
            cutoff=self.cutoff_knob.get_value(),
            resonance=self.resonance_knob.get_value(),
            filter_type=normalize_filter_type(
                self.type_combo.currentText(), allow_notch=True
            ),
            drive_db=self.drive_knob.get_value(),
            output_gain_db=self.output_gain_knob.get_value(),
            sample_rate=audio_config.sample_rate,
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Filter the connected input for one render cycle."""
        if self._require_input_or_silence(num_samples):
            return

        filter_type = normalize_filter_type(
            str_parameter(parameters, "filter_type", self.type_combo.currentText),
            allow_notch=True,
        )
        cutoff = float_parameter(parameters, "cutoff", self.cutoff_knob.get_value)
        resonance = float_parameter(
            parameters, "resonance", self.resonance_knob.get_value
        )
        drive_db = float_parameter(parameters, "drive_db", self.drive_knob.get_value)
        output_gain_db = float_parameter(
            parameters, "output_gain_db", self.output_gain_knob.get_value
        )
        filter_params = (cutoff, resonance, filter_type, drive_db, output_gain_db)

        if self.component is None:
            self.component = BiquadResonantFilter(
                cutoff=cutoff,
                resonance=resonance,
                filter_type=filter_type,
                drive_db=drive_db,
                output_gain_db=output_gain_db,
                sample_rate=audio_config.sample_rate,
            )
            self._last_cutoff = cutoff
            self._last_resonance = resonance
            self._last_drive_db = drive_db
            self._last_output_gain_db = output_gain_db
        elif filter_params != self._runtime_filter_params:
            self.component.configure(
                cutoff=cutoff,
                resonance=resonance,
                filter_type=filter_type,
                drive_db=drive_db,
                output_gain_db=output_gain_db,
            )

        self._runtime_filter_params = filter_params

        input_signal = read_samples(self.in_port, num_samples)
        cutoff_values = ramp_if_changed(self._last_cutoff, cutoff, num_samples)
        resonance_values = ramp_if_changed(self._last_resonance, resonance, num_samples)
        drive_db_values = ramp_if_changed(self._last_drive_db, drive_db, num_samples)
        output_gain_db_values = ramp_if_changed(
            self._last_output_gain_db, output_gain_db, num_samples
        )

        cutoff_values = apply_octave_cutoff_cv(
            cutoff,
            cutoff_values,
            self.cutoff_cv_port,
            float_parameter(
                parameters, "cv_depth_octaves", self.cv_depth_knob.get_value
            ),
            num_samples,
        )

        self.out_port.write(
            self.component.process_modulated(
                input_signal,
                cutoff_values,
                resonance_values,
                drive_db_values,
                output_gain_db_values,
            )
        )
        self._last_cutoff = cutoff
        self._last_resonance = resonance
        self._last_drive_db = drive_db
        self._last_output_gain_db = output_gain_db
