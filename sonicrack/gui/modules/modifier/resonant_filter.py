"""Resonant synth filter module for the modular synthesizer GUI."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout
from soniclab.dsp.filters.butterworth import BiquadResonantFilter

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import (
    float_parameter,
    read_samples,
    silence,
    str_parameter,
)
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class ResonantFilterModule(ModuleWidget):
    """Synth-style resonant biquad filter with cutoff CV and drive."""

    runtime_kind = "resonant_filter"

    metadata = ModuleMetadata(
        title="Resonant Filter",
        category=ModuleCategory.MODIFIER,
        description="Resonant synth filter with cutoff CV, drive, and output gain",
    )

    def __init__(self) -> None:
        super().__init__(
            width=300,
            height=330,
            color=QColor(80, 170, 150),
        )

        self.in_port = self.add_input("In")
        self.cutoff_cv_port = self.add_input("Cutoff CV")
        self.out_port = self.add_output("Out")

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout(spacing=6)

        type_layout = QHBoxLayout()
        type_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Low-pass", "High-pass", "Band-pass", "Notch"])
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        cutoff_row = QHBoxLayout()
        cutoff_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cutoff_layout = QVBoxLayout()
        cutoff_label = QLabel("Cutoff")
        cutoff_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cutoff_knob = Knob(
            label="Hz",
            min_value=20,
            max_value=20000,
            default_value=1200,
            logarithmic=True,
        )
        self.cutoff_value_label = QLabel("1200 Hz")
        self.cutoff_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cutoff_knob.value_changed.connect(self._on_cutoff_changed)
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
            max_value=12.0,
            default_value=0.707,
            logarithmic=True,
        )
        self.resonance_value_label = QLabel("0.71")
        self.resonance_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.resonance_knob.value_changed.connect(self._on_resonance_changed)
        resonance_layout.addWidget(resonance_label)
        resonance_layout.addWidget(self.resonance_knob)
        resonance_layout.addWidget(self.resonance_value_label)
        cutoff_row.addLayout(resonance_layout)
        layout.addLayout(cutoff_row)

        character_row = QHBoxLayout()
        character_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cv_layout = QVBoxLayout()
        cv_label = QLabel("CV Depth")
        cv_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cv_depth_knob = Knob(
            label="Oct", min_value=-5.0, max_value=5.0, default_value=0.0
        )
        self.cv_depth_value_label = QLabel("0.0 oct")
        self.cv_depth_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cv_depth_knob.value_changed.connect(self._on_cv_depth_changed)
        cv_layout.addWidget(cv_label)
        cv_layout.addWidget(self.cv_depth_knob)
        cv_layout.addWidget(self.cv_depth_value_label)
        character_row.addLayout(cv_layout)

        drive_layout = QVBoxLayout()
        drive_label = QLabel("Drive")
        drive_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drive_knob = Knob(
            label="dB", min_value=0.0, max_value=24.0, default_value=0.0
        )
        self.drive_value_label = QLabel("0 dB")
        self.drive_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drive_knob.value_changed.connect(self._on_drive_changed)
        drive_layout.addWidget(drive_label)
        drive_layout.addWidget(self.drive_knob)
        drive_layout.addWidget(self.drive_value_label)
        output_layout = QVBoxLayout()
        output_label = QLabel("Output")
        output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_gain_knob = Knob(
            label="dB", min_value=-24.0, max_value=12.0, default_value=-6.0
        )
        self.output_gain_value_label = QLabel("-6 dB")
        self.output_gain_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_gain_knob.value_changed.connect(self._on_output_gain_changed)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_gain_knob)
        output_layout.addWidget(self.output_gain_value_label)
        character_row.addLayout(drive_layout)
        character_row.addLayout(output_layout)
        layout.addLayout(character_row)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.register_parameter("cutoff", self.cutoff_knob)
        self.register_parameter("resonance", self.resonance_knob)
        self.register_parameter("cv_depth_octaves", self.cv_depth_knob)
        self.register_parameter("drive_db", self.drive_knob)
        self.register_parameter("output_gain_db", self.output_gain_knob)
        self.register_parameter(
            "filter_type",
            self.type_combo,
            getter="currentText",
            setter="setCurrentText",
        )

        self.component = self.create_engine_component()
        self._runtime_filter_params: (
            tuple[float, float, Literal["low", "high", "band", "notch"], float, float]
            | None
        ) = None
        self._last_cutoff = self.cutoff_knob.get_value()
        self._last_resonance = self.resonance_knob.get_value()
        self._last_drive_db = self.drive_knob.get_value()
        self._last_output_gain_db = self.output_gain_knob.get_value()
        self._sample_rate_listener = self._on_global_sample_rate_changed
        audio_config.add_sample_rate_listener(self._sample_rate_listener)
        self.destroyed.connect(self._cleanup_audio_config_listeners)

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    @staticmethod
    def _normalize_filter_type(text: str) -> Literal["low", "high", "band", "notch"]:
        type_map: dict[str, Literal["low", "high", "band", "notch"]] = {
            "Low-pass": "low",
            "High-pass": "high",
            "Band-pass": "band",
            "Notch": "notch",
            "low": "low",
            "high": "high",
            "band": "band",
            "notch": "notch",
        }
        return type_map.get(text, "low")

    def _on_cutoff_changed(self) -> None:
        value = self.cutoff_knob.get_value()
        self.cutoff_value_label.setText(f"{int(value)} Hz")
        self.parameter_changed.emit("cutoff", value)

    def _on_resonance_changed(self) -> None:
        value = self.resonance_knob.get_value()
        self.resonance_value_label.setText(f"{value:.2f}")
        self.parameter_changed.emit("resonance", value)

    def _on_cv_depth_changed(self) -> None:
        value = self.cv_depth_knob.get_value()
        self.cv_depth_value_label.setText(f"{value:.1f} oct")
        self.parameter_changed.emit("cv_depth_octaves", value)

    def _on_drive_changed(self) -> None:
        value = self.drive_knob.get_value()
        self.drive_value_label.setText(f"{value:.0f} dB")
        self.parameter_changed.emit("drive_db", value)

    def _on_output_gain_changed(self) -> None:
        value = self.output_gain_knob.get_value()
        self.output_gain_value_label.setText(f"{value:.0f} dB")
        self.parameter_changed.emit("output_gain_db", value)

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        _ = new_sample_rate
        self.component = self.create_engine_component()
        self._runtime_filter_params = None

    def _cleanup_audio_config_listeners(self, *_args: object) -> None:
        audio_config.remove_sample_rate_listener(self._sample_rate_listener)

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> BiquadResonantFilter:
        del input_components, modulation_components
        return BiquadResonantFilter(
            cutoff=self.cutoff_knob.get_value(),
            resonance=self.resonance_knob.get_value(),
            filter_type=self._normalize_filter_type(self.type_combo.currentText()),
            drive_db=self.drive_knob.get_value(),
            output_gain_db=self.output_gain_knob.get_value(),
            sample_rate=audio_config.sample_rate,
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Filter the connected input for one render cycle."""
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        filter_type = self._normalize_filter_type(
            str_parameter(parameters, "filter_type", self.type_combo.currentText)
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
        cutoff_values = self._ramp_if_changed(self._last_cutoff, cutoff, num_samples)
        resonance_values = self._ramp_if_changed(
            self._last_resonance, resonance, num_samples
        )
        drive_db_values = self._ramp_if_changed(
            self._last_drive_db, drive_db, num_samples
        )
        output_gain_db_values = self._ramp_if_changed(
            self._last_output_gain_db, output_gain_db, num_samples
        )

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

    @staticmethod
    def _ramp_if_changed(
        previous: float, current: float, num_samples: int
    ) -> np.ndarray | None:
        if previous == current:
            return None
        return np.linspace(previous, current, num_samples, dtype=np.float32)
