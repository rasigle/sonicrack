"""Acid-style resonant filter module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from src.engine.filter_303 import AcidResonantFilter
from src.gui.audio_config import audio_config
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, read_samples, silence
from src.gui.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


@register_module()
class AcidFilterModule(ModuleWidget):
    """303-oriented resonant low-pass filter with env and accent CV."""

    runtime_kind = "acid_filter"

    metadata = ModuleMetadata(
        title="Acid Filter",
        category=ModuleCategory.MODIFIER,
        description="Acid-style resonant low-pass filter with drive",
    )

    def __init__(self) -> None:
        super().__init__(width=260, height=315, color=QColor(85, 165, 120))

        self.in_port = self.add_input("In")
        self.cutoff_cv_port = self.add_input("Cutoff CV")
        self.env_cv_port = self.add_input("Env CV")
        self.accent_cv_port = self.add_input("Accent CV")
        self.out_port = self.add_output("Out")

        self.component = AcidResonantFilter(sample_rate=audio_config.sample_rate)

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        tone_row = QHBoxLayout()
        self.cutoff_knob = Knob("Cutoff", 20.0, 12000.0, 700.0, logarithmic=True)
        self.cutoff_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("cutoff", self.cutoff_knob.get_value())
        )
        tone_row.addWidget(self.cutoff_knob)

        self.resonance_knob = Knob("Res", 0.1, 18.0, 8.0, logarithmic=True)
        self.resonance_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "resonance", self.resonance_knob.get_value()
            )
        )
        tone_row.addWidget(self.resonance_knob)
        layout.addLayout(tone_row)

        mod_row = QHBoxLayout()
        self.env_amount_knob = Knob("Env", 0.0, 6.0, 2.5)
        self.env_amount_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "env_amount", self.env_amount_knob.get_value()
            )
        )
        mod_row.addWidget(self.env_amount_knob)

        self.accent_amount_knob = Knob("Accent", 0.0, 4.0, 1.0)
        self.accent_amount_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "accent_amount", self.accent_amount_knob.get_value()
            )
        )
        mod_row.addWidget(self.accent_amount_knob)
        layout.addLayout(mod_row)

        gain_row = QHBoxLayout()
        self.drive_knob = Knob("Drive", 0.0, 24.0, 6.0)
        self.drive_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("drive_db", self.drive_knob.get_value())
        )
        gain_row.addWidget(self.drive_knob)

        self.output_gain_knob = Knob("Out", -24.0, 12.0, -6.0)
        self.output_gain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "output_gain_db", self.output_gain_knob.get_value()
            )
        )
        gain_row.addWidget(self.output_gain_knob)
        layout.addLayout(gain_row)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.register_parameter("cutoff", self.cutoff_knob)
        self.register_parameter("resonance", self.resonance_knob)
        self.register_parameter("env_amount", self.env_amount_knob)
        self.register_parameter("accent_amount", self.accent_amount_knob)
        self.register_parameter("drive_db", self.drive_knob)
        self.register_parameter("output_gain_db", self.output_gain_knob)

        self._sample_rate_listener = self._on_global_sample_rate_changed
        audio_config.add_sample_rate_listener(self._sample_rate_listener)
        self.destroyed.connect(self._cleanup_audio_config_listeners)

    def get_required_inputs(self) -> list[str]:
        return ["In"]

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component = AcidResonantFilter(sample_rate=new_sample_rate)

    def _cleanup_audio_config_listeners(self, *_args: object) -> None:
        audio_config.remove_sample_rate_listener(self._sample_rate_listener)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        self.component.cutoff = float_parameter(
            parameters, "cutoff", self.cutoff_knob.get_value
        )
        self.component.resonance = float_parameter(
            parameters, "resonance", self.resonance_knob.get_value
        )
        self.component.env_amount = float_parameter(
            parameters, "env_amount", self.env_amount_knob.get_value
        )
        self.component.accent_amount = float_parameter(
            parameters, "accent_amount", self.accent_amount_knob.get_value
        )
        self.component.drive_db = float_parameter(
            parameters, "drive_db", self.drive_knob.get_value
        )
        self.component.output_gain_db = float_parameter(
            parameters, "output_gain_db", self.output_gain_knob.get_value
        )

        self.out_port.write(
            self.component.process_modulated(
                read_samples(self.in_port, num_samples),
                cutoff_cv=(
                    read_samples(self.cutoff_cv_port, num_samples)
                    if self.cutoff_cv_port.is_connected
                    else None
                ),
                env_cv=(
                    read_samples(self.env_cv_port, num_samples)
                    if self.env_cv_port.is_connected
                    else None
                ),
                accent_cv=(
                    read_samples(self.accent_cv_port, num_samples)
                    if self.accent_cv_port.is_connected
                    else None
                ),
            )
        )
