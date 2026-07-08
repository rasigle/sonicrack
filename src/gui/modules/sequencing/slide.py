"""Pitch slide module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QCheckBox
from soniclab.sequencing import SlideProcessor

from src.gui.audio_config import audio_config
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, read_samples, silence
from src.gui.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


@register_module()
class SlideModule(ModuleWidget):
    """Smooth pitch CV changes when slide CV is active."""

    runtime_kind = "slide"

    metadata = ModuleMetadata(
        title="Slide",
        category=ModuleCategory.MODIFIER,
        description="Portamento processor for 1V/oct pitch CV",
    )

    def __init__(self) -> None:
        super().__init__(width=180, height=190, color=QColor(110, 145, 95))

        self.freq_input = self.add_input("Freq In")
        self.slide_input = self.add_input("Slide")
        self.freq_output = self.add_output("Freq Out")
        self.component = SlideProcessor(sample_rate=audio_config.sample_rate)

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        self.time_knob = Knob(
            label="Time", min_value=0.0, max_value=0.5, default_value=0.08
        )
        self.time_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("time", self.time_knob.get_value())
        )
        layout.addWidget(self.time_knob)

        self.always_checkbox = QCheckBox("Always")
        self.always_checkbox.toggled.connect(
            lambda value: self.parameter_changed.emit("always_on", value)
        )
        layout.addWidget(self.always_checkbox)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.register_parameter("time", self.time_knob)
        self.register_parameter(
            "always_on",
            self.always_checkbox,
            getter="isChecked",
            setter="setChecked",
        )

        self._sample_rate_listener = self._on_global_sample_rate_changed
        audio_config.add_sample_rate_listener(self._sample_rate_listener)
        self.destroyed.connect(self._cleanup_audio_config_listeners)

    def get_required_inputs(self) -> list[str]:
        return ["Freq In"]

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component.sample_rate = new_sample_rate
        self.component.reset()

    def _cleanup_audio_config_listeners(self, *_args: object) -> None:
        audio_config.remove_sample_rate_listener(self._sample_rate_listener)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if not self.freq_input.is_connected:
            self.freq_output.write(silence(num_samples))
            return

        self.component.time = float_parameter(
            parameters, "time", self.time_knob.get_value
        )
        self.component.always_on = bool(
            parameters.get("always_on", self.always_checkbox.isChecked())
        )
        slide_signal = (
            read_samples(self.slide_input, num_samples)
            if self.slide_input.is_connected
            else None
        )
        self.freq_output.write(
            self.component.process(
                read_samples(self.freq_input, num_samples), slide_signal
            )
        )
