"""Accent CV processor module."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.sequencing import AccentProcessor

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters


@register_module()
class AccentModule(ModuleWidget):
    """Shape accent gates into reusable CV outputs."""

    runtime_kind = "accent"

    metadata = ModuleMetadata(
        title="Accent",
        category=ModuleCategory.MODULATED_SOURCE,
        description="Convert accent gates into amp, cutoff, and envelope CV",
    )

    def __init__(self) -> None:
        super().__init__(width=288, height=245, color=QColor(175, 120, 55))

        self.accent_input = self.add_input("Accent")
        self.amp_port = self.add_output("Amp CV")
        self.cutoff_port = self.add_output("Cutoff CV")
        self.env_port = self.add_output("Env CV")
        self.component = AccentProcessor(sample_rate=audio_config.sample_rate)

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        top_row = QHBoxLayout()
        top_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.amount_knob = Knob(
            label="Amount", min_value=0.0, max_value=1.0, default_value=1.0
        )
        self.amount_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("amount", self.amount_knob.get_value())
        )
        top_row.addWidget(self.amount_knob)

        self.decay_knob = Knob(
            label="Decay", min_value=0.0, max_value=0.5, default_value=0.08
        )
        self.decay_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("decay", self.decay_knob.get_value())
        )
        top_row.addWidget(self.decay_knob)
        layout.addLayout(top_row)

        depth_row = QHBoxLayout()
        depth_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.amp_depth_knob = Knob(
            label="Amp", min_value=0.0, max_value=1.0, default_value=0.35
        )
        self.amp_depth_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "amp_depth", self.amp_depth_knob.get_value()
            )
        )
        depth_row.addWidget(self.amp_depth_knob)

        self.cutoff_depth_knob = Knob(
            label="Cutoff", min_value=0.0, max_value=1.0, default_value=0.6
        )
        self.cutoff_depth_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "cutoff_depth", self.cutoff_depth_knob.get_value()
            )
        )
        depth_row.addWidget(self.cutoff_depth_knob)

        self.env_depth_knob = Knob(
            label="Env", min_value=0.0, max_value=1.0, default_value=0.5
        )
        self.env_depth_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "envelope_depth", self.env_depth_knob.get_value()
            )
        )
        depth_row.addWidget(self.env_depth_knob)
        layout.addLayout(depth_row)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.register_parameter("amount", self.amount_knob)
        self.register_parameter("decay", self.decay_knob)
        self.register_parameter("amp_depth", self.amp_depth_knob)
        self.register_parameter("cutoff_depth", self.cutoff_depth_knob)
        self.register_parameter("envelope_depth", self.env_depth_knob)

        self._sample_rate_listener = self._on_global_sample_rate_changed
        audio_config.add_sample_rate_listener(self._sample_rate_listener)
        self.destroyed.connect(self._cleanup_audio_config_listeners)

    def get_required_inputs(self) -> list[str]:
        return ["Accent"]

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component.sample_rate = new_sample_rate
        self.component.reset()

    def _cleanup_audio_config_listeners(self, *_args: object) -> None:
        audio_config.remove_sample_rate_listener(self._sample_rate_listener)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        if not self.accent_input.is_connected:
            self.amp_port.write(silence(num_samples))
            self.cutoff_port.write(silence(num_samples))
            self.env_port.write(silence(num_samples))
            return

        self.component.amount = float_parameter(
            parameters, "amount", self.amount_knob.get_value
        )
        self.component.decay = float_parameter(
            parameters, "decay", self.decay_knob.get_value
        )
        self.component.amp_depth = float_parameter(
            parameters, "amp_depth", self.amp_depth_knob.get_value
        )
        self.component.cutoff_depth = float_parameter(
            parameters, "cutoff_depth", self.cutoff_depth_knob.get_value
        )
        self.component.envelope_depth = float_parameter(
            parameters, "envelope_depth", self.env_depth_knob.get_value
        )

        frame = self.component.process(read_samples(self.accent_input, num_samples))
        self.amp_port.write(frame.amp)
        self.cutoff_port.write(frame.cutoff)
        self.env_port.write(frame.envelope)
