"""Peak limiter effect module."""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.effects import Limiter

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.effects._cv_modulation import ControlRateCvSpec
from sonicrack.gui.modules.effects._simple_effect import SimpleEffectModule
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module


@register_module()
class LimiterModule(SimpleEffectModule):
    """Peak limiter for finishing loud patches without digital trash."""

    runtime_kind = "limiter"
    metadata = ModuleMetadata(
        title="Limiter",
        category=ModuleCategory.EFFECT,
        description="Peak limiter with threshold, release, and makeup gain",
    )

    def __init__(self) -> None:
        super().__init__(width=220, height=235, color=QColor(160, 90, 90))
        self._setup_effect_io(Limiter(sample_rate=audio_config.sample_rate))

        layout = self._begin_controls()
        row = QHBoxLayout()
        self.threshold_knob = Knob(
            label="Thresh",
            description="Peak threshold (linear)",
            min_value=0.1,
            max_value=1.0,
            default_value=0.9,
        )
        self.bind_parameter_knob(self.threshold_knob, "threshold", register=True)
        row.addWidget(self.threshold_knob)

        self.release_knob = Knob(
            label="Release",
            description="Release time in milliseconds",
            min_value=5.0,
            max_value=500.0,
            default_value=50.0,
            logarithmic=True,
        )
        self.bind_parameter_knob(self.release_knob, "release", register=True)
        row.addWidget(self.release_knob)
        layout.addLayout(row)

        self.makeup_knob = Knob(
            label="Makeup",
            description="Output makeup gain",
            min_value=0.5,
            max_value=4.0,
            default_value=1.0,
        )
        self.bind_parameter_knob(self.makeup_knob, "makeup", register=True)
        layout.addWidget(self.makeup_knob)
        self._finish_controls(layout)
        self._install_sample_rate_listener()

    def control_rate_specs(self) -> tuple[ControlRateCvSpec, ...]:
        return (
            ControlRateCvSpec(
                "threshold",
                "threshold",
                self.threshold_knob.get_value,
                None,
                self.threshold_knob.min_value,
                self.threshold_knob.max_value,
            ),
            ControlRateCvSpec(
                "release_ms",
                "release",
                self.release_knob.get_value,
                None,
                self.release_knob.min_value,
                self.release_knob.max_value,
            ),
            ControlRateCvSpec(
                "makeup",
                "makeup",
                self.makeup_knob.get_value,
                None,
                self.makeup_knob.min_value,
                self.makeup_knob.max_value,
            ),
        )

    def _on_global_sample_rate_changed(self, new_sample_rate: int) -> None:
        self.component = Limiter(
            threshold=self.threshold_knob.get_value(),
            release_ms=self.release_knob.get_value(),
            makeup=self.makeup_knob.get_value(),
            sample_rate=new_sample_rate,
        )
