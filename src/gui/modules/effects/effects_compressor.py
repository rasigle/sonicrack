import logging

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.effects import Compressor

from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, read_samples, silence
from src.gui.module_registry import register_module
from src.gui.modules._modulated_base import ModulatedModuleBase
from src.gui.widgets import Knob

logger = logging.getLogger(__name__)


@register_module()
class CompressorModule(ModulatedModuleBase):
    """Compressor module."""

    runtime_kind = "compressor"
    metadata = ModuleMetadata(
        title="Compressor",
        category=ModuleCategory.MODIFIER,
        description="Control dynamic range with threshold, ratio, and timing",
    )

    def __init__(self):
        super().__init__(
            width=260,
            height=270,
            color=QColor(160, 90, 190),
        )

        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.component = None

        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        threshold_ratio_row = QHBoxLayout()
        threshold_ratio_row.setSpacing(10)
        self.threshold_knob = Knob(
            label="Thresh",
            description="Sets the threshold for the compressor",
            min_value=-60.0,
            max_value=0.0,
            default_value=-18.0,
        )
        self.threshold_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "threshold_db", self.threshold_knob.get_value()
            )
        )
        threshold_ratio_row.addWidget(self.threshold_knob)

        self.ratio_knob = Knob(
            label="Ratio",
            description="Sets the compression ratio",
            min_value=1.0,
            max_value=20.0,
            default_value=4.0,
        )
        self.ratio_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("ratio", self.ratio_knob.get_value())
        )
        threshold_ratio_row.addWidget(self.ratio_knob)
        layout.addLayout(threshold_ratio_row)

        timing_row = QHBoxLayout()
        timing_row.setSpacing(10)
        self.attack_knob = Knob(
            label="Attack",
            description="Sets the attack time of the compressor",
            min_value=0.001,
            max_value=200.0,
            default_value=10.0,
        )
        self.attack_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "attack_ms", self.attack_knob.get_value()
            )
        )
        timing_row.addWidget(self.attack_knob)

        self.release_knob = Knob(
            label="Release",
            description="Sets the release time of the compressor",
            min_value=1.0,
            max_value=1000.0,
            default_value=100.0,
        )
        self.release_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "release_ms", self.release_knob.get_value()
            )
        )
        timing_row.addWidget(self.release_knob)
        layout.addLayout(timing_row)

        output_row = QHBoxLayout()
        output_row.setSpacing(10)
        self.makeup_knob = Knob(
            label="Makeup",
            description="Adjusts the makeup gain after compression",
            min_value=-24.0,
            max_value=24.0,
            default_value=0.0,
        )
        self.makeup_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "makeup_gain_db", self.makeup_knob.get_value()
            )
        )
        output_row.addWidget(self.makeup_knob)

        self.mix_knob = Knob(
            label="Mix",
            description="Controls the dry/wet mix of the compressor",
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
        )
        self.mix_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("mix", self.mix_knob.get_value())
        )
        output_row.addWidget(self.mix_knob)
        layout.addLayout(output_row)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.register_parameter("threshold_db", self.threshold_knob)
        self.register_parameter("ratio", self.ratio_knob)
        self.register_parameter("attack_ms", self.attack_knob)
        self.register_parameter("release_ms", self.release_knob)
        self.register_parameter("makeup_gain_db", self.makeup_knob)
        self.register_parameter("mix", self.mix_knob)

        self.control_knob = self.threshold_knob

    def get_required_inputs(self) -> list[str]:
        """Compressor requires an audio input."""
        return ["In"]

    def create_modulated_component(self, mod_comp):
        """Create modulated compressor (not implemented yet)."""
        _ = mod_comp
        return self.create_unmodulated_component()

    def create_unmodulated_component(self):
        """Create compressor engine component from current controls."""
        return Compressor(
            threshold_db=self.threshold_knob.get_value(),
            ratio=self.ratio_knob.get_value(),
            attack_ms=self.attack_knob.get_value(),
            release_ms=self.release_knob.get_value(),
            makeup_gain_db=self.makeup_knob.get_value(),
            mix=self.mix_knob.get_value(),
        )

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Apply compression during an engine-owned render cycle."""
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        # Thread-safe component access with lock
        with self._component_lock:
            if self.component is None:
                self.component = self.create_unmodulated_component()

            self.component.threshold_db = float_parameter(
                parameters, "threshold_db", self.threshold_knob.get_value
            )
            self.component.ratio = float_parameter(
                parameters, "ratio", self.ratio_knob.get_value
            )
            self.component.attack_ms = float_parameter(
                parameters, "attack_ms", self.attack_knob.get_value
            )
            self.component.release_ms = float_parameter(
                parameters, "release_ms", self.release_knob.get_value
            )
            self.component.makeup_gain_db = float_parameter(
                parameters, "makeup_gain_db", self.makeup_knob.get_value
            )
            self.component.mix = float_parameter(
                parameters, "mix", self.mix_knob.get_value
            )

            self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
