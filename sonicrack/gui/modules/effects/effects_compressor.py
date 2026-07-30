import logging

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.effects import Compressor

from sonicrack.gui.modules._modulated_base import ModulatedModuleBase
from sonicrack.gui.modules.effects._cv_modulation import (
    modulate_param,
    read_optional_cv,
)
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)


@register_module()
class CompressorModule(ModulatedModuleBase):
    """Compressor module with optional CV for threshold, ratio, and mix."""

    runtime_kind = "compressor"
    metadata = ModuleMetadata(
        title="Compressor",
        category=ModuleCategory.MODIFIER,
        description="Control dynamic range with threshold, ratio, and timing",
    )

    def __init__(self):
        super().__init__(
            width=260,
            height=330,
            color=QColor(160, 90, 190),
        )

        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.threshold_cv_port = self.add_input("CV_Thresh")
        self.ratio_cv_port = self.add_input("CV_Ratio")
        self.mix_cv_port = self.add_input("CV_Mix")
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

    def get_modulation_inputs(self) -> list[str]:
        """Compressor accepts CV modulation for threshold, ratio, and mix."""
        return ["CV_Thresh", "CV_Ratio", "CV_Mix"]

    def get_cv_range(self, port_name: str = "CV_Thresh") -> tuple[float, float]:
        """Compressor CV inputs expect bipolar offsets [-1, 1]."""
        _ = port_name
        return -1.0, 1.0

    def create_modulated_component(self, mod_comp):
        """CV is applied in process_runtime; return the base component."""
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

            base_threshold = float_parameter(
                parameters, "threshold_db", self.threshold_knob.get_value
            )
            base_ratio = float_parameter(parameters, "ratio", self.ratio_knob.get_value)
            base_attack = float_parameter(
                parameters, "attack_ms", self.attack_knob.get_value
            )
            base_release = float_parameter(
                parameters, "release_ms", self.release_knob.get_value
            )
            base_makeup = float_parameter(
                parameters, "makeup_gain_db", self.makeup_knob.get_value
            )
            base_mix = float_parameter(parameters, "mix", self.mix_knob.get_value)

            threshold_cv = read_optional_cv(self.threshold_cv_port, num_samples)
            ratio_cv = read_optional_cv(self.ratio_cv_port, num_samples)
            mix_cv = read_optional_cv(self.mix_cv_port, num_samples)

            # Scale CV so ±1 is a useful offset in dB / ratio units.
            self.component.threshold_db = modulate_param(
                base_threshold,
                threshold_cv,
                minimum=-60.0,
                maximum=0.0,
                scale=24.0,
            )
            self.component.ratio = modulate_param(
                base_ratio, ratio_cv, minimum=1.0, maximum=20.0, scale=4.0
            )
            self.component.attack_ms = base_attack
            self.component.release_ms = base_release
            self.component.makeup_gain_db = base_makeup
            self.component.mix = modulate_param(
                base_mix, mix_cv, minimum=0.0, maximum=1.0
            )

            self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
