import logging

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout
from soniclab.dsp.effects import Compressor

from sonicrack.gui.modules._modulated_base import ModulatedModuleBase
from sonicrack.gui.modules.effects._cv_modulation import (
    ControlRateCvSpec,
    apply_control_rate_cv,
)
from sonicrack.gui.widgets import Knob, LevelMeter
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)


@register_module()
class CompressorModule(ModulatedModuleBase):
    """Compressor module with optional CV for threshold, ratio, and mix."""

    runtime_kind = "compressor"
    metadata = ModuleMetadata(
        title="Compressor",
        category=ModuleCategory.EFFECT,
        description="Control dynamic range with threshold, ratio, and timing",
    )

    def __init__(self):
        super().__init__(
            width=280,
            height=350,
            color=QColor(160, 90, 190),
        )

        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.threshold_cv_port = self.add_input("CV_Thresh")
        self.ratio_cv_port = self.add_input("CV_Ratio")
        self.mix_cv_port = self.add_input("CV_Mix")
        self.component = None

        layout = self._begin_controls()

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

        meter_col = QVBoxLayout()
        meter_col.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        gr_label = QLabel("GR")
        gr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meter_col.addWidget(gr_label)
        self.gr_meter = LevelMeter(width=12, height=52)
        self.gr_meter.setToolTip("Gain reduction")
        meter_col.addWidget(self.gr_meter)
        output_row.addLayout(meter_col)
        layout.addLayout(output_row)

        self._gr_level = 0.0
        self._meter_timer = QTimer(self)
        self._meter_timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._meter_timer.setInterval(40)
        self._meter_timer.timeout.connect(self._refresh_gr_meter)
        self._meter_timer.start()

        self._finish_controls(layout)

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
        if self._require_input_or_silence(num_samples):
            return

        # Thread-safe component access with lock
        with self._component_lock:
            if self.component is None:
                self.component = self.create_unmodulated_component()

            self.component.attack_ms = float_parameter(
                parameters, "attack_ms", self.attack_knob.get_value
            )
            self.component.release_ms = float_parameter(
                parameters, "release_ms", self.release_knob.get_value
            )
            self.component.makeup_gain_db = float_parameter(
                parameters, "makeup_gain_db", self.makeup_knob.get_value
            )

            # Scale threshold/ratio CV so ±1 is a useful offset in dB / ratio.
            apply_control_rate_cv(
                self.component,
                parameters,
                (
                    ControlRateCvSpec(
                        "threshold_db",
                        "threshold_db",
                        self.threshold_knob.get_value,
                        self.threshold_cv_port,
                        -60.0,
                        0.0,
                        scale=24.0,
                    ),
                    ControlRateCvSpec(
                        "ratio",
                        "ratio",
                        self.ratio_knob.get_value,
                        self.ratio_cv_port,
                        1.0,
                        20.0,
                        scale=4.0,
                    ),
                    ControlRateCvSpec(
                        "mix",
                        "mix",
                        self.mix_knob.get_value,
                        self.mix_cv_port,
                        0.0,
                        1.0,
                    ),
                ),
                num_samples,
            )

            self.out_port.write(self.component(read_samples(self.in_port, num_samples)))
            reduction = float(getattr(self.component, "gain_reduction_db", 0.0))
            self._gr_level = min(1.0, abs(reduction) / 24.0)

    def _refresh_gr_meter(self) -> None:
        self.gr_meter.set_level(self._gr_level)
        self.gr_meter.decay()
        self._gr_level *= 0.85
