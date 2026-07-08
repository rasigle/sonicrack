import logging

from PyQt6 import QtWidgets
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.dsp.effects import Distortion

from sonicrack.gui.modules._modulated_base import ModulatedModuleBase
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import (
    float_parameter,
    read_samples,
    silence,
    str_parameter,
)
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)


DEFAULT_DISTORTION_TYPE = "Distortion"


@register_module()
class DistortionModule(ModulatedModuleBase):
    """Distortion module"""

    runtime_kind = "distortion"
    metadata = ModuleMetadata(
        title="Distortion",
        category=ModuleCategory.MODIFIER,
        description="Apply distortion effect to audio signal",
    )

    def __init__(self):
        """Initialize panner module."""
        super().__init__(
            width=220,
            height=240,
            color=QColor(180, 80, 180),
        )

        # Add ports
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")
        self.drive_cv_port = self.add_input("CV_Drive")
        self.mix_cv_port = self.add_input("CV_Mix")
        self.component = None

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        knobs_row = QHBoxLayout()
        knobs_row.setSpacing(10)

        self.drive_knob = Knob(
            label="Drive",
            description="Controls the amount of distortion",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self.drive_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("drive", self.drive_knob.get_value())
        )
        knobs_row.addWidget(self.drive_knob)

        self.mix_knob = Knob(
            label="Mix",
            description="Controls the dry/wet mix of the distortion effect",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self.mix_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("mix", self.mix_knob.get_value())
        )
        knobs_row.addWidget(self.mix_knob)
        layout.addLayout(knobs_row)

        self.distortion_combo = QtWidgets.QComboBox()
        self.distortion_combo.addItems(["soft", "hard", "fuzz", "tube"])
        self.distortion_combo.currentTextChanged.connect(self._on_type_changed)
        layout.addWidget(QtWidgets.QLabel("Type:"))
        layout.addWidget(self.distortion_combo)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("drive", self.drive_knob)
        self.register_parameter("mix", self.mix_knob)
        self.register_parameter(
            "distortion_type",
            self.distortion_combo,
            getter="currentText",
            setter="setCurrentText",
        )

        # Set control_knob for base class functionality
        self.control_knob = self.drive_knob

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Distortion requires the In port to be connected."""
        return ["In"]

    def get_modulation_inputs(self) -> list[str]:
        """Distortion accepts CV modulation for drive and mix."""
        return ["CV_Drive", "CV_Mix"]

    def get_cv_range(self, port_name: str = "CV_Drive") -> tuple[float, float]:
        """Distortion CV inputs expect bipolar offsets [-1, 1].

        Returns:
            (-1.0, 1.0) - bipolar range
        """
        _ = port_name
        return -1.0, 1.0

    def _on_type_changed(self, distortion_type: str):
        """Handle distortion type change."""
        self.component = self.create_engine_component()
        self.parameter_changed.emit("distortion_type", distortion_type)

    # Implement abstract methods from ModulatedModuleBase
    def create_modulated_component(self, mod_comp):
        """Create modulated distortion (not implemented yet)."""
        _ = mod_comp
        return self.create_unmodulated_component()

    def create_unmodulated_component(self):
        """Create simple Distortion without modulation."""
        drive = self.drive_knob.get_value()
        mix = self.mix_knob.get_value()
        return Distortion(drive=drive, mix=mix)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Apply distortion during an engine-owned render cycle."""
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        # Thread-safe component access with lock
        with self._component_lock:
            if self.component is None:
                self.component = self.create_unmodulated_component()

            self.component.drive = float_parameter(
                parameters, "drive", self.drive_knob.get_value
            )
            self.component.mix = float_parameter(
                parameters, "mix", self.mix_knob.get_value
            )
            self.component.distortion_type = str_parameter(
                parameters, "distortion_type", self.distortion_combo.currentText
            )

            input_signal = read_samples(self.in_port, num_samples)
            drive_cv = (
                read_samples(self.drive_cv_port, num_samples)
                if self.drive_cv_port.is_connected
                else None
            )
            mix_cv = (
                read_samples(self.mix_cv_port, num_samples)
                if self.mix_cv_port.is_connected
                else None
            )
            if drive_cv is None and mix_cv is None:
                self.out_port.write(self.component(input_signal))
                return

            self.out_port.write(
                self._process_modulated_distortion(
                    input_signal,
                    base_drive=self.component.drive,
                    base_mix=self.component.mix,
                    drive_cv=drive_cv,
                    mix_cv=mix_cv,
                )
            )

    def _process_modulated_distortion(
        self,
        input_signal,
        *,
        base_drive: float,
        base_mix: float,
        drive_cv,
        mix_cv,
    ):
        """Process audio while applying per-sample drive/mix CV offsets."""
        import numpy as np

        output = np.empty(len(input_signal), dtype=np.float32)
        drive_values = (
            np.clip(base_drive + drive_cv, 0.0, 10.0)
            if drive_cv is not None
            else np.full(len(input_signal), base_drive, dtype=np.float32)
        )
        mix_values = (
            np.clip(base_mix + mix_cv, 0.0, 1.0)
            if mix_cv is not None
            else np.full(len(input_signal), base_mix, dtype=np.float32)
        )

        for index, sample in enumerate(input_signal):
            self.component.drive = float(drive_values[index])
            self.component.mix = float(mix_values[index])
            output[index] = self.component(float(sample))

        self.component.drive = base_drive
        self.component.mix = base_mix
        return output
