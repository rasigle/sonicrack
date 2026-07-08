import logging
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from soniclab import Clipper

from sonicrack.config.audio_config import get_sample_rate
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import read_samples, silence
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)


@register_module()
class ClipperModule(ModuleWidget):
    """Clipper module for distortion/limiting."""

    runtime_kind = "clipper"

    metadata = ModuleMetadata(
        title="Clipper",
        category=ModuleCategory.MODIFIER,
        description="Audio clipper for distortion/limiting",
    )

    def __init__(self):
        """Initialize clipper module."""
        super().__init__(
            width=140,
            height=180,
            color=QColor(200, 150, 80),
        )

        # Add ports
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Threshold knob
        self.threshold_knob = Knob(
            label="Threshold",
            description="Sets the clipping threshold",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self.threshold_knob.value_changed.connect(self._on_threshold_changed)
        layout.addWidget(self.threshold_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("threshold", self.threshold_knob)

        self.component = self.create_engine_component()

    def _on_threshold_changed(self):
        """Handle threshold knob changes by updating Clipper component wave_range.

        This method is called whenever the threshold knob value changes.
        It updates the clipper's wave range to [-threshold, +threshold].
        """
        new_threshold = self.threshold_knob.get_value()
        logger.debug(f"Clipper: threshold knob changed to {new_threshold:.3f}")

        # Update the Clipper component wave_range (click-free)
        if self.component:
            self.component.wave_range = (-new_threshold, new_threshold)
            logger.debug(f"Clipper: wave_range set to +/- {new_threshold:.3f}")

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Clipper requires the In port to be connected."""
        return ["In"]

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the clipper component."""
        threshold = self.threshold_knob.get_value()
        sample_rate = get_sample_rate()
        return Clipper((-threshold, threshold), sample_rate=sample_rate)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Clip the connected input for one render cycle."""
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        if self.component is None:
            self.component = self.create_engine_component()

        samples = read_samples(self.in_port, num_samples)
        self.out_port.write(self.component(samples))
