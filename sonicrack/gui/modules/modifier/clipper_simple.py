import logging
from typing import Any

from PyQt6.QtGui import QColor
from soniclab import Clipper

from sonicrack.config.audio_config import get_sample_rate
from sonicrack.gui.modules.modifier._simple_base import SimpleModifierBase
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import read_samples
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)


@register_module()
class ClipperModule(SimpleModifierBase):
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

        self._setup_audio_io()

        self.threshold_knob = Knob(
            label="Threshold",
            description="Sets the clipping threshold",
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
        )
        self._build_single_knob_controls(
            self.threshold_knob,
            "threshold",
            on_change=self._on_threshold_changed,
        )

        self.component = self.create_engine_component()

    def _on_threshold_changed(self, new_threshold: float) -> None:
        """Handle threshold knob changes by updating Clipper component wave_range."""
        logger.debug(f"Clipper: threshold knob changed to {new_threshold:.3f}")
        if self.component:
            self.component.wave_range = (-new_threshold, new_threshold)
            logger.debug(f"Clipper: wave_range set to +/- {new_threshold:.3f}")

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the clipper component."""
        del input_components, modulation_components
        threshold = self.threshold_knob.get_value()
        sample_rate = get_sample_rate()
        return Clipper((-threshold, threshold), sample_rate=sample_rate)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Clip the connected input for one render cycle."""
        del parameters
        if self._require_input_or_silence(num_samples):
            return

        if self.component is None:
            self.component = self.create_engine_component()

        samples = read_samples(self.in_port, num_samples)
        self.out_port.write(self.component(samples))
