import logging
from typing import Any

from PyQt6.QtGui import QColor
from soniclab import Volume

from sonicrack.constants import DEFAULT_GAIN_DB
from sonicrack.gui.modules.modifier._simple_base import SimpleModifierBase
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)


@register_module()
class SimpleVolumeModule(SimpleModifierBase):
    """Simple volume/gain module without modulation input."""

    runtime_kind = "volume"

    metadata = ModuleMetadata(
        title="Volume",
        category=ModuleCategory.MODIFIER,
        description="Simple volume/gain control",
    )

    def __init__(self):
        """Initialize simple volume module."""
        super().__init__(
            width=140,
            height=175,
            color=QColor(160, 100, 60),
        )

        self._setup_audio_io()

        # Gain in dB (linear mapping of dB values, since dB is already logarithmic)
        # Range: -60 dB (very quiet) to +12 dB (boost)
        # Default: -20 dB (safe for mixing)
        self.gain_knob = Knob(
            label="Gain (dB)",
            min_value=-60,
            max_value=12,
            default_value=DEFAULT_GAIN_DB,
            logarithmic=False,
        )
        self._build_single_knob_controls(
            self.gain_knob,
            "gain_db",
            on_change=self._on_gain_changed,
        )

        self.component = self.create_engine_component()

    def _on_gain_changed(self, value: float) -> None:
        """Handle gain knob changes by updating Volume component."""
        # Knob is calibrated in dB; Volume.amplitude expects linear gain >= 0.
        if self.component is not None:
            self.component.gain_db = value
        logger.debug(f"Volume: gain set to {value:.3f} dB")

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the volume component."""
        del input_components, modulation_components
        return Volume(gain_db=self.gain_knob.get_value())

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Apply gain to the connected input for one render cycle."""
        if self._require_input_or_silence(num_samples):
            return

        input_signal = read_samples(self.in_port, num_samples)
        if self.component is None:
            self.component = self.create_engine_component()

        self.component.gain_db = float_parameter(
            parameters, "gain_db", self.gain_knob.get_value
        )
        self.out_port.write(self.component(input_signal))
