from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from soniclab import ModulatedVolume, Volume

from sonicrack.constants import DEFAULT_GAIN_DB
from sonicrack.gui.modules._modulated_base import ModulatedModuleBase
from sonicrack.gui.widgets import Knob
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter
from sonicrack.runtime.specs import RuntimeParameters

logger = logging.getLogger(__name__)


@register_module()
class VolumeModule(ModulatedModuleBase):
    """Volume/Gain module with modulation support."""

    runtime_kind = "volume_mod"

    metadata = ModuleMetadata(
        title="Volume (Mod)",
        category=ModuleCategory.MODIFIER,
        description="Volume control with modulation input",
    )

    def __init__(self):
        """Initialize volume module."""
        super().__init__(
            width=140,
            height=150,
            color=QColor(180, 120, 80),
        )

        self._setup_modulated_io()

        self.gain_knob = Knob(
            label="Gain (dB)",
            min_value=-60,
            max_value=12,
            default_value=DEFAULT_GAIN_DB,
            logarithmic=False,
        )
        layout = self._begin_controls()
        self.bind_parameter_knob(self.gain_knob, "gain_db")
        layout.addWidget(self.gain_knob, alignment=Qt.AlignmentFlag.AlignCenter)
        self._finish_controls(layout)

        self.register_parameter("gain_db", self.gain_knob)
        self.control_knob = self.gain_knob
        self.component = self.create_unmodulated_component()

    def get_required_inputs(self) -> list[str]:
        """Volume requires the In port to be connected."""
        return ["In"]

    def create_modulated_component(self, mod_comp):
        """Create ModulatedVolume with modulation."""
        logger.info(f"VolumeModule: Creating ModulatedVolume with modulator {mod_comp}")
        return ModulatedVolume(mod_comp, modulation_target="gain_db")

    def create_unmodulated_component(self):
        """Create simple Volume without modulation."""
        gain_db = self.gain_knob.get_value()
        logger.debug(f"VolumeModule: Creating simple Volume with gain_db={gain_db}")
        return Volume(gain_db=gain_db)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Apply gain for one render cycle."""
        input_signal = self.read_modulated_input_or_silence(num_samples)
        if input_signal is None:
            return

        with self._component_lock:
            gain_db = float_parameter(parameters, "gain_db", self.gain_knob.get_value)

            if self.mod_port.is_connected:
                # When modulated: knob controls modulation depth (0.0 to 1.0)
                # Map gain_db range [-60, 12] to modulation amount [0.0, 1.0]
                modulation_amount = (gain_db + 60) / 72
                modulation_amount = max(0.0, min(1.0, modulation_amount))
                if self.port_adapter is not None:
                    self.port_adapter.modulation_amount = modulation_amount
            elif self.component is not None:
                self.component.gain_db = gain_db

            if self.component is not None:
                self.out_port.write(self.component(input_signal))
            else:
                self._write_silence(num_samples)
