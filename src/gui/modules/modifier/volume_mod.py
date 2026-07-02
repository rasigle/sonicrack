from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.constants import DEFAULT_GAIN_DB
from src.engine import ModulatedVolume, Volume
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter, read_samples, silence
from src.gui.module_registry import register_module
from src.gui.modules._modulated_base import ModulatedModuleBase
from src.gui.widgets import Knob

if TYPE_CHECKING:
    from src.gui.core.port import Port

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
            height=180,
            color=QColor(180, 120, 80),
        )

        # Add ports
        self.in_port: Port = self.add_input("In")
        self.mod_port: Port = self.add_input("Mod")
        self.out_port: Port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

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
        self.gain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("gain_db", self.gain_knob.get_value())
        )
        layout.addWidget(self.gain_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("gain_db", self.gain_knob)

        # Set control_knob for base class functionality
        self.control_knob = self.gain_knob

        # Track modulation state to detect changes
        self._was_modulated = False

        # Create initial unmodulated component
        self.component = self.create_unmodulated_component()

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Volume requires the In port to be connected."""
        return ["In"]

    # Implement abstract methods from ModulatedModuleBase
    def create_modulated_component(self, mod_comp):
        """Create ModulatedVolume with modulation. This is used when modulation is
        connected.

        Args:
            mod_comp: The modulation component

        Returns:
            ModulatedVolume instance
        """
        logger.info(f"VolumeModule: Creating ModulatedVolume with modulator {mod_comp}")

        # ModulatedVolume expects a generator as the modulation source
        # The modulation will control the gain_db parameter
        return ModulatedVolume(mod_comp, modulation_target="gain_db")

    def create_unmodulated_component(self):
        """Create simple Volume without modulation. This is used when no modulation is
        connected.

        Returns:
            Volume instance
        """
        gain_db = self.gain_knob.get_value()
        logger.debug(f"VolumeModule: Creating simple Volume with gain_db={gain_db}")
        return Volume(gain_db=gain_db)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Apply static or modulated gain for one render cycle."""
        if not self.in_port.is_connected:
            self.out_port.write(silence(num_samples))
            return

        input_signal = read_samples(self.in_port, num_samples)

        if self.mod_port.is_connected:
            gain = read_samples(self.mod_port, num_samples)
            gain_db = float_parameter(parameters, "gain_db", self.gain_knob.get_value)
            base_gain = 10.0 ** (gain_db / 20.0)
            self.out_port.write(input_signal * base_gain * gain)
            return

        if self.component is None or isinstance(self.component, ModulatedVolume):
            self.component = Volume(
                gain_db=float_parameter(parameters, "gain_db", self.gain_knob.get_value)
            )
        else:
            self.component.gain_db = float_parameter(
                parameters, "gain_db", self.gain_knob.get_value
            )

        self.out_port.write(self.component(input_signal))
