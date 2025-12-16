import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.constants import DEFAULT_GAIN_DB
from src.engine import ModulatedVolume, Volume
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.modules._modulated_base import ModulatedModuleBase
from src.gui.widgets import Knob

logger = logging.getLogger(__name__)


@register_module()
class VolumeModule(ModulatedModuleBase):
    """Volume/Gain module with modulation support."""

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
        self.in_port = self.add_input("In")
        self.mod_port = self.add_input("Mod")
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Gain in dB (linear mapping of dB values, since dB is already logarithmic)
        # Range: -60 dB (very quiet) to +12 dB (boost)
        # Default: -20 dB (safe for mixing)
        self.gain_knob = Knob("Gain (dB)", -60, 12, DEFAULT_GAIN_DB, logarithmic=False)
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

    def process(self, num_samples: int = 1):
        """Process audio through the volume control.

        Uses either Volume or ModulatedVolume component to process the audio.
        Automatically recreates component when modulation connection state changes.

        Args:
            num_samples: Number of samples to process
        """
        # Check if input is connected
        if not self.in_port.is_connected:
            self.out_port.write(0.0)
            return

        # Read input signal
        input_signal = self.in_port.read()
        if input_signal is None:
            self.out_port.write(0.0)
            return

        # Check if modulation connection state changed
        is_modulated = self.mod_port.is_connected

        if is_modulated != self._was_modulated:
            # Modulation state changed - recreate component
            logger.info(f"VolumeModule: Modulation state changed to {is_modulated}")

            if is_modulated:
                # Modulation just connected - create ModulatedVolume
                # Create a modulator that reads from the mod port
                class PortModulator:
                    """Simple modulator that reads from a port.

                    Implements the iterator protocol and get_samples() method
                    that ModulatedVolume expects.
                    """
                    def __init__(self, port):
                        self.port = port

                    def __iter__(self):
                        """Make this iterable."""
                        return self

                    def __next__(self):
                        """Iterator protocol - return next sample."""
                        value = self.port.read()
                        if value is None:
                            return 0.0
                        return float(value) if not hasattr(value, '__len__') else float(value[0])

                    def get_samples(self, n, reset=False, mode="vectorized"):
                        """Read samples from port (vectorized interface)."""
                        import numpy as np
                        value = self.port.read()
                        if value is None:
                            return np.zeros(n, dtype=np.float32)

                        # If value is already an array with correct size, return it
                        if isinstance(value, np.ndarray):
                            if len(value) == n:
                                return value.astype(np.float32)
                            elif len(value) > n:
                                return value[:n].astype(np.float32)
                            else:
                                # Pad with zeros
                                result = np.zeros(n, dtype=np.float32)
                                result[:len(value)] = value
                                return result
                        else:
                            # Scalar - repeat n times
                            return np.full(n, float(value), dtype=np.float32)

                mod_comp = PortModulator(self.mod_port)
                self.component = self.create_modulated_component(mod_comp)

                # Disable knob
                self.gain_knob.setEnabled(False)
                self.gain_knob.setStyleSheet("opacity: 0.5;")
                self.gain_knob.setToolTip("Gain controlled by Mod input")
            else:
                # Modulation disconnected - create simple Volume
                self.component = self.create_unmodulated_component()

                # Enable knob
                self.gain_knob.setEnabled(True)
                self.gain_knob.setStyleSheet("")
                self.gain_knob.setToolTip("Manual gain control")

            self._was_modulated = is_modulated

        # Safety check: component must exist
        if self.component is None:
            self.out_port.write(0.0)
            return

        # Update component parameters if not modulated
        if not isinstance(self.component, ModulatedVolume):
            # Simple Volume - update gain_db parameter
            gain_db = self.gain_knob.get_value()
            self.component.gain_db = gain_db

        # Process through the component (Volume or ModulatedVolume)
        output_signal = self.component(input_signal)

        # Write to output port
        self.out_port.write(output_signal)

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
