import logging

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from src.engine import Chain, Volume, WaveAdder
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.runtime import RuntimeParameters
from src.gui.runtime_helpers import read_samples, silence
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL_VOLUME = 0.7  # Default 70% gain for newly connected channels


@register_module()
class MixerModule(ModuleWidget):
    """Mixer module for combining multiple audio signals.

    Uses WaveAdder to mix multiple inputs together (averages them).
    """

    runtime_kind = "mixer"
    runtime_input_names = ("In 1", "In 2", "In 3", "In 4")
    runtime_output_names = ("Out",)
    runtime_parameter_names = ("gain1", "gain2", "gain3", "gain4")

    metadata = ModuleMetadata(
        title="Mixer",
        category=ModuleCategory.MIXER,
        description="4-channel audio mixer",
    )

    def __init__(self):
        """Initialize mixer module."""
        super().__init__(width=220, height=260, color=QColor(100, 150, 100))

        # Add multiple input ports
        self.in1_port = self.add_input("In 1")
        self.in2_port = self.add_input("In 2")
        self.in3_port = self.add_input("In 3")
        self.in4_port = self.add_input("In 4")

        # Add output port
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # First row of knobs (Ch 1 & 2)
        self.gain1_knob = Knob(
            "Ch 1", 0.0, 1.0, DEFAULT_CHANNEL_VOLUME, logarithmic=False
        )
        self.gain1_knob.value_changed.connect(lambda: self._on_gain_changed(0))

        self.gain2_knob = Knob(
            "Ch 2", 0.0, 1.0, DEFAULT_CHANNEL_VOLUME, logarithmic=False
        )
        self.gain2_knob.value_changed.connect(lambda: self._on_gain_changed(1))

        knobs_row1 = QHBoxLayout()
        knobs_row1.addWidget(self.gain1_knob)
        knobs_row1.addWidget(self.gain2_knob)
        layout.addLayout(knobs_row1)

        self.gain3_knob = Knob(
            "Ch 3", 0.0, 1.0, DEFAULT_CHANNEL_VOLUME, logarithmic=False
        )
        self.gain3_knob.value_changed.connect(lambda: self._on_gain_changed(2))

        self.gain4_knob = Knob(
            "Ch 4", 0.0, 1.0, DEFAULT_CHANNEL_VOLUME, logarithmic=False
        )
        self.gain4_knob.value_changed.connect(lambda: self._on_gain_changed(3))

        self.gain_knobs = [
            self.gain1_knob,
            self.gain2_knob,
            self.gain3_knob,
            self.gain4_knob,
        ]

        # Second row of knobs (Ch 3 & 4)
        knobs_row2 = QHBoxLayout()
        knobs_row2.addWidget(self.gain3_knob)
        knobs_row2.addWidget(self.gain4_knob)
        layout.addLayout(knobs_row2)
        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("gain1", self.gain1_knob)
        self.register_parameter("gain2", self.gain2_knob)
        self.register_parameter("gain3", self.gain3_knob)
        self.register_parameter("gain4", self.gain4_knob)

        # Create Volume components for each channel (for gain control)
        self._volume_components = [
            Volume(amplitude=DEFAULT_CHANNEL_VOLUME),
            Volume(amplitude=DEFAULT_CHANNEL_VOLUME),
            Volume(amplitude=DEFAULT_CHANNEL_VOLUME),
            Volume(amplitude=DEFAULT_CHANNEL_VOLUME),
        ]

        logger.debug("Mixer initialized with 4 channels")

    def _on_gain_changed(self, channel_index: int):
        """Handle gain knob changes by updating Volume component amplitude.

        Args:
            channel_index: Index of the channel (0-3)
        """
        if channel_index < 0 or channel_index >= len(self.gain_knobs):
            logger.error(f"Invalid channel index {channel_index} for gain change")
            return

        new_gain = self.gain_knobs[channel_index].get_value()

        # Update the Volume component amplitude (click-free)
        self._volume_components[channel_index].amplitude = new_gain

        logger.debug(f"ðŸŽšï¸ Mixer: Ch {channel_index + 1} gain set to {new_gain:.3f}")

    def create_engine_component(
        self,
        input_components=None,
        modulation_components=None,
    ):
        """Create a mixer component for the patch compiler.

        Each input is wrapped with its corresponding channel gain, then all active
        channels are summed together.
        """
        if not input_components:
            return None

        processed_inputs = []
        for channel_idx, input_component in enumerate(input_components):
            if input_component is None:
                continue

            gain = DEFAULT_CHANNEL_VOLUME
            if channel_idx < len(self.gain_knobs):
                gain = self.gain_knobs[channel_idx].get_value()

            processed_inputs.append(Chain(input_component, Volume(amplitude=gain)))

        if not processed_inputs:
            return None

        return WaveAdder(*processed_inputs, mix_mode="sum")

    def process_runtime(
        self, num_samples: int, parameters: RuntimeParameters
    ) -> None:
        """Mix connected input channels for the current engine cycle."""
        del parameters
        mixed_signal = None
        input_ports = [self.in1_port, self.in2_port, self.in3_port, self.in4_port]

        for channel_idx, port in enumerate(input_ports):
            if not port.is_connected:
                continue
            signal = read_samples(port, num_samples)
            gained_signal = self._volume_components[channel_idx](signal)
            mixed_signal = (
                gained_signal
                if mixed_signal is None
                else mixed_signal + gained_signal
            )

        self.out_port.write(
            mixed_signal if mixed_signal is not None else silence(num_samples)
        )

