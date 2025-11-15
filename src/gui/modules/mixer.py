import logging
from typing import Any

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from src.engine import Chain, Volume, WaveAdder
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL_VOLUME = 0.0  # Default no gain for newly connected channels


@register_module()
class MixerModule(ModuleWidget):
    """Mixer module for combining multiple audio signals.

    Uses WaveAdder to mix multiple inputs together (averages them).
    """

    metadata = ModuleMetadata(
        title="Mixer",
        category=ModuleCategory.MIXER,
        description="4-channel audio mixer",
    )

    def __init__(self):
        """Initialize mixer module."""
        super().__init__(width=220, height=280, color=QColor(100, 150, 100))

        # Add multiple input ports
        self.in1_port = self.add_input_port("In 1")
        self.in2_port = self.add_input_port("In 2")
        self.in3_port = self.add_input_port("In 3")
        self.in4_port = self.add_input_port("In 4")

        # Add output port
        self.out_port = self.add_output_port("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # First row of knobs (Ch 1 & 2)
        knobs_row1 = QHBoxLayout()

        self.gain1_knob = Knob(
            "Ch 1", 0.0, 1.0, DEFAULT_CHANNEL_VOLUME, logarithmic=False
        )
        self.gain1_knob.value_changed.connect(lambda: self._on_gain_changed(0))
        knobs_row1.addWidget(self.gain1_knob)

        self.gain2_knob = Knob(
            "Ch 2", 0.0, 1.0, DEFAULT_CHANNEL_VOLUME, logarithmic=False
        )
        self.gain2_knob.value_changed.connect(lambda: self._on_gain_changed(1))
        knobs_row1.addWidget(self.gain2_knob)

        layout.addLayout(knobs_row1)

        # Second row of knobs (Ch 3 & 4)
        knobs_row2 = QHBoxLayout()

        self.gain3_knob = Knob(
            "Ch 3", 0.0, 1.0, DEFAULT_CHANNEL_VOLUME, logarithmic=False
        )
        self.gain3_knob.value_changed.connect(lambda: self._on_gain_changed(2))
        knobs_row2.addWidget(self.gain3_knob)

        self.gain4_knob = Knob(
            "Ch 4", 0.0, 1.0, DEFAULT_CHANNEL_VOLUME, logarithmic=False
        )
        self.gain4_knob.value_changed.connect(lambda: self._on_gain_changed(3))
        knobs_row2.addWidget(self.gain4_knob)

        layout.addLayout(knobs_row2)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("gain1", self.gain1_knob)
        self.register_parameter("gain2", self.gain2_knob)
        self.register_parameter("gain3", self.gain3_knob)
        self.register_parameter("gain4", self.gain4_knob)

        # Track individual Volume components for each channel (for hot-swapping)
        self._volume_components = [None, None, None, None]

        self.component = self.create_engine_component()

    def _on_gain_changed(self, channel_index: int):
        """Handle gain knob changes using hot-swapping (no recompile).

        Args:
            channel_index: Index of the channel (0-3)
        """
        gain_knobs = [
            self.gain1_knob,
            self.gain2_knob,
            self.gain3_knob,
            self.gain4_knob,
        ]

        # Get the input port for this channel
        input_ports = [self.in1_port, self.in2_port, self.in3_port, self.in4_port]

        new_gain = gain_knobs[channel_index].get_value()
        is_connected = len(input_ports[channel_index].cables) > 0
        has_volume_comp = self._volume_components[channel_index] is not None

        logger.debug(
            f"🎚️ MIXER: Ch {channel_index + 1} gain knob changed to {new_gain:.3f} "
            f"(connected={is_connected}, has_comp={has_volume_comp})"
        )

        # Only hot-swap if the input is connected and Volume component exists
        if is_connected and has_volume_comp:
            # Hot-swap: update amplitude directly on the Volume component
            try:
                vol_comp = self._volume_components[channel_index]
                old_amp = vol_comp.amplitude
                vol_comp.amplitude = new_gain
                logger.debug(
                    f"✓ Hot-swapped Ch {channel_index + 1} Volume: {old_amp:.3f} → "
                    f"{new_gain:.3f} (Volume obj id={id(vol_comp)})"
                )
            except (AttributeError, ValueError) as e:
                logger.warning(
                    f"❌ Failed to hotswap gain for Ch {channel_index + 1}: {e}"
                )
        else:
            logger.debug(
                f"⏭️ Skipping hot-swap for Ch {channel_index + 1}: "
                f"not connected or no Volume component"
            )

    # AudioModuleInterface implementation
    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the mixer component with per-channel gain control.

        Applies a Volume modifier to each input channel before mixing them together.
        Only creates Volume components for channels with connected inputs.
        Stores Volume references for hot-swapping to prevent clicks.
        """
        if input_components and len(input_components) > 0:
            # Get input ports and knobs
            input_ports = [self.in1_port, self.in2_port, self.in3_port, self.in4_port]
            gain_knobs = [
                self.gain1_knob,
                self.gain2_knob,
                self.gain3_knob,
                self.gain4_knob,
            ]

            # Log all knob values for debugging
            all_gains = [k.get_value() for k in gain_knobs]
            logger.debug(
                f"Mixer knob values at compilation: Ch1={all_gains[0]:.3f}, "
                f"Ch2={all_gains[1]:.3f}, Ch3={all_gains[2]:.3f}, "
                f"Ch4={all_gains[3]:.3f}"
            )

            # Map each input_component to its corresponding port index
            # The patch compiler provides components in the order of connected ports
            port_indices = []
            for port_idx, port in enumerate(input_ports):
                if len(port.cables) > 0:
                    port_indices.append(port_idx)
                    logger.debug(
                        f"  Port {port_idx} ({port.port_name}) has "
                        f"{len(port.cables)} cable(s)"
                    )

            logger.debug(
                f"Mixer creating components: {len(input_components)} inputs, "
                f"connected ports (0-indexed): {port_indices} = "
                f"{[p+1 for p in port_indices]} (1-indexed)"
            )

            # Reset all volume components
            self._volume_components = [None, None, None, None]

            # Apply volume to each available input
            processed_inputs = []
            for comp_idx, input_comp in enumerate(input_components):
                # Get the actual port index for this component
                if comp_idx < len(port_indices):
                    port_idx = port_indices[comp_idx]

                    # Get the corresponding gain knob
                    gain = gain_knobs[port_idx].get_value()

                    # Create Volume component and store reference for hot-swapping
                    volume_comp = Volume(amplitude=gain)
                    self._volume_components[port_idx] = volume_comp

                    # Chain input with volume control
                    processed_inputs.append(Chain(input_comp, volume_comp))

                    logger.info(
                        f"  Ch {port_idx + 1}: Created Volume with gain={gain:.3f} "
                        f"(Volume obj id={id(volume_comp)})"
                    )
                else:
                    # Shouldn't happen, but handle gracefully
                    processed_inputs.append(input_comp)
                    logger.warning(
                        f"  Comp {comp_idx}: No port mapping, passing through"
                    )

            # Use mix_mode='sum' for standard mixer behavior (maintains volume levels)
            logger.info(
                f"Creating WaveAdder with mix_mode='sum' for {len(processed_inputs)} "
                f"inputs"
            )
            return WaveAdder(*processed_inputs, mix_mode="sum")

        return None
