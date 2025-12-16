from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import Panner
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.core.module_registry import register_module


@register_module()
class SimplePannerModule(ModuleWidget):
    """Simple panner module without modulation input."""

    metadata = ModuleMetadata(
        title="Panner",
        category=ModuleCategory.MODIFIER,
        description="Simple stereo panner without modulation input",
    )

    def __init__(self):
        """Initialize simple panner module."""
        super().__init__(
            width=140,
            height=150,
            color=QColor(160, 60, 160),
        )

        # Add ports (no modulation input)
        self.in_port = self.add_input("In")
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Pan knob
        self.pan_knob = Knob("Pan", -1.0, 1.0, 0.0)
        self.pan_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("position", self.pan_knob.get_value())
        )
        layout.addWidget(self.pan_knob, alignment=Qt.AlignmentFlag.AlignCenter)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("position", self.pan_knob)

        self.component = self.create_engine_component()

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Panner requires the In port to be connected."""
        return ["In"]

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the panner component."""
        pan = self.pan_knob.get_value()
        return Panner(pan)

    def process(self, num_samples: int = 1):
        """Apply stereo panning to input signal.

        Reads from the input port, applies panning control, and writes to the output port.

        Args:
            num_samples: Number of samples to process (default: 1 for per-sample processing)

        Note:
            In the current architecture, this method is not actively called during playback.
            The audio engine directly calls get_samples() on the compiled AudioComponents.
            This method exists to satisfy the AudioModule interface.
        """
        if not self.in_port.is_connected:
            return

        # Read input signal
        input_signal = self.in_port.read()

        # Apply panning (simple implementation)
        import numpy as np
        pan = self.pan_knob.get_value()  # -1 (left) to +1 (right)

        # Convert pan position to left/right gains
        left_gain = np.sqrt(0.5 * (1 - pan))
        right_gain = np.sqrt(0.5 * (1 + pan))

        # Create stereo output
        if isinstance(input_signal, np.ndarray):
            if len(input_signal.shape) == 1:  # Mono input
                output_signal = np.stack([input_signal * left_gain, input_signal * right_gain], axis=-1)
            else:  # Already stereo
                output_signal = input_signal
        else:  # Scalar
            output_signal = np.array([input_signal * left_gain, input_signal * right_gain])

        # Write to output port
        self.out_port.write(output_signal)

