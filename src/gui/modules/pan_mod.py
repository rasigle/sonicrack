import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.engine import ModulatedPanner, Panner
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.modules._modulated_base import ModulatedModuleBase
from src.gui.widgets import Knob

logger = logging.getLogger(__name__)


@register_module()
class PannerModule(ModulatedModuleBase):
    """Panner module for stereo positioning with modulation support."""

    metadata = ModuleMetadata(
        title="Panner (Mod)",
        category=ModuleCategory.MODIFIER,
        description="Stereo panner with modulation input",
    )

    def __init__(self):
        """Initialize panner module."""
        super().__init__(
            width=140,
            height=180,
            color=QColor(180, 80, 180),
        )

        # Add ports
        self.in_port = self.add_input("In")
        self.mod_port = self.add_input("Mod")
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

        # Set control_knob for base class functionality
        self.control_knob = self.pan_knob

    # AudioModuleInterface implementation
    def get_required_inputs(self) -> list[str]:
        """Panner requires the In port to be connected."""
        return ["In"]

    def process(self, num_samples: int = 1):
        """Process audio through the panner.

        Args:
            num_samples: Number of samples to process
        """
        if self.in_port.is_connected:
            samples = self.in_port.read()
            if samples is not None:
                # Apply panning (simplified for process-based flow)
                import numpy as np
                position = self.pan_knob.get_value()

                # Convert to stereo if mono
                if samples.ndim == 1:
                    samples = np.column_stack([samples, samples])

                # Apply pan law
                left_gain = np.sqrt(0.5 * (1.0 - position))
                right_gain = np.sqrt(0.5 * (1.0 + position))

                panned = samples.copy()
                panned[:, 0] *= left_gain
                panned[:, 1] *= right_gain

                self.out_port.write(panned)

    def get_cv_range(self, port_name: str = "Mod") -> tuple[float, float]:
        """Panner expects bipolar CV range [-1, 1] for pan position.

        Returns:
            (-1.0, 1.0) - bipolar range for pan control
        """
        return -1.0, 1.0

    # Implement abstract methods from ModulatedModuleBase
    def create_modulated_component(self, mod_comp):
        """Create ModulatedPanner with modulation."""
        return ModulatedPanner(mod_comp)

    def create_unmodulated_component(self):
        """Create simple Panner without modulation."""
        position = self.pan_knob.get_value()
        return Panner(position)
