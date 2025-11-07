from typing import Any
import logging

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from src.engine import Volume
from src.engine.modifier import ModulatedVolume
from src.gui.audio_module_interface import ModuleCategory, ModuleMetadata
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.module_registry import register_module

logger = logging.getLogger(__name__)


@register_module()
class VCAModule(ModuleWidget):
    """Voltage-Controlled Amplifier (VCA) with amplitude modulation.

    A VCA controls the amplitude/volume of an input signal. It can be controlled
    manually with the amplitude knob, or via a control voltage (CV) input.

    When CV is connected, the knob is disabled and amplitude is controlled by
    the CV signal (typically an envelope or LFO).

    Inputs:
        - In: Audio input signal
        - CV: Control voltage input (0.0 to 1.0) for amplitude modulation

    Outputs:
        - Out: Amplitude-controlled audio output

    Parameters:
        - Amplitude: Manual amplitude control (0.0 to 1.0)
    """

    metadata = ModuleMetadata(
        title="VCA",
        category=ModuleCategory.MODIFIER,
        description="Voltage-Controlled Amplifier with CV modulation",
        version="1.0.0",
        author="AudioPlayground",
    )

    def __init__(self):
        """Initialize VCA module."""
        super().__init__(
            width=180,
            height=180,
            color=QColor(100, 140, 180),
        )

        # Add ports
        self.in_port = self.add_input_port("In")
        self.cv_port = self.add_input_port("CV")
        self.out_port = self.add_output_port("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Amplitude control
        knobs_layout = QHBoxLayout()
        self.amp_knob = Knob("Amplitude", 0.0, 1.0, 0.7, logarithmic=False)
        self.amp_knob.setToolTip("Amplitude/Volume control (0.0 to 1.0)")
        self.amp_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("amplitude", self.amp_knob.get_value())
        )
        knobs_layout.addWidget(self.amp_knob)
        layout.addLayout(knobs_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("amplitude", self.amp_knob)

        self.component = self.create_engine_component()

    def get_required_inputs(self) -> list[str]:
        """VCA requires audio input on 'In' port."""
        return ["In"]

    def get_modulation_inputs(self) -> list[str]:
        """VCA accepts modulation on CV port."""
        return ["CV"]

    def get_cv_range(self, port_name: str = "CV") -> tuple[float, float]:
        """VCA CV port expects unipolar range [0, 1] for amplitude modulation.

        Returns:
            (0.0, 1.0) - unipolar range (0 = silence, 1 = full amplitude)
        """
        return 0.0, 1.0

    def update_knob_state(self):
        """Update amplitude knob enabled state based on CV port connection.

        When CV is connected, the knob is disabled to show that amplitude
        is controlled externally.
        """
        # Check CV port connection
        cv_port = None
        for port in self.input_ports:
            if port.port_name == "CV":
                cv_port = port
                break

        has_cv = cv_port and len(cv_port.cables) > 0

        logger.info(
            f"VCA update_knob_state: CV port has "
            f"{len(cv_port.cables) if cv_port else 0} cables, has_cv={has_cv}"
        )

        if has_cv:
            # Amplitude controlled by CV - disable knob
            self.amp_knob.setEnabled(False)
            self.amp_knob.setStyleSheet("opacity: 0.5;")
            self.amp_knob.setToolTip("Amplitude controlled by CV input")
            logger.info("VCA: Amplitude knob DISABLED (CV connected)")
        else:
            # No CV - enable knob for manual control
            self.amp_knob.setEnabled(True)
            self.amp_knob.setStyleSheet("")
            self.amp_knob.setToolTip("Manual amplitude control (0.0 to 1.0)")
            logger.info("VCA: Amplitude knob ENABLED (no CV)")

    def create_engine_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the VCA component.

        If CV is connected, creates a ModulatedVolume that responds to CV input.
        Otherwise, creates a simple Volume with manual amplitude control.

        Args:
            input_components: Audio input signal (required)
            modulation_components: CV modulation input (optional)

        Returns:
            Volume or ModulatedVolume component (patch compiler wraps in Chain)
        """
        amplitude = self.amp_knob.get_value()

        # Check for CV modulation
        cv_modulator = (
            modulation_components.get("CV")
            if modulation_components and isinstance(modulation_components, dict)
            else None
        )
        has_cv = cv_modulator is not None

        logger.info(
            f"VCA: Creating component with amplitude={amplitude:.3f}, has_cv={has_cv}, "
            f"modulation_components={modulation_components}"
        )

        # Update UI state
        if has_cv:
            self.amp_knob.setEnabled(False)
            self.amp_knob.setStyleSheet("opacity: 0.5;")
            self.amp_knob.setToolTip("Amplitude controlled by CV input")
            logger.info(
                "VCA create_engine_component: Amplitude knob DISABLED (CV connected)"
            )

            # Create ModulatedVolume with CV control
            # CV input directly controls amplitude (should be in [0, 1] range)
            # Patch compiler will wrap: Chain(input_signal, ModulatedVolume)
            modulated_component = ModulatedVolume(
                cv_modulator, modulation_target="amplitude"
            )
            logger.info(
                f"VCA: Returning ModulatedVolume component: {modulated_component}, "
                f"type={type(modulated_component).__name__}"
            )
            return modulated_component

        # Static volume component (no CV)
        self.amp_knob.setEnabled(True)
        self.amp_knob.setStyleSheet("")
        self.amp_knob.setToolTip("Manual amplitude control (0.0 to 1.0)")
        logger.info("VCA create_engine_component: Amplitude knob ENABLED (no CV)")

        # Create simple Volume with manual control
        # Patch compiler will wrap: Chain(input_signal, Volume)
        volume_component = Volume(amplitude=amplitude)
        logger.info(
            f"VCA: Returning Volume component: {volume_component}, "
            f"type={type(volume_component).__name__}"
        )
        return volume_component
