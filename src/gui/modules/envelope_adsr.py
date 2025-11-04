from typing import Any

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from src.engine import ADSREnvelope
from src.engine.gate_triggered_adsr import GateTriggeredADSR
from src.gui.audio_module_interface import ModuleCategory, ModuleMetadata
from src.gui.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


@register_module()
class ADSRModule(ModuleWidget):
    """ADSR envelope module with optional gate input.

    Can be triggered by:
    - External gate signal (e.g., from MIDI Input)
    - Manual trigger button
    """

    metadata = ModuleMetadata(
        title="ADSR Envelope",
        category=ModuleCategory.MODULATED_SOURCE,  # Receives gate input
        description="ADSR envelope generator with gate input for MIDI triggering",
    )

    def __init__(self):
        """Initialize ADSR module."""
        super().__init__(
            width=220,
            height=220,
            color=QColor(120, 180, 80),
        )

        # Add input port for gate signal (optional)
        self.gate_input = self.add_input_port("Gate")

        # Add output port
        self.out_port = self.add_output_port("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # ADSR controls
        knobs_layout = QHBoxLayout()

        self.attack_knob = Knob("Attack", 0.005, 5.0, 0.01)  # Min 5ms, default 10ms
        self.attack_knob.setToolTip(
            "Attack time (seconds)\n"
            "Range: 0.005-5.0s\n"
            "Lower values may cause clicks"
        )
        self.attack_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("attack_duration", self.attack_knob.get_value())
        )
        knobs_layout.addWidget(self.attack_knob)

        self.decay_knob = Knob("Decay", 0.001, 5.0, 0.2)
        self.decay_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("decay_duration", self.decay_knob.get_value())
        )
        knobs_layout.addWidget(self.decay_knob)

        layout.addLayout(knobs_layout)

        knobs_layout2 = QHBoxLayout()

        self.sustain_knob = Knob("Sustain", 0.0, 1.0, 0.7)
        self.sustain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "sustain_level", self.sustain_knob.get_value()
            )
        )
        knobs_layout2.addWidget(self.sustain_knob)

        self.release_knob = Knob("Release", 0.001, 5.0, 0.3)
        self.release_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "release_duration", self.release_knob.get_value()
            )
        )
        knobs_layout2.addWidget(self.release_knob)

        layout.addLayout(knobs_layout2)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("attack_duration", self.attack_knob)
        self.register_parameter("decay_duration", self.decay_knob)
        self.register_parameter("sustain_level", self.sustain_knob)
        self.register_parameter("release_duration", self.release_knob)

        self.component = self.create_component()

    def get_required_inputs(self) -> list[str]:
        """Gate input is optional - ADSR works without gate triggering."""
        return []  # No required inputs - Gate is optional

    # AudioModuleInterface implementation
    def create_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the ADSR component.

        If a gate signal is connected, wraps the ADSR in a GateTriggeredADSR
        that automatically triggers on gate transitions.
        """
        # Create base ADSR envelope
        adsr = ADSREnvelope(
            attack_duration=self.attack_knob.get_value(),
            decay_duration=self.decay_knob.get_value(),
            sustain_level=self.sustain_knob.get_value(),
            release_duration=self.release_knob.get_value(),
        )

        # If gate input is connected, wrap with gate-triggered version
        if input_components and len(input_components) > 0:
            gate_source = input_components[0]
            return GateTriggeredADSR(adsr, gate_source)

        # No gate input, return plain ADSR (can be manually triggered)
        return adsr
