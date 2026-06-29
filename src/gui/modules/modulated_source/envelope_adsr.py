import logging
from typing import Any

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout, QPushButton

from src.engine.modulator import ADSREnvelope, GateTriggeredADSR
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.runtime import RuntimeParameters
from src.gui.core.runtime_helpers import float_parameter
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

    runtime_kind = "adsr"

    metadata = ModuleMetadata(
        title="ADSR Envelope",
        category=ModuleCategory.MODULATED_SOURCE,  # Receives gate input
        description="ADSR envelope generator with gate input for MIDI triggering",
    )

    def __init__(self):
        """Initialize ADSR module."""
        super().__init__(
            width=220,
            height=275,
            color=QColor(120, 180, 80),
        )

        # Add input port for gate signal (optional)
        self.gate_input = self.add_input("Gate")

        # Add output port
        self.out_port = self.add_output("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # ADSR controls
        knobs_layout = QHBoxLayout()

        self.attack_knob = Knob("Attack", 0.005, 5.0, 0.01)  # Min 5ms, default 10ms
        self.attack_knob.setToolTip(
            "Attack time (seconds)\nRange: 0.005-5.0s\nLower values may cause clicks"
        )
        self.attack_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "attack_duration", self.attack_knob.get_value()
            )
        )
        knobs_layout.addWidget(self.attack_knob)

        self.decay_knob = Knob("Decay", 0.001, 5.0, 0.2)
        self.decay_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "decay_duration", self.decay_knob.get_value()
            )
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

        # Manual trigger button
        trigger_layout = QHBoxLayout()
        self.trigger_button = QPushButton("Trigger")
        self.trigger_button.setCheckable(False)  # Not a toggle, just a momentary push
        self.trigger_button.setMinimumHeight(35)
        self.trigger_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: 2px solid #45a049;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:pressed {
                background-color: #45a049;
                border: 2px solid #3d8b40;
            }
            QPushButton:hover {
                background-color: #5cbf60;
            }
        """)
        self.trigger_button.setToolTip(
            "Manual Trigger\n"
            "Press: Start attack phase\n"
            "Hold: Sustain phase\n"
            "Release: Trigger release phase"
        )
        # Connect press and release events
        self.trigger_button.pressed.connect(self._on_trigger_pressed)
        self.trigger_button.released.connect(self._on_trigger_released)
        trigger_layout.addWidget(self.trigger_button)
        layout.addLayout(trigger_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("attack_duration", self.attack_knob)
        self.register_parameter("decay_duration", self.decay_knob)
        self.register_parameter("sustain_level", self.sustain_knob)
        self.register_parameter("release_duration", self.release_knob)

        # Track ADSR component for manual triggering
        self._adsr_component: ADSREnvelope | GateTriggeredADSR | None = None

        self.component = self.create_engine_component()

    def _trigger_adsr(self, note_on: bool) -> None:
        """Trigger ADSR note on/off if available."""
        if self._adsr_component is None:
            return

        component = self._adsr_component
        adsr = component.adsr if isinstance(component, GateTriggeredADSR) else component

        if note_on:
            adsr.trigger_note_on()
            logging.debug("ADSR manually triggered (note on)")
        else:
            adsr.trigger_note_off()
            logging.debug("ADSR manually released (note off)")

    def _on_trigger_pressed(self) -> None:
        """Handle trigger button press - start attack phase."""
        self._trigger_adsr(note_on=True)

    def _on_trigger_released(self) -> None:
        """Handle trigger button release - start release phase."""
        self._trigger_adsr(note_on=False)

    def get_required_inputs(self) -> list[str]:
        """Gate input is optional - ADSR works without gate triggering."""
        return []  # No required inputs - Gate is optional

    def get_cv_output_range(self) -> tuple[float, float]:
        """ADSR envelope outputs unipolar signal [0, 1].

        Returns:
            (0.0, 1.0) - unipolar output range
        """
        return 0.0, 1.0

    # AudioModuleInterface implementation
    def create_engine_component(
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
            self._adsr_component = GateTriggeredADSR(adsr, gate_source)
        else:
            # No gate input, return plain ADSR (can be manually triggered)
            self._adsr_component = adsr

        return self._adsr_component

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Render the ADSR envelope for the current engine cycle."""
        if self._adsr_component is None:
            self._adsr_component = self.create_engine_component()

        self._adsr_component.attack_duration = float_parameter(
            parameters, "attack_duration", self.attack_knob.get_value
        )
        self._adsr_component.decay_duration = float_parameter(
            parameters, "decay_duration", self.decay_knob.get_value
        )
        self._adsr_component.sustain_level = float_parameter(
            parameters, "sustain_level", self.sustain_knob.get_value
        )
        self._adsr_component.release_duration = float_parameter(
            parameters, "release_duration", self.release_knob.get_value
        )
        self.out_port.write(self._adsr_component.get_samples(num_samples))
