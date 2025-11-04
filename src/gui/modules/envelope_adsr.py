from typing import Any

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from src.engine import ADSREnvelope
from src.gui.audio_module_interface import ModuleCategory, ModuleMetadata
from src.gui.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget


@register_module()
class ADSRModule(ModuleWidget):
    """ADSR envelope module."""

    metadata = ModuleMetadata(
        title="ADSR Envelope",
        category=ModuleCategory.SOURCE,
        description="ADSR envelope generator for modulation",
    )

    def __init__(self):
        """Initialize ADSR module."""
        super().__init__(
            width=220,
            height=200,
            color=QColor(120, 180, 80),
        )

        # Add output port
        self.out_port = self.add_output_port("Out")

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # ADSR controls
        knobs_layout = QHBoxLayout()

        self.attack_knob = Knob("Attack", 0.001, 5.0, 0.1)
        self.attack_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("attack", self.attack_knob.get_value())
        )
        knobs_layout.addWidget(self.attack_knob)

        self.decay_knob = Knob("Decay", 0.001, 5.0, 0.2)
        self.decay_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("decay", self.decay_knob.get_value())
        )
        knobs_layout.addWidget(self.decay_knob)

        layout.addLayout(knobs_layout)

        knobs_layout2 = QHBoxLayout()

        self.sustain_knob = Knob("Sustain", 0.0, 1.0, 0.7)
        self.sustain_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "sustain", self.sustain_knob.get_value()
            )
        )
        knobs_layout2.addWidget(self.sustain_knob)

        self.release_knob = Knob("Release", 0.001, 5.0, 0.3)
        self.release_knob.value_changed.connect(
            lambda: self.parameter_changed.emit(
                "release", self.release_knob.get_value()
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

    # AudioModuleInterface implementation
    def create_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ):
        """Create the ADSR component."""
        return ADSREnvelope(
            attack_duration=self.attack_knob.get_value(),
            decay_duration=self.decay_knob.get_value(),
            sustain_level=self.sustain_knob.get_value(),
            release_duration=self.release_knob.get_value(),
        )
