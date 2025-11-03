from typing import Dict, Any, List, Optional

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGraphicsProxyWidget

from src.engine import ADSREnvelope
from src.gui.widgets.module_widget import ModuleWidget
from src.gui.widgets import Knob
from src.gui.audio_module_interface import ModuleCategory

TITLE = "ADSR Envelope"


class ADSRModule(ModuleWidget):
    """ADSR envelope module."""

    def __init__(self):
        """Initialize ADSR module."""
        super().__init__(
            TITLE,
            width=220,
            height=200,
            color=QColor(120, 180, 80),
        )

        # Add output port
        self.out_port = self.add_output_port("Out")

        # Create control widget
        self.controls_widget = QWidget()
        self.controls_widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

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

        # Add controls as proxy widget - position below title bar
        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(self.controls_widget)
        self.proxy.setPos(0, 42)

        self.component = self.create_component()

    # AudioModuleInterface implementation
    @property
    def module_category(self) -> ModuleCategory:
        """Return SOURCE since envelopes generate control signals."""
        return ModuleCategory.SOURCE

    def create_component(
        self,
        input_components: Optional[list[Any]] = None,
        modulation_components: Optional[dict[str, Any]] = None,
    ):
        """Create the ADSR component."""
        return ADSREnvelope(
            attack_duration=self.attack_knob.get_value(),
            decay_duration=self.decay_knob.get_value(),
            sustain_level=self.sustain_knob.get_value(),
            release_duration=self.release_knob.get_value(),
        )

    def get_parameters(self) -> dict[str, Any]:
        """Get current parameters."""
        return {
            "attack_duration": self.attack_knob.get_value(),
            "decay_duration": self.decay_knob.get_value(),
            "sustain_level": self.sustain_knob.get_value(),
            "release_duration": self.release_knob.get_value(),
        }

    def set_parameters(self, params: dict[str, Any]):
        """Set parameters from dictionary."""
        if "attack_duration" in params:
            self.attack_knob.set_value(params["attack_duration"])
        if "decay_duration" in params:
            self.decay_knob.set_value(params["decay_duration"])
        if "sustain_level" in params:
            self.sustain_knob.set_value(params["sustain_level"])
        if "release_duration" in params:
            self.release_knob.set_value(params["release_duration"])
