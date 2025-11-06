from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from src.engine.composer import WaveAdder
from src.engine.oscillator import (
    SineOscillator,
    SquareOscillator,
    SawtoothOscillator,
    TriangleOscillator,
)
from src.gui.audio_module_interface import ModuleCategory, ModuleMetadata
from src.gui.module_registry import register_module
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget

if TYPE_CHECKING:
    from src.engine.audio_component import AudioComponent

OSCILLATOR_DEFAULT_GAIN_DB = 0.0


@register_module()
class OscillatorModule(ModuleWidget):
    """Oscillator module with frequency and gain controls."""

    metadata = ModuleMetadata(
        title="Oscillator",
        category=ModuleCategory.SOURCE,
        description="Multi-waveform oscillator with frequency and gain controls",
    )

    def __init__(self):
        """Initialize oscillator module."""
        super().__init__(
            width=220,
            height=200,
            color=QColor(80, 120, 200),
        )

        # Add four output ports - one for each waveform
        self.sine_port = self.add_output_port("Sine")
        self.triangle_port = self.add_output_port("Triangle")
        self.sawtooth_port = self.add_output_port("Sawtooth")
        self.square_port = self.add_output_port("Square")

        # Map port names to port objects for easy lookup
        self.port_map = {
            "Sine": self.sine_port,
            "Triangle": self.triangle_port,
            "Sawtooth": self.sawtooth_port,
            "Square": self.square_port,
        }

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Frequency control (knobs)
        knobs_layout = QHBoxLayout()
        self.freq_knob = Knob("Freq (Hz)", 20, 2000, 440)
        self.freq_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("frequency", self.freq_knob.get_value())
        )
        knobs_layout.addWidget(self.freq_knob)
        layout.addLayout(knobs_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.register_parameter("frequency", self.freq_knob)

        self.component = self.create_engine_component()

    # AudioModuleInterface implementation
    def _on_wave_changed(self, wave_type: str):
        """Handle waveform type change."""
        self.create_engine_component()
        self.parameter_changed.emit("waveform", wave_type)

    def create_engine_component(
        self,
        input_components: list[AudioComponent] | None = None,
        modulation_components: dict[str, AudioComponent] | None = None,
    ) -> AudioComponent | None:
        """Create the LFO components for all connected outputs.

        Returns:
            Dictionary mapping port names to their oscillator components
        """
        freq = self.freq_knob.get_value()
        components = {}

        # Create oscillators only for connected outputs
        if len(self.sine_port.cables) > 0:
            components["Sine"] = SineOscillator(
                freq, gain_db=OSCILLATOR_DEFAULT_GAIN_DB
            )

        if len(self.triangle_port.cables) > 0:
            components["Triangle"] = TriangleOscillator(
                freq, gain_db=OSCILLATOR_DEFAULT_GAIN_DB
            )

        if len(self.sawtooth_port.cables) > 0:
            components["Sawtooth"] = SawtoothOscillator(
                freq, gain_db=OSCILLATOR_DEFAULT_GAIN_DB
            )

        if len(self.square_port.cables) > 0:
            components["Square"] = SquareOscillator(
                freq, gain_db=OSCILLATOR_DEFAULT_GAIN_DB
            )

        if not components:
            # No outputs connected - return a silent component
            return None
        elif len(components) == 1:
            # Only one output connected - return that component directly
            return next(iter(components.values()))

        return WaveAdder(*components.values())
