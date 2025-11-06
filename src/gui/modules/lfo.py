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


LFO_FREQUENCY_RANGE = (0.01, 20.0)  # LFO frequency range in Hz
LFO_DEFAULT_GAIN_DB = 0


@register_module()
class LFOModule(ModuleWidget):
    """LFO (Low Frequency Oscillator) module for modulation.

    Similar to Oscillator but optimized for modulation (0.01 Hz - 20 Hz).
    """

    metadata = ModuleMetadata(
        title="LFO",
        category=ModuleCategory.SOURCE,
        description="Low-frequency oscillator for modulation",
    )

    def __init__(self):
        """Initialize LFO module."""
        super().__init__(
            width=240,
            height=260,
            color=QColor(100, 140, 200),
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

        # Frequency control (knob) - optimized for LFO range
        knobs_layout = QHBoxLayout()
        self.freq_knob = Knob("Frequency (Hz)", LFO_FREQUENCY_RANGE[0], LFO_FREQUENCY_RANGE[1], 1.0)
        self.freq_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("frequency", self.freq_knob.get_value())
        )
        knobs_layout.addWidget(self.freq_knob)
        layout.addLayout(knobs_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        # Register parameters for automatic get/set
        self.register_parameter("frequency", self.freq_knob)

        # Create initial components
        self.component = self.create_engine_component()

    # AudioModuleInterface implementation
    @staticmethod
    def get_cv_output_range() -> tuple[float, float]:
        """LFO outputs bipolar signal [-1, 1].

        Returns:
            (-1.0, 1.0) - bipolar output range
        """
        return -1.0, 1.0

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
                freq, gain_db=LFO_DEFAULT_GAIN_DB, wave_range=(-1, 1)
            )

        if len(self.triangle_port.cables) > 0:
            components["Triangle"] = TriangleOscillator(
                freq, gain_db=LFO_DEFAULT_GAIN_DB, wave_range=(-1, 1)
            )

        if len(self.sawtooth_port.cables) > 0:
            components["Sawtooth"] = SawtoothOscillator(
                freq, gain_db=LFO_DEFAULT_GAIN_DB, wave_range=(-1, 1)
            )

        if len(self.square_port.cables) > 0:
            components["Square"] = SquareOscillator(
                freq, gain_db=LFO_DEFAULT_GAIN_DB, wave_range=(-1, 1)
            )

        if not components:
            # No outputs connected - return a silent component
            return None
        elif len(components) == 1:
            # Only one output connected - return that component directly
            return next(iter(components.values()))

        return WaveAdder(*components.values())
