from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from src.engine.composer import WaveAdder
from src.engine.oscillator import (
    SineOscillator,
    SawtoothOscillator,
    TriangleOscillator,
)
from engine import SquareOscillator
from src.gui.audio_module_interface import ModuleCategory, ModuleMetadata
from src.gui.module_registry import register_module
from src.gui.ui_constants import (
    MIN_PW_PERCENTAGE_VALUE,
    MAX_PW_PERCENTAGE_VALUE,
    DEFAULT_PW_PERCENTAGE_VALUE,
)
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget

if TYPE_CHECKING:
    from src.engine.audio_component import AudioComponent

OSCILLATOR_DEFAULT_GAIN_DB = 0.0
OSCILLATOR_DEFAULT_FREQUENCY = 120
OSCILLATOR_MIN_FREQUENCY = 20
OSCILLATOR_MAX_FREQUENCY = 2000


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
            height=240,
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
        self.freq_knob = Knob(
            "Freq (Hz)",
            OSCILLATOR_MIN_FREQUENCY,
            OSCILLATOR_MAX_FREQUENCY,
            OSCILLATOR_DEFAULT_FREQUENCY,
        )
        self.freq_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("frequency", self.freq_knob.get_value())
        )
        knobs_layout.addWidget(self.freq_knob)
        layout.addLayout(knobs_layout)

        # Pulse width control (for square wave)
        pw_layout = QHBoxLayout()
        self.pulsewidth_knob = Knob(
            "PW",
            MIN_PW_PERCENTAGE_VALUE / 100,
            MAX_PW_PERCENTAGE_VALUE / 100,
            DEFAULT_PW_PERCENTAGE_VALUE / 100,
        )
        self.pulsewidth_knob.value_changed.connect(self._on_pulsewidth_changed)
        pw_layout.addWidget(self.pulsewidth_knob)
        layout.addLayout(pw_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("pulsewidth", self.pulsewidth_knob)

        # Track individual oscillator components for hotswap
        self._square_oscillator = None

        self.component = self.create_engine_component()

    # AudioModuleInterface implementation
    def _on_pulsewidth_changed(self):
        """Handle pulse width changes - only update square oscillator if connected."""
        if len(self.square_port.cables) > 0 and self._square_oscillator is not None:
            # Hotswap: update pulse width directly on the square oscillator
            try:
                self._square_oscillator.pulsewidth = self.pulsewidth_knob.get_value()
                self.parameter_changed.emit(
                    "pulsewidth", self.pulsewidth_knob.get_value()
                )
            except (AttributeError, ValueError) as e:
                # If hotswap fails, log warning (but don't recreate component)
                import logging

                logging.warning(f"Failed to hotswap pulsewidth: {e}")

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
        pulsewidth = self.pulsewidth_knob.get_value()
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
            self._square_oscillator = SquareOscillator(
                freq, gain_db=OSCILLATOR_DEFAULT_GAIN_DB, pulsewidth=pulsewidth
            )
            components["Square"] = self._square_oscillator
        else:
            # Clear reference when not connected
            self._square_oscillator = None

        if not components:
            # No outputs connected - return a silent component
            return None
        elif len(components) == 1:
            # Only one output connected - return that component directly
            return next(iter(components.values()))

        return WaveAdder(*components.values())

    def get_output_component(self, port_name: str) -> AudioComponent | None:
        """Get the component for a specific output port.

        This allows each output to be independent instead of mixing them all together.
        This is the correct behavior for oscillators - Sine, Triangle, Square, and Sawtooth
        outputs should be separate signals, not mixed.

        Args:
            port_name: Name of the output port (e.g., "Sine", "Triangle")

        Returns:
            The oscillator component for that specific output, or None if not connected
        """
        freq = self.freq_knob.get_value()
        pulsewidth = self.pulsewidth_knob.get_value()

        # Return the specific oscillator for the requested output port
        if port_name == "Sine" and len(self.sine_port.cables) > 0:
            return SineOscillator(freq, gain_db=OSCILLATOR_DEFAULT_GAIN_DB)
        elif port_name == "Triangle" and len(self.triangle_port.cables) > 0:
            return TriangleOscillator(freq, gain_db=OSCILLATOR_DEFAULT_GAIN_DB)
        elif port_name == "Sawtooth" and len(self.sawtooth_port.cables) > 0:
            return SawtoothOscillator(freq, gain_db=OSCILLATOR_DEFAULT_GAIN_DB)
        elif port_name == "Square" and len(self.square_port.cables) > 0:
            # Create and store reference for hot-swapping pulsewidth
            self._square_oscillator = SquareOscillator(
                freq, gain_db=OSCILLATOR_DEFAULT_GAIN_DB, pulsewidth=pulsewidth
            )
            return self._square_oscillator

        return None
