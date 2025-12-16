from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from src.engine.oscillator import SquareOscillator
from src.engine.composer import WaveAdder
from src.engine.oscillator import (
    SineOscillator,
    SawtoothOscillator,
    TriangleOscillator,
)
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.ui_constants import (
    MIN_PW_PERCENTAGE_VALUE,
    MAX_PW_PERCENTAGE_VALUE,
    DEFAULT_PW_PERCENTAGE_VALUE,
)
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget

if TYPE_CHECKING:
    from src.engine.audio_component import AudioComponent


LFO_MIN_FREQUENCY = 0.01
LFO_MAX_FREQUENCY = 20.0  # LFO frequency range in Hz
LFO_DEFAULT_FREQUENCY = 1.0  # Default LFO frequency in Hz
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
            height=300,
            color=QColor(100, 140, 200),
        )

        # Add four output ports - one for each waveform
        self.sine_port = self.add_output("Sine")
        self.triangle_port = self.add_output("Triangle")
        self.sawtooth_port = self.add_output("Sawtooth")
        self.square_port = self.add_output("Square")

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
        self.freq_knob = Knob(
            "Frequency (Hz)",
            LFO_MIN_FREQUENCY,
            LFO_MAX_FREQUENCY,
            LFO_DEFAULT_FREQUENCY,
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

        # Register parameters for automatic get/set
        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("pulsewidth", self.pulsewidth_knob)

        # Track individual oscillator components for hotswap
        self._square_oscillator = None

        # Create initial components
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
        pulsewidth = self.pulsewidth_knob.get_value()
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
            self._square_oscillator = SquareOscillator(
                freq,
                gain_db=LFO_DEFAULT_GAIN_DB,
                wave_range=(-1, 1),
                pulsewidth=pulsewidth,
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

    def process(self, num_samples: int = 1):
        """Generate LFO signals and write to output ports.

        Generates low-frequency modulation signals from each oscillator type
        and writes them to the corresponding output ports if connected.

        Args:
            num_samples: Number of samples to generate (default: 1 for per-sample processing)

        Note:
            In the current architecture, this method is not actively called during playback.
            The audio engine directly calls get_samples() on the compiled AudioComponents.
            This method exists to satisfy the AudioModule interface and for potential
            future use in a more modular processing pipeline.
        """
        freq = self.freq_knob.get_value()
        pulsewidth = self.pulsewidth_knob.get_value()

        # Generate and write samples for each connected output
        port_oscillators = {
            "Sine": (self.sine_port, lambda: SineOscillator(freq, gain_db=LFO_DEFAULT_GAIN_DB, wave_range=(-1, 1))),
            "Triangle": (self.triangle_port, lambda: TriangleOscillator(freq, gain_db=LFO_DEFAULT_GAIN_DB, wave_range=(-1, 1))),
            "Sawtooth": (self.sawtooth_port, lambda: SawtoothOscillator(freq, gain_db=LFO_DEFAULT_GAIN_DB, wave_range=(-1, 1))),
            "Square": (self.square_port, lambda: SquareOscillator(freq, gain_db=LFO_DEFAULT_GAIN_DB, wave_range=(-1, 1), pulsewidth=pulsewidth)),
        }

        for port_name, (port, osc_factory) in port_oscillators.items():
            if port.is_connected:
                # Get or create oscillator
                if port_name == "Square" and self._square_oscillator is not None:
                    osc = self._square_oscillator
                else:
                    osc = osc_factory()

                # Generate and write samples
                samples = osc.get_samples(num_samples)
                port.write(samples)

