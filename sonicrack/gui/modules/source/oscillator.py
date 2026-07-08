from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout
from soniclab.generators.oscillators.oscillator import (
    SawtoothOscillator,
    SineOscillator,
    SquareOscillator,
    TriangleOscillator,
)

from sonicrack.gui.audio_config import audio_config
from sonicrack.gui.core.module import ModuleCategory, ModuleMetadata
from sonicrack.gui.core.runtime import RuntimeParameters
from sonicrack.gui.core.runtime_helpers import float_parameter
from sonicrack.gui.module_registry import register_module
from sonicrack.gui.modules.source._oscillator_runtime import render_with_frequency_ramp
from sonicrack.gui.ui_constants import (
    AUDIO_FREQUENCY_KNOB_CURVE,
    DEFAULT_PW_PERCENTAGE_VALUE,
    MAX_PW_PERCENTAGE_VALUE,
    MIN_PW_PERCENTAGE_VALUE,
)
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget

if TYPE_CHECKING:
    from sonicrack.gui.core.port import Port

OSCILLATOR_DEFAULT_GAIN_DB = 0.0
OSCILLATOR_DEFAULT_FREQUENCY = 120
OSCILLATOR_MIN_FREQUENCY = 11
OSCILLATOR_MAX_FREQUENCY = 6000


@register_module()
class OscillatorModule(ModuleWidget):
    """Oscillator module with frequency and gain controls."""

    runtime_kind = "multi_oscillator"

    metadata = ModuleMetadata(
        title="Oscillator",
        category=ModuleCategory.SOURCE,
        description="Multi-waveform oscillator with frequency and gain controls",
    )

    def __init__(self):
        """Initialize oscillator module."""
        super().__init__(
            width=220,
            height=235,
            color=QColor(80, 120, 200),
        )

        # Create oscillator components FIRST (before creating ports)
        # This allows us to pass component references to ports
        freq = OSCILLATOR_DEFAULT_FREQUENCY
        sample_rate = audio_config.sample_rate
        pulsewidth = DEFAULT_PW_PERCENTAGE_VALUE / 100

        self._sine_oscillator = SineOscillator(
            freq,
            gain_db=OSCILLATOR_DEFAULT_GAIN_DB,
            sample_rate=sample_rate,
            mode="pure",
        )
        self._triangle_oscillator = TriangleOscillator(
            freq,
            gain_db=OSCILLATOR_DEFAULT_GAIN_DB,
            sample_rate=sample_rate,
            mode="pure",
        )
        self._sawtooth_oscillator = SawtoothOscillator(
            freq,
            gain_db=OSCILLATOR_DEFAULT_GAIN_DB,
            sample_rate=sample_rate,
            mode="vcv",
        )
        self._square_oscillator = SquareOscillator(
            freq,
            gain_db=OSCILLATOR_DEFAULT_GAIN_DB,
            pulsewidth=pulsewidth,
            sample_rate=sample_rate,
            mode="vcv",
        )

        # Add four output ports - one for each waveform
        # Pass component references so ports know their associated engine components
        self.sine_port: Port = self.add_output("Sine", component=self._sine_oscillator)
        self.triangle_port: Port = self.add_output(
            "Triangle", component=self._triangle_oscillator
        )
        self.sawtooth_port: Port = self.add_output(
            "Sawtooth", component=self._sawtooth_oscillator
        )
        self.square_port: Port = self.add_output(
            "Square", component=self._square_oscillator
        )

        self.ports = [
            self.sine_port,
            self.triangle_port,
            self.sawtooth_port,
            self.square_port,
        ]
        self.oscs = [
            self._sine_oscillator,
            self._triangle_oscillator,
            self._sawtooth_oscillator,
            self._square_oscillator,
        ]
        self._last_runtime_frequency = freq

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Frequency control (knobs)
        knobs_layout = QHBoxLayout()
        self.freq_knob = Knob(
            label="Freq",
            description="Sets the base frequency of the oscillator",
            min_value=OSCILLATOR_MIN_FREQUENCY,
            max_value=OSCILLATOR_MAX_FREQUENCY,
            default_value=OSCILLATOR_DEFAULT_FREQUENCY,
            curve_points=AUDIO_FREQUENCY_KNOB_CURVE,
        )
        self.freq_knob.value_changed.connect(self._on_frequency_changed)
        knobs_layout.addWidget(self.freq_knob)
        layout.addLayout(knobs_layout)

        # Pulse width control (for square wave)
        pw_layout = QHBoxLayout()
        self.pulsewidth_knob = Knob(
            label="PW",
            description="Adjusts the pulse width of the square wave",
            min_value=MIN_PW_PERCENTAGE_VALUE / 100,
            max_value=MAX_PW_PERCENTAGE_VALUE / 100,
            default_value=DEFAULT_PW_PERCENTAGE_VALUE / 100,
        )
        self.pulsewidth_knob.value_changed.connect(self._on_pulsewidth_changed)
        pw_layout.addWidget(self.pulsewidth_knob)
        layout.addLayout(pw_layout)

        self.controls_widget.setLayout(layout)
        self.proxy = self._add_controls_to_module(self.controls_widget)

        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("pulsewidth", self.pulsewidth_knob)

        # Register with audio_config to receive sample rate change notifications
        audio_config.add_sample_rate_listener(self._on_global_sample_rate_changed)

    def _on_global_sample_rate_changed(self, new_sample_rate: int):
        """Handle global sample rate changes from audio_config.

        This is called automatically when the sample rate changes anywhere
        in the system (e.g., from the Output module).

        Args:
            new_sample_rate: New sample rate in Hz
        """
        # Update all oscillators with new sample rate
        if self._sine_oscillator:
            self._sine_oscillator.sample_rate = new_sample_rate
        if self._triangle_oscillator:
            self._triangle_oscillator.sample_rate = new_sample_rate
        if self._sawtooth_oscillator:
            self._sawtooth_oscillator.sample_rate = new_sample_rate
        if self._square_oscillator:
            self._square_oscillator.sample_rate = new_sample_rate

    def create_engine_component(
        self, input_components=None, modulation_components=None
    ) -> None:
        """Return no shared engine component for this module.

        The oscillator exposes independent components per output port via
        ``get_output_component()`` rather than a single component shared by all
        outputs.
        """
        return None

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Render each oscillator output for the current engine cycle."""
        frequency = float_parameter(parameters, "frequency", self.freq_knob.get_value)
        pulsewidth = float_parameter(
            parameters, "pulsewidth", self.pulsewidth_knob.get_value
        )

        self._square_oscillator.pulsewidth = pulsewidth

        rendered_frequency = self._last_runtime_frequency
        for port, osc in zip(self.ports, self.oscs, strict=False):
            if osc is not None:
                samples, rendered_frequency = render_with_frequency_ramp(
                    osc,
                    self._last_runtime_frequency,
                    frequency,
                    num_samples,
                )
                port.write(samples)
        self._last_runtime_frequency = rendered_frequency

    def get_output_component(self, port_name: str):
        """Return the engine component backing a specific waveform output."""
        port = self.outputs.get(port_name)
        if port is None or not port.is_connected:
            return None

        components = {
            "Sine": self._sine_oscillator,
            "Triangle": self._triangle_oscillator,
            "Sawtooth": self._sawtooth_oscillator,
            "Square": self._square_oscillator,
        }
        return components.get(port_name)

    # AudioModuleInterface implementation
    def _on_frequency_changed(self):
        """Handle frequency control changes."""
        new_freq = self.freq_knob.get_value()
        self.parameter_changed.emit("frequency", new_freq)

    def _on_pulsewidth_changed(self):
        """Handle pulse width changes - only update square oscillator if connected."""
        if self.square_port.is_connected:
            self._square_oscillator.pulsewidth = self.pulsewidth_knob.get_value()
