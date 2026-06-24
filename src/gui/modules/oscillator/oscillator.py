from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from src.engine.oscillator import (
    SawtoothOscillator,
    SineOscillator,
    SquareOscillator,
    TriangleOscillator,
)
from src.gui.audio_config import audio_config
from src.gui.core.module import ModuleCategory, ModuleMetadata
from src.gui.core.module_registry import register_module
from src.gui.ui_constants import (
    DEFAULT_PW_PERCENTAGE_VALUE,
    MAX_PW_PERCENTAGE_VALUE,
    MIN_PW_PERCENTAGE_VALUE,
)
from src.gui.widgets import Knob
from src.gui.widgets.module_widget import ModuleWidget

if TYPE_CHECKING:
    from src.gui.core.port import Port

OSCILLATOR_DEFAULT_GAIN_DB = 0.0
OSCILLATOR_DEFAULT_FREQUENCY = 120
OSCILLATOR_MIN_FREQUENCY = 20
OSCILLATOR_MAX_FREQUENCY = 6000


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
            mode="analog",
        )
        self._triangle_oscillator = TriangleOscillator(
            freq,
            gain_db=OSCILLATOR_DEFAULT_GAIN_DB,
            sample_rate=sample_rate,
            mode="analog",
        )
        self._sawtooth_oscillator = SawtoothOscillator(
            freq,
            gain_db=OSCILLATOR_DEFAULT_GAIN_DB,
            sample_rate=sample_rate,
            mode="analog",
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
        self.freq_knob.value_changed.connect(self._on_frequency_changed)
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

        # Register with audio_config to receive sample rate change notifications
        audio_config.add_sample_rate_listener(self._on_global_sample_rate_changed)

    def process(self, num_samples: int | None = None):
        if num_samples is None:
            num_samples = audio_config.buffer_size

        # Generate samples for all ports that are connected
        # This ensures visualization modules can pull samples even without audio output
        for port, osc in zip(self.ports, self.oscs, strict=False):
            if port.is_connected and osc is not None:
                samples = osc.get_samples(num_samples)
                port.write(samples)

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
        """Handle frequency changes - update all connected oscillators."""
        new_freq = self.freq_knob.get_value()

        # Hotswap: update frequency directly on all oscillators
        self._sine_oscillator.frequency = new_freq
        self._triangle_oscillator.frequency = new_freq
        self._sawtooth_oscillator.frequency = new_freq
        self._square_oscillator.frequency = new_freq

    def _on_pulsewidth_changed(self):
        """Handle pulse width changes - only update square oscillator if connected."""
        if self.square_port.is_connected:
            self._square_oscillator.pulsewidth = self.pulsewidth_knob.get_value()
