from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHBoxLayout

from gui.audio_config import audio_config
from src.engine.oscillator import (
    SineOscillator,
    SawtoothOscillator,
    TriangleOscillator,
    SquareOscillator
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
    from src.gui.core.port import Port



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
        self.sine_port: Port = self.add_output("Sine")
        self.triangle_port: Port = self.add_output("Triangle")
        self.sawtooth_port: Port = self.add_output("Sawtooth")
        self.square_port: Port = self.add_output("Square")

        # Store ports for easy iteration
        self.ports = [
            self.sine_port,
            self.triangle_port,
            self.sawtooth_port,
            self.square_port,
        ]

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

        # Register parameters for automatic get/set
        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("pulsewidth", self.pulsewidth_knob)

        # Track individual oscillator components
        self._sine_oscillator = None
        self._triangle_oscillator = None
        self._sawtooth_oscillator = None
        self._square_oscillator = None

        # Create all oscillators
        self._create_oscillators()

        # Store oscillators for easy iteration
        self.oscs = [
            self._sine_oscillator,
            self._triangle_oscillator,
            self._sawtooth_oscillator,
            self._square_oscillator,
        ]

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

    def _on_frequency_changed(self):
        """Handle frequency changes - update all oscillators via hotswap."""
        new_freq = self.freq_knob.get_value()

        # Hotswap: update frequency directly on all oscillators
        self._sine_oscillator.frequency = new_freq
        self._triangle_oscillator.frequency = new_freq
        self._sawtooth_oscillator.frequency = new_freq
        self._square_oscillator.frequency = new_freq

    def _on_pulsewidth_changed(self):
        """Handle pulse width changes - only update square oscillator."""
        self._square_oscillator.pulsewidth = self.pulsewidth_knob.get_value()

    def _create_oscillators(self):
        """Create all LFO oscillators with bipolar output [-1, 1]."""
        freq = self.freq_knob.get_value()
        pulsewidth = self.pulsewidth_knob.get_value()
        # Use global sample rate from audio_config
        sample_rate = audio_config.sample_rate

        # Create oscillators with bipolar output for modulation
        self._sine_oscillator = SineOscillator(
            freq,
            gain_db=LFO_DEFAULT_GAIN_DB,
            wave_range=(-1, 1),
            sample_rate=sample_rate
        )
        self._triangle_oscillator = TriangleOscillator(
            freq,
            gain_db=LFO_DEFAULT_GAIN_DB,
            wave_range=(-1, 1),
            sample_rate=sample_rate
        )
        self._sawtooth_oscillator = SawtoothOscillator(
            freq,
            gain_db=LFO_DEFAULT_GAIN_DB,
            wave_range=(-1, 1),
            sample_rate=sample_rate
        )
        self._square_oscillator = SquareOscillator(
            freq,
            gain_db=LFO_DEFAULT_GAIN_DB,
            wave_range=(-1, 1),
            pulsewidth=pulsewidth,
            sample_rate=sample_rate
        )

    @staticmethod
    def get_cv_output_range() -> tuple[float, float]:
        """LFO outputs bipolar signal [-1, 1] for modulation.

        Returns:
            (-1.0, 1.0) - bipolar output range
        """
        return -1.0, 1.0

    def process(self, num_samples: int = 1):
        """Generate LFO signals and write to output ports.

        Generates low-frequency modulation signals from each oscillator type
        and writes them to the corresponding output ports if connected.

        Args:
            num_samples: Number of samples to generate (default: 1 for per-sample processing)
        """
        # Generate samples for each connected output
        for port, osc in zip(self.ports, self.oscs):
            if port.is_connected and osc is not None:
                # Generate samples from oscillator
                samples = osc.get_samples(num_samples)
                # Write to port for downstream modules
                port.write(samples)

