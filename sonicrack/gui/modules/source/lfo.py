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

from sonicrack.config.audio_config import audio_config
from sonicrack.gui.modules.source._oscillator_runtime import render_with_clock_resets
from sonicrack.constants import MIN_PW_PERCENTAGE_VALUE, MAX_PW_PERCENTAGE_VALUE, \
    DEFAULT_PW_PERCENTAGE_VALUE
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples
from sonicrack.runtime.specs import RuntimeParameters

if TYPE_CHECKING:
    from sonicrack.patching.port import Port


LFO_MIN_FREQUENCY = 0.01
LFO_MAX_FREQUENCY = 20.0  # LFO frequency range in Hz
LFO_DEFAULT_FREQUENCY = 1.0  # Default LFO frequency in Hz
LFO_DEFAULT_GAIN_DB = 0
# Increased slew time to prevent clicks/pops when changing frequency
# Smoothing happens in log (pitch) space for natural-sounding transitions
LFO_FREQUENCY_SLEW_TIME_MS = 250.0  # 250ms for click-free frequency changes
LFO_CLOCK_RESET_SMOOTHING_MS = 2.0


@register_module()
class LFOModule(ModuleWidget):
    """LFO (Low Frequency Oscillator) module for modulation.

    Similar to Oscillator but optimized for modulation (0.01 Hz - 20 Hz).
    """

    runtime_kind = "multi_oscillator"

    metadata = ModuleMetadata(
        title="LFO",
        category=ModuleCategory.SOURCE,
        description="Low-frequency 1V/oct control-voltage source",
    )

    def __init__(self):
        """Initialize LFO module."""
        super().__init__(
            width=240,
            height=300,
            color=QColor(100, 140, 200),
        )

        self.clock_input: Port = self.add_input("Clock", signal=PortSignal.GATE)

        # Create oscillator components FIRST (before creating ports)
        freq = LFO_DEFAULT_FREQUENCY
        pulsewidth = DEFAULT_PW_PERCENTAGE_VALUE / 100
        sample_rate = audio_config.sample_rate

        # Create oscillators with bipolar output for modulation
        self._sine_oscillator = SineOscillator(
            freq,
            gain_db=LFO_DEFAULT_GAIN_DB,
            wave_range=(-1, 1),
            sample_rate=sample_rate,
            mode="pure",
        )
        self._triangle_oscillator = TriangleOscillator(
            freq,
            gain_db=LFO_DEFAULT_GAIN_DB,
            wave_range=(-1, 1),
            sample_rate=sample_rate,
            mode="pure",
        )
        self._sawtooth_oscillator = SawtoothOscillator(
            freq,
            gain_db=LFO_DEFAULT_GAIN_DB,
            wave_range=(-1, 1),
            sample_rate=sample_rate,
            mode="vcv",
        )
        self._square_oscillator = SquareOscillator(
            freq,
            gain_db=LFO_DEFAULT_GAIN_DB,
            wave_range=(-1, 1),
            pulsewidth=pulsewidth,
            sample_rate=sample_rate,
            mode="vcv",
        )

        # Add four output ports - one for each waveform
        # Pass component references so ports know their associated engine components
        self.sine_port: Port = self.add_output(
            "Sine",
            component=self._sine_oscillator,
            signal=PortSignal.CONTROL_CV,
        )
        self.triangle_port: Port = self.add_output(
            "Triangle",
            component=self._triangle_oscillator,
            signal=PortSignal.CONTROL_CV,
        )
        self.sawtooth_port: Port = self.add_output(
            "Sawtooth",
            component=self._sawtooth_oscillator,
            signal=PortSignal.CONTROL_CV,
        )
        self.square_port: Port = self.add_output(
            "Square",
            component=self._square_oscillator,
            signal=PortSignal.CONTROL_CV,
        )

        # Store ports for easy iteration
        self.ports = [
            self.sine_port,
            self.triangle_port,
            self.sawtooth_port,
            self.square_port,
        ]

        # Store oscillators for easy iteration
        self.oscs = [
            self._sine_oscillator,
            self._triangle_oscillator,
            self._sawtooth_oscillator,
            self._square_oscillator,
        ]
        self._last_runtime_frequencies = [freq] * len(self.oscs)
        self._previous_clock = 0.0
        self._last_output_values: list[float | None] = [None] * len(self.oscs)

        # Use helper methods for UI construction
        self.controls_widget = self._create_controls_container()
        layout = self._create_standard_layout()

        # Frequency control (knob) - optimized for LFO range
        knobs_layout = QHBoxLayout()
        self.freq_knob = Knob(
            label="Freq",
            description="Sets the frequency of the LFO",
            min_value=LFO_MIN_FREQUENCY,
            max_value=LFO_MAX_FREQUENCY,
            default_value=LFO_DEFAULT_FREQUENCY,
            logarithmic=True,
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

        # Register parameters for automatic get/set
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

    def _on_frequency_changed(self):
        """Handle frequency control changes."""
        new_freq = self.freq_knob.get_value()
        self.parameter_changed.emit("frequency", new_freq)

    def _on_pulsewidth_changed(self):
        """Handle pulse width changes - only update square oscillator."""
        pulsewidth = self.pulsewidth_knob.get_value()
        self._square_oscillator.pulsewidth = pulsewidth
        self.parameter_changed.emit("pulsewidth", pulsewidth)

    def process_runtime(self, num_samples: int, parameters: RuntimeParameters) -> None:
        """Render each LFO output for the current engine cycle."""
        frequency = float_parameter(parameters, "frequency", self.freq_knob.get_value)
        pulsewidth = float_parameter(
            parameters, "pulsewidth", self.pulsewidth_knob.get_value
        )

        self._square_oscillator.pulsewidth = pulsewidth

        clock_signal = (
            read_samples(self.clock_input, num_samples)
            if self.clock_input.is_connected
            else None
        )

        final_clock = 0.0
        for index, (port, osc) in enumerate(zip(self.ports, self.oscs, strict=False)):
            if osc is not None:
                samples, rendered_frequency, final_clock = render_with_clock_resets(
                    osc,
                    self._last_runtime_frequencies[index],
                    frequency,
                    num_samples,
                    clock_signal,
                    self._previous_clock,
                    self._last_output_values[index],
                    LFO_CLOCK_RESET_SMOOTHING_MS,
                    LFO_FREQUENCY_SLEW_TIME_MS,
                )
                port.write(samples)
                self._last_runtime_frequencies[index] = rendered_frequency
                if len(samples) > 0:
                    self._last_output_values[index] = float(samples[-1])
        self._previous_clock = final_clock

    @staticmethod
    def get_cv_output_range() -> tuple[float, float]:
        """LFO outputs bipolar control voltage in volts.

        Returns:
            (-1.0, 1.0) - bipolar modulation range; when patched to pitch inputs,
            each volt corresponds to one octave.
        """
        return -1.0, 1.0
