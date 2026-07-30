from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel
from soniclab.generators.oscillators.oscillator import (
    SawtoothOscillator,
    SineOscillator,
    SquareOscillator,
    TriangleOscillator,
)

from sonicrack.config.audio_config import audio_config
from sonicrack.constants import (
    DEFAULT_PW_PERCENTAGE_VALUE,
    MAX_PW_PERCENTAGE_VALUE,
    MIN_PW_PERCENTAGE_VALUE,
)
from sonicrack.gui.modules.source._oscillator_runtime import render_with_clock_resets
from sonicrack.gui.widgets import Knob
from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import register_module
from sonicrack.runtime.helpers import float_parameter, read_samples, str_parameter
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

        self.clock_input: Port = self.add_input("Clock", signal=PortSignal.TRIGGER)

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
        layout = self._begin_controls()

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

        self.amount_knob = Knob(
            label="Amount",
            description="Output depth / scale of the LFO",
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
        )
        self.amount_knob.value_changed.connect(
            lambda: self.parameter_changed.emit("amount", self.amount_knob.get_value())
        )
        pw_layout.addWidget(self.amount_knob)
        layout.addLayout(pw_layout)

        polarity_layout = QHBoxLayout()
        polarity_layout.addWidget(QLabel("Polarity:"))
        self.polarity_combo = QComboBox()
        self.polarity_combo.addItems(["Bipolar", "Unipolar"])
        self.polarity_combo.setToolTip(
            "Bipolar: -1…+1\nUnipolar: remap to 0…1 (amount still applies)"
        )
        self.polarity_combo.currentTextChanged.connect(
            lambda value: self.parameter_changed.emit("polarity", value)
        )
        polarity_layout.addWidget(self.polarity_combo)
        layout.addLayout(polarity_layout)

        self._finish_controls(layout)

        # Register parameters for automatic get/set
        self.register_parameter("frequency", self.freq_knob)
        self.register_parameter("pulsewidth", self.pulsewidth_knob)
        self.register_parameter("amount", self.amount_knob)
        self.register_parameter(
            "polarity",
            self.polarity_combo,
            getter="currentText",
            setter="setCurrentText",
        )

        # Register with audio_config to receive sample rate change notifications
        self._install_sample_rate_listener()

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
        amount = float_parameter(parameters, "amount", self.amount_knob.get_value)
        polarity = str_parameter(
            parameters, "polarity", self.polarity_combo.currentText
        ).lower()
        unipolar = polarity.startswith("uni")

        self._square_oscillator.pulsewidth = pulsewidth

        clock_signal = (
            read_samples(self.clock_input, num_samples)
            if self.clock_input.is_connected
            else None
        )

        # Only render live (connected) outputs. When nothing is connected yet
        # (unit tests / offline), fall back to all oscillators so state advances.
        active = [
            (index, port, osc)
            for index, (port, osc) in enumerate(
                zip(self.ports, self.oscs, strict=False)
            )
            if osc is not None and port.is_connected
        ]
        if not active:
            active = [
                (index, port, osc)
                for index, (port, osc) in enumerate(
                    zip(self.ports, self.oscs, strict=False)
                )
                if osc is not None
            ]

        final_clock = self._previous_clock if clock_signal is not None else 0.0
        for index, port, osc in active:
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
            samples = np.asarray(samples, dtype=np.float32)
            if unipolar:
                # Map bipolar [-1, 1] → [0, 1], then scale by amount.
                samples = (samples * 0.5 + 0.5) * amount
            else:
                samples = samples * amount
            port.write(samples)
            self._last_runtime_frequencies[index] = rendered_frequency
            if len(samples) > 0:
                self._last_output_values[index] = float(samples[-1])
        self._previous_clock = final_clock

    def get_cv_output_range(self) -> tuple[float, float]:
        """LFO output range depends on polarity mode.

        Returns:
            (-1.0, 1.0) for bipolar or (0.0, 1.0) for unipolar.
        """
        polarity = self.polarity_combo.currentText().lower()
        if polarity.startswith("uni"):
            return 0.0, 1.0
        return -1.0, 1.0
