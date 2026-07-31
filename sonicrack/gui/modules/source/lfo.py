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


class _RandomLfo:
    """Free-running / clocked sample-and-hold random LFO generator."""

    def __init__(self, sample_rate: float = 44100.0) -> None:
        self.sample_rate = float(sample_rate)
        self.phase = 0.0
        self.value = 0.0
        self._prev_clock = 0.0
        self._rng = np.random.default_rng()

    def reset(self) -> None:
        self.phase = 0.0
        self.value = 0.0
        self._prev_clock = 0.0

    def process(
        self,
        frequency: float,
        num_samples: int,
        clock: np.ndarray | None = None,
        prev_clock: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        n = int(num_samples)
        out = np.empty(n, dtype=np.float32)
        freq = max(0.0, float(frequency))
        phase = self.phase
        value = self.value
        sr = max(1.0, self.sample_rate)

        if clock is not None:
            c = np.asarray(clock, dtype=np.float32).reshape(-1)
            if c.size < n:
                c = np.pad(c, (0, n - c.size))
            prev = float(prev_clock)
            for i in range(n):
                clk = float(c[i])
                if prev < 0.5 <= clk:
                    value = float(self._rng.uniform(-1.0, 1.0))
                out[i] = value
                prev = clk
            self._prev_clock = prev
        else:
            # Free-run: new random sample every cycle of `frequency`.
            for i in range(n):
                phase += freq / sr
                if phase >= 1.0:
                    phase -= int(phase)
                    value = float(self._rng.uniform(-1.0, 1.0))
                out[i] = value
        self.phase = phase
        self.value = value
        return out, value


class _SmoothRandomLfo:
    """Low-pass filtered random walk (smooth random / noise LFO)."""

    def __init__(self, sample_rate: float = 44100.0) -> None:
        self.sample_rate = float(sample_rate)
        self.state = 0.0
        self._rng = np.random.default_rng()

    def reset(self) -> None:
        self.state = 0.0

    def process(self, frequency: float, num_samples: int) -> np.ndarray:
        n = int(num_samples)
        out = np.empty(n, dtype=np.float32)
        sr = max(1.0, self.sample_rate)
        # Cutoff roughly tracks LFO rate; higher rate = faster wander.
        cutoff = max(0.01, min(20.0, float(frequency)))
        # One-pole coefficient from cutoff.
        coeff = 1.0 - np.exp(-2.0 * np.pi * cutoff / sr)
        state = self.state
        noise = self._rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
        for i in range(n):
            state += coeff * (float(noise[i]) - state)
            out[i] = state
        self.state = state
        return out


@register_module()
class LFOModule(ModuleWidget):
    """LFO (Low Frequency Oscillator) module for modulation.

    Classic shapes (sine/tri/saw/square) plus ramp, sample-and-hold random,
    and smooth random. Clock input resets classic oscillators and re-samples
    the random output.
    """

    runtime_kind = "multi_oscillator"

    metadata = ModuleMetadata(
        title="LFO",
        category=ModuleCategory.SOURCE,
        description="Low-frequency modulation source with classic + random shapes",
    )

    def __init__(self):
        """Initialize LFO module."""
        super().__init__(
            width=260,
            height=340,
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
        self._random_lfo = _RandomLfo(sample_rate=sample_rate)
        self._smooth_random_lfo = _SmoothRandomLfo(sample_rate=sample_rate)

        # Classic waveform ports
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
        self.ramp_port: Port = self.add_output(
            "Ramp",
            component=self._sawtooth_oscillator,
            signal=PortSignal.CONTROL_CV,
        )
        self.square_port: Port = self.add_output(
            "Square",
            component=self._square_oscillator,
            signal=PortSignal.CONTROL_CV,
        )
        self.random_port: Port = self.add_output(
            "Random",
            signal=PortSignal.CONTROL_CV,
        )
        self.smooth_port: Port = self.add_output(
            "Smooth",
            signal=PortSignal.CONTROL_CV,
        )

        # Classic oscillator ports/oscs iterated together
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
        self._last_runtime_frequencies = [freq] * len(self.oscs)
        self._previous_clock = 0.0
        self._last_output_values: list[float | None] = [None] * len(self.oscs)
        self._last_saw_samples: np.ndarray | None = None

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
        self._random_lfo.sample_rate = float(new_sample_rate)
        self._smooth_random_lfo.sample_rate = float(new_sample_rate)

    def _on_frequency_changed(self):
        """Handle frequency control changes."""
        new_freq = self.freq_knob.get_value()
        self.parameter_changed.emit("frequency", new_freq)

    def _on_pulsewidth_changed(self):
        """Handle pulse width changes - only update square oscillator."""
        pulsewidth = self.pulsewidth_knob.get_value()
        self._square_oscillator.pulsewidth = pulsewidth
        self.parameter_changed.emit("pulsewidth", pulsewidth)

    def _scale_polarity(
        self, samples: np.ndarray, *, unipolar: bool, amount: float
    ) -> np.ndarray:
        if unipolar:
            return (samples * 0.5 + 0.5) * amount
        return samples * amount

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

        all_outs = (
            *self.ports,
            self.ramp_port,
            self.random_port,
            self.smooth_port,
        )
        any_connected = any(p.is_connected for p in all_outs)
        # When nothing is patched (tests / offline), advance every shape.
        force_all = not any_connected

        active = [
            (index, port, osc)
            for index, (port, osc) in enumerate(
                zip(self.ports, self.oscs, strict=False)
            )
            if osc is not None and (force_all or port.is_connected)
        ]
        need_saw = force_all or self.sawtooth_port.is_connected or self.ramp_port.is_connected
        if need_saw and not any(i == 2 for i, _, _ in active):
            active.append((2, self.sawtooth_port, self._sawtooth_oscillator))

        final_clock = self._previous_clock if clock_signal is not None else 0.0
        saw_samples: np.ndarray | None = None
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
            if index == 2:
                saw_samples = samples
            if force_all or port.is_connected:
                port.write(
                    self._scale_polarity(samples, unipolar=unipolar, amount=amount)
                )
            self._last_runtime_frequencies[index] = rendered_frequency
            if len(samples) > 0:
                self._last_output_values[index] = float(samples[-1])
        self._previous_clock = final_clock
        self._last_saw_samples = saw_samples

        # Ramp = inverted sawtooth
        if force_all or self.ramp_port.is_connected:
            if saw_samples is None:
                saw_samples = np.zeros(num_samples, dtype=np.float32)
            self.ramp_port.write(
                self._scale_polarity(-saw_samples, unipolar=unipolar, amount=amount)
            )

        # Random sample-and-hold (free-run at Freq, or clocked)
        if force_all or self.random_port.is_connected:
            random_samples, _ = self._random_lfo.process(
                frequency,
                num_samples,
                clock=clock_signal,
                prev_clock=0.0,
            )
            self.random_port.write(
                self._scale_polarity(random_samples, unipolar=unipolar, amount=amount)
            )

        # Smooth random walk
        if force_all or self.smooth_port.is_connected:
            smooth = self._smooth_random_lfo.process(frequency, num_samples)
            self.smooth_port.write(
                self._scale_polarity(smooth, unipolar=unipolar, amount=amount)
            )

    def get_cv_output_range(self) -> tuple[float, float]:
        """LFO output range depends on polarity mode.

        Returns:
            (-1.0, 1.0) for bipolar or (0.0, 1.0) for unipolar.
        """
        polarity = self.polarity_combo.currentText().lower()
        if polarity.startswith("uni"):
            return 0.0, 1.0
        return -1.0, 1.0
