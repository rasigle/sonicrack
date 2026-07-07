"""PolyBLEP Oscillator Implementation - API Compatible with Standard Oscillators.

This module provides bandlimited oscillators using the PolyBLEP (Polynomial
Bandlimited Step) algorithm, which is simpler and more efficient than MinBLEP while
still providing good antialiasing characteristics.

PolyBLEP works by smoothing discontinuities in waveforms using polynomial corrections
at the transition points. This reduces aliasing without requiring large lookup tables.

The PolyBLEPOscillator class is a drop-in replacement for standard oscillators,
matching the full Oscillator API including gain_db, amplitude, phase control,
iterator interface, and vectorized generation.

Based on the Cmajor Standard Library oscillators.
Copyright (C)2024 Cmajor Software Ltd - ISC License
Ported to Python for AudioPlayground
"""

import threading
from enum import Enum

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    ComponentDescriptor,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.core.registry import ComponentCategory, register_component
from src.engine.generators.oscillators.oscillator_base import Oscillator
from src.engine.utils.decorators import track_provided_args


class WaveShape(Enum):
    """Supported waveform shapes."""

    SINE = "sine"
    TRIANGLE = "triangle"
    SQUARE = "square"
    SAWTOOTH_UP = "sawtoothup"
    SAWTOOTH_DOWN = "sawtoothdown"


class PolyBLEPWaveforms:
    """Static methods for generating PolyBLEP waveforms.

    PolyBLEP (Polynomial Bandlimited Step) uses polynomial corrections
    near discontinuities to reduce aliasing.
    """

    @staticmethod
    def polyblep(phase: float, increment: float) -> float:
        """Calculate PolyBLEP correction for a discontinuity.

        Args:
            phase: Current phase [0.0, 1.0)
            increment: Phase increment per sample

        Returns:
            PolyBLEP correction value
        """
        # Near start of cycle
        if phase < increment:
            p = phase / increment
            return (p + p) - (p * p) - 1.0

        # Near end of cycle
        if phase > 1.0 - increment:
            p = (phase - 1.0) / increment
            return (p + p) + (p * p) + 1.0

        return 0.0

    @staticmethod
    def sine(phase: float) -> float:
        """Generate sine wave (no PolyBLEP needed)."""
        return np.sin(2.0 * np.pi * phase)

    @staticmethod
    def square(phase: float, increment: float, pulsewidth: float = 0.5) -> float:
        """Generate antialiased square wave with variable pulse width.

        Args:
            phase: Current phase [0.0, 1.0)
            increment: Phase increment per sample
            pulsewidth: Pulse width (0.0 to 1.0), default 0.5 for 50% duty cycle

        Returns:
            Square wave sample value
        """
        # Generate naive square with specified pulsewidth
        naive = -1.0 if phase < pulsewidth else 1.0

        # Apply PolyBLEP at rising edge (phase = 0)
        correction = PolyBLEPWaveforms.polyblep(phase, increment)

        # Apply PolyBLEP at falling edge (phase = pulsewidth)
        phase_shifted = (phase - pulsewidth + 1.0) % 1.0
        correction -= PolyBLEPWaveforms.polyblep(phase_shifted, increment)

        return naive - correction

    @staticmethod
    def sawtooth(phase: float, increment: float) -> float:
        """Generate antialiased sawtooth wave."""
        naive = (phase * 2.0) - 1.0
        correction = PolyBLEPWaveforms.polyblep(phase, increment)
        return naive - correction


@register_component()
class PolyBLEPOscillator(Oscillator):
    """PolyBLEP oscillator with full Oscillator API compatibility.

    Inherits from Oscillator, providing standard amplitude smoothing (RuntimeParameter),
    wave range conversion, and property hooks. Adds bandlimited waveform generation
    via PolyBLEP correction at discontinuities.

    Supports all waveform shapes: sine, square, sawtooth (up/down), and triangle.
    Square waves support variable pulse width.

    Example:
        >>> osc = PolyBLEPOscillator(frequency=440, gain_db=-12,
        ...                          wave_shape=WaveShape.SQUARE)
        >>> samples = osc.get_samples(1000)
        >>> osc.gain_db = -6
        >>> osc.frequency = 880
    """

    descriptor = ComponentDescriptor(
        name="PolyBLEPOscillator",
        category=ComponentCategory.OSCILLATOR,
        fluent_api_name="polyblep_oscillator",
        description="PolyBLEP Bandlimited Oscillator",
        tags=["oscillator", "polyblep", "bandlimited", "synthesis"],
        parameters=make_parameter_descriptors(
            "frequency",
            "gain_db",
            "amplitude",
            "phase",
            "sample_rate",
            "wave_range",
            "wave_shape",
            "pulsewidth",
            wave_shape=ParameterDescriptor(
                name="wave_shape",
                default=WaveShape.SAWTOOTH_UP,
                choices=tuple(WaveShape),
                description="PolyBLEP waveform shape.",
            ),
            pulsewidth=ParameterDescriptor(
                name="pulsewidth",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
                description="Square wave pulse width.",
            ),
        ),
    )

    @track_provided_args
    def __init__(
        self,
        frequency: float = 440.0,
        amplitude: float = 1.0,
        gain_db: float | None = DEFAULT_GAIN_DB,
        phase: float = 0.0,
        sample_rate: int | float = DEFAULT_SAMPLE_RATE,
        wave_range: tuple[float, float] = (-1, 1),
        wave_shape: WaveShape = WaveShape.SAWTOOTH_UP,
        pulsewidth: float = 0.5,
    ):
        """Initialize PolyBLEP oscillator.

        Args:
            frequency: Oscillator frequency in Hz (default: 440.0)
            amplitude: Linear amplitude (0.0 to 1.0+, default: 1.0)
                Note: Ignored if gain_db is specified
            gain_db: Gain in decibels (default: 0.0)
                Set to None to use amplitude parameter instead
            phase: Initial phase in degrees (default: 0.0)
            sample_rate: Sample rate in Hz (default: 44100)
            wave_range: Output range tuple (min, max), default: (-1, 1)
            wave_shape: Waveform shape (default: SAWTOOTH_UP)
            pulsewidth: Pulse width for square wave (0.0 to 1.0, default: 0.5)
                0.5 = 50% duty cycle (standard square wave)
                0.1 = 10% duty cycle (narrow pulse)
                0.9 = 90% duty cycle (wide pulse)
        """
        if not 0.0 < pulsewidth < 1.0:
            raise ValueError(
                f"pulsewidth must be between 0.0 and 1.0, got {pulsewidth}"
            )
        self._pulsewidth = pulsewidth
        self.wave_shape = wave_shape

        # Initialize base class (handles freq, amp, gain_db, phase, wave_range,
        # RuntimeParameter amplitude smoothing, and calls iter(self) -> reset())
        super().__init__(
            frequency=frequency,
            amplitude=amplitude,
            gain_db=gain_db,
            phase=phase,
            sample_rate=sample_rate,
            wave_range=wave_range,
        )

        # PolyBLEP-specific state
        self._phase_normalized = (phase % 360.0) / 360.0
        self._increment = (frequency / self._sample_rate) % 1.0
        self._triangle_accumulator = 0.0
        self._state_lock = threading.RLock()

    # --- Hook overrides to sync PolyBLEP internal state ---

    def _post_freq_set(self) -> None:
        """Update phase increment when frequency changes."""
        self._increment = (self._f / self._sample_rate) % 1.0

    def _post_phase_set(self) -> None:
        """Update normalized phase when phase (degrees) changes."""
        self._phase_normalized = (self._p % 360.0) / 360.0

    def _post_sample_rate_set(self) -> None:
        """Update phase increment when sample rate changes."""
        self._increment = (self._f / self._sample_rate) % 1.0

    def _initialize_osc(self) -> None:
        """Reset oscillator-specific state (called by reset())."""
        self._phase_normalized = (self._phase % 360.0) / 360.0
        self._increment = (self._f / self._sample_rate) % 1.0
        self._triangle_accumulator = 0.0

    # --- Pulse width property (PolyBLEP-specific) ---

    @property
    def pulsewidth(self) -> float:
        """Current pulse width (0.0 to 1.0) for square wave.

        Only affects square wave generation.
        0.5 = 50% duty cycle (standard square)
        0.1 = 10% duty cycle (narrow pulse)
        0.9 = 90% duty cycle (wide pulse)
        """
        return self._pulsewidth

    @pulsewidth.setter
    def pulsewidth(self, value: float) -> None:
        """Set pulse width for square wave.

        Args:
            value: Pulse width between 0.0 and 1.0

        Raises:
            ValueError: If value is outside valid range
        """
        if not 0.0 < value < 1.0:
            raise ValueError(f"pulsewidth must be between 0.0 and 1.0, got {value}")
        self._pulsewidth = value

    # --- Iterator interface ---

    def __next__(self) -> float:
        """Generate next sample (iterator interface)."""
        with self._state_lock:
            phase = self._phase_normalized

            # Generate raw waveform value (no amplitude, no range)
            if self.wave_shape == WaveShape.SINE:
                value = PolyBLEPWaveforms.sine(phase)
            elif self.wave_shape == WaveShape.SQUARE:
                value = PolyBLEPWaveforms.square(
                    phase, self._increment, self._pulsewidth
                )
            elif self.wave_shape == WaveShape.SAWTOOTH_UP:
                value = PolyBLEPWaveforms.sawtooth(phase, self._increment)
            elif self.wave_shape == WaveShape.SAWTOOTH_DOWN:
                value = -PolyBLEPWaveforms.sawtooth(phase, self._increment)
            elif self.wave_shape == WaveShape.TRIANGLE:
                square_val = PolyBLEPWaveforms.square(
                    phase, self._increment, self._pulsewidth
                )
                self._triangle_accumulator = (
                    self._increment * square_val
                    + self._triangle_accumulator * (1.0 - (0.25 * self._increment))
                )
                value = self._triangle_accumulator * 4.0
            else:
                value = 0.0

            # Apply wave range and amplitude via base class
            value = self._apply_wave_range_value(value)
            amp = float(
                self._apply_amplitude_to_buffer(np.array([value], dtype=np.float32))[0]
            )

            # Advance phase
            self._phase_normalized = (self._phase_normalized + self._increment) % 1.0

            return amp

    # --- Vectorized generation ---

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using vectorization (high performance).

        Args:
            n: Number of samples to generate

        Returns:
            Array of n samples
        """
        n = int(n)
        with self._state_lock:
            # Generate phase array
            phases: np.ndarray | None = (
                self._phase_normalized + self._increment * np.arange(n)
            ) % 1.0

            # Generate raw waveform (no amplitude, no range)
            if self.wave_shape == WaveShape.SINE:
                samples = np.sin(2.0 * np.pi * phases)
            elif self.wave_shape == WaveShape.SQUARE:
                samples = self._generate_square_vectorized(phases)
            elif self.wave_shape == WaveShape.SAWTOOTH_UP:
                samples = self._generate_sawtooth_vectorized(phases)
            elif self.wave_shape == WaveShape.SAWTOOTH_DOWN:
                samples = -self._generate_sawtooth_vectorized(phases)
            elif self.wave_shape == WaveShape.TRIANGLE:
                samples = self._generate_triangle_vectorized(n)
                phases = None  # Triangle updates phase internally
            else:
                samples = np.zeros(n, dtype=np.float32)

            # Apply wave range and amplitude via base class
            samples = self._apply_wave_range_values(samples)
            samples = self._apply_amplitude_to_buffer(samples)

            # Update phase
            if phases is not None:
                self._phase_normalized = (
                    (phases[-1] + self._increment) % 1.0
                    if n > 0
                    else self._phase_normalized
                )

            return samples.astype(np.float32)

    def _generate_square_vectorized(self, phases: np.ndarray) -> np.ndarray:
        """Generate PolyBLEP square wave with variable pulsewidth (vectorized)."""
        output = np.where(phases < self._pulsewidth, -1.0, 1.0).astype(np.float64)

        # PolyBLEP at rising edge (phase = 0)
        mask1 = phases < self._increment
        if np.any(mask1):
            p = phases[mask1] / self._increment
            output[mask1] -= (p + p) - (p * p) - 1.0

        mask2 = phases > 1.0 - self._increment
        if np.any(mask2):
            p = (phases[mask2] - 1.0) / self._increment
            output[mask2] -= (p + p) + (p * p) + 1.0

        # PolyBLEP at falling edge (phase = pulsewidth)
        phases_shifted = (phases - self._pulsewidth + 1.0) % 1.0
        mask3 = phases_shifted < self._increment
        if np.any(mask3):
            p = phases_shifted[mask3] / self._increment
            output[mask3] += (p + p) - (p * p) - 1.0

        mask4 = phases_shifted > 1.0 - self._increment
        if np.any(mask4):
            p = (phases_shifted[mask4] - 1.0) / self._increment
            output[mask4] += (p + p) + (p * p) + 1.0

        return output

    def _generate_sawtooth_vectorized(self, phases: np.ndarray) -> np.ndarray:
        """Generate PolyBLEP sawtooth wave (vectorized)."""
        output = (phases * 2.0 - 1.0).astype(np.float64)

        mask1 = phases < self._increment
        if np.any(mask1):
            p = phases[mask1] / self._increment
            output[mask1] -= (p + p) - (p * p) - 1.0

        mask2 = phases > 1.0 - self._increment
        if np.any(mask2):
            p = (phases[mask2] - 1.0) / self._increment
            output[mask2] -= (p + p) + (p * p) + 1.0

        return output

    def _generate_triangle_vectorized(self, n: int) -> np.ndarray:
        """Generate triangle via vectorized square + sequential accumulator.

        The square wave is vectorized for performance; the leaky integrator
        accumulator is inherently sequential but operates on vectorized input.
        """
        # Generate all phase values upfront
        phases = (self._phase_normalized + self._increment * np.arange(n)) % 1.0

        # Vectorized square wave generation
        square_output = np.where(phases < self._pulsewidth, -1.0, 1.0).astype(
            np.float64
        )

        # PolyBLEP at rising edge
        mask1 = phases < self._increment
        if np.any(mask1):
            p = phases[mask1] / self._increment
            square_output[mask1] -= (p + p) - (p * p) - 1.0

        mask2 = phases > 1.0 - self._increment
        if np.any(mask2):
            p = (phases[mask2] - 1.0) / self._increment
            square_output[mask2] -= (p + p) + (p * p) + 1.0

        # PolyBLEP at falling edge
        phases_shifted = (phases - self._pulsewidth + 1.0) % 1.0
        mask3 = phases_shifted < self._increment
        if np.any(mask3):
            p = phases_shifted[mask3] / self._increment
            square_output[mask3] += (p + p) - (p * p) - 1.0

        mask4 = phases_shifted > 1.0 - self._increment
        if np.any(mask4):
            p = (phases_shifted[mask4] - 1.0) / self._increment
            square_output[mask4] += (p + p) + (p * p) + 1.0

        # Sequential leaky integrator (inherently stateful)
        samples = np.empty(n, dtype=np.float64)
        acc = self._triangle_accumulator
        decay = 1.0 - 0.25 * self._increment
        for i in range(n):
            acc = self._increment * square_output[i] + acc * decay
            samples[i] = acc * 4.0
        self._triangle_accumulator = acc

        # Update phase
        self._phase_normalized = (
            (phases[-1] + self._increment) % 1.0 if n > 0 else self._phase_normalized
        )

        return samples

    # --- Modulation API (for ModulatedOscillator vectorized path) ---

    def render_modulated_waveform(
        self,
        freqs: np.ndarray,
        phase_offsets_deg: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Render waveform with per-sample frequency and phase modulation.

        Args:
            freqs: Array of frequencies (Hz) per sample
            phase_offsets_deg: Optional phase offsets in degrees per sample

        Returns:
            Tuple of (waveform_samples, phase_state_dict)
        """
        n = len(freqs)
        increments = (freqs / self._sample_rate) % 1.0

        # Build phase array from current state
        phases = np.empty(n, dtype=np.float64)
        p = self._phase_normalized
        for i in range(n):
            phases[i] = p
            p = (p + increments[i]) % 1.0
        final_phase = p

        # Apply phase offsets if provided
        if phase_offsets_deg is not None:
            phases = (phases + phase_offsets_deg / 360.0) % 1.0

        # Generate raw waveform
        if self.wave_shape == WaveShape.SINE:
            samples = np.sin(2.0 * np.pi * phases)
        elif self.wave_shape == WaveShape.SQUARE:
            samples = self._generate_square_vectorized(phases)
        elif self.wave_shape == WaveShape.SAWTOOTH_UP:
            samples = self._generate_sawtooth_vectorized(phases)
        elif self.wave_shape == WaveShape.SAWTOOTH_DOWN:
            samples = -self._generate_sawtooth_vectorized(phases)
        elif self.wave_shape == WaveShape.TRIANGLE:
            # Triangle modulation falls back to sequential
            samples = self._generate_triangle_vectorized(n)
            final_phase = self._phase_normalized
        else:
            samples = np.zeros(n, dtype=np.float64)

        # Apply wave range (no amplitude here — ModulatedOscillator handles it)
        samples = self._apply_wave_range_values(samples)

        return samples.astype(np.float32), {"phase": float(final_phase)}

    def commit_modulated_phase_state(self, state: dict[str, float]) -> None:
        """Commit phase state produced by ``render_modulated_waveform``.

        Args:
            state: Dict with "phase" key containing normalized phase [0, 1)
        """
        self._phase_normalized = float(state["phase"]) % 1.0
        self._p = self._phase_normalized * 360.0


# Convenience functions
def generate_sine(
    frequency: float,
    duration: float,
    sample_rate: int | float = 44100,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a sine wave."""
    n_samples = int(duration * sample_rate)
    osc = PolyBLEPOscillator(
        frequency, amplitude, None, 0, sample_rate, wave_shape=WaveShape.SINE
    )
    return osc.get_samples(n_samples)


def generate_square(
    frequency: float,
    duration: float,
    sample_rate: int | float = 44100,
    amplitude: float = 1.0,
    pulsewidth: float = 0.5,
) -> np.ndarray:
    """Generate an antialiased square wave with variable pulsewidth.

    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (linear scale)
        pulsewidth: Pulse width (0.0 to 1.0), default 0.5 for 50% duty cycle

    Returns:
        Array of samples
    """
    n_samples = int(duration * sample_rate)
    osc = PolyBLEPOscillator(
        frequency,
        amplitude,
        None,
        0,
        sample_rate,
        wave_shape=WaveShape.SQUARE,
        pulsewidth=pulsewidth,
    )
    return osc.get_samples(n_samples)


def generate_sawtooth(
    frequency: float,
    duration: float,
    sample_rate: int | float = 44100,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate an antialiased sawtooth wave."""
    n_samples = int(duration * sample_rate)
    osc = PolyBLEPOscillator(
        frequency, amplitude, None, 0, sample_rate, wave_shape=WaveShape.SAWTOOTH_UP
    )
    return osc.get_samples(n_samples)


def generate_triangle(
    frequency: float,
    duration: float,
    sample_rate: int | float = 44100,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a triangle wave."""
    n_samples = int(duration * sample_rate)
    osc = PolyBLEPOscillator(
        frequency, amplitude, None, 0, sample_rate, wave_shape=WaveShape.TRIANGLE
    )
    return osc.get_samples(n_samples)
