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

from enum import Enum

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    ComponentDescriptor,
    Generator,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.core.registry import (
    ComponentCategory,
    register_component,
)
from src.engine.core.sample_mode import SampleMode, VALID_SAMPLE_MODES
from src.engine.generators.oscillators.oscillator import _derive_amplitude_from_init
from src.engine.utils.decorators import track_provided_args
from src.engine.utils.ramping import consume_linear_ramp, duration_ms_to_samples
from src.engine.utils.validation import validate_sample_count, validate_sample_rate


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
class PolyBLEPOscillator(Generator):
    """PolyBLEP oscillator with full Oscillator API compatibility.

    This is a drop-in replacement for standard oscillators that provides
    antialiased waveforms using the PolyBLEP algorithm. It matches the
    complete Oscillator API including:

    - gain_db and amplitude control
    - Phase control in degrees
    - Iterator interface (__next__, __iter__)
    - Vectorized generation (get_samples_vectorized)
    - Automatic mode selection (get_samples)
    - Wave range conversion
    - Amplitude smoothing (prevents clicks)
    - Phase continuity between modes

    Example:
        >>> # Drop-in replacement for any oscillator
        >>> osc = PolyBLEPOscillator(frequency=440, gain_db=-12,
        ...                          wave_shape=WaveShape.SQUARE)
        >>> samples = osc.get_samples(1000)
        >>>
        >>> # Works in iterator mode
        >>> for sample in osc:
        ...     next(sample)
        >>>
        >>> # All standard properties work
        >>> osc.gain_db = -6
        >>> osc.frequency = 880
        >>> osc.phase = 45.0
    """

    descriptor = ComponentDescriptor(
        name="PolyBLEPOscillator",
        category=ComponentCategory.OSCILLATOR,
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
            gain_db: Gain in decibels (default: -20.0 for safe mixing)
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
        sample_rate = validate_sample_rate(sample_rate)
        super().__init__(sample_rate)
        self.wave_shape = wave_shape

        # Validate pulsewidth
        if not 0.0 < pulsewidth < 1.0:
            raise ValueError(
                f"pulsewidth must be between 0.0 and 1.0, got {pulsewidth}"
            )
        self._pulsewidth = pulsewidth

        # Store initial values
        self._freq = frequency
        self._initial_amp = _derive_amplitude_from_init(
            self._provided_args,
            amplitude,
            gain_db,  # noqa
        )
        self._phase_degrees = phase
        self._wave_range = wave_range

        # Runtime properties
        self._f = frequency
        self._a = self._initial_amp
        self._p = phase

        # Phase in [0, 1) range (PolyBLEP uses normalized phase)
        self._phase_normalized = (phase % 360.0) / 360.0
        self._increment = (frequency / sample_rate) % 1.0

        # State for triangle wave integration
        self._triangle_accumulator = 0.0

        # Amplitude smoothing (prevents clicks)
        self._target_amplitude = self._initial_amp
        self._current_amplitude = self._initial_amp
        self._smoothing_samples_remaining = 0
        self._smoothing_samples_duration_total = duration_ms_to_samples(
            sample_rate, 10.0
        )

        # Wave range conversion
        self._update_range_conversion()

    @staticmethod
    def _derive_amplitude(amplitude: float, gain_db: float | None) -> float:
        """Derive amplitude from gain_db or amplitude parameter."""
        if gain_db is not None:
            return 10.0 ** (gain_db / 20.0)
        return amplitude

    def _update_range_conversion(self):
        """Update wave range conversion constants."""
        self._needs_range_conversion = self._wave_range != (-1, 1)
        if self._needs_range_conversion:
            self._range_scale = (self._wave_range[1] - self._wave_range[0]) / 2.0
            self._range_offset = (self._wave_range[1] + self._wave_range[0]) / 2.0
        else:
            self._range_scale = 1.0
            self._range_offset = 0.0

    def _apply_range_conversion(self, value: float) -> float:
        """Convert value from [-1, 1] to wave_range."""
        if self._needs_range_conversion:
            return value * self._range_scale + self._range_offset
        return value

    def _get_current_amplitude(self) -> float:
        """Get amplitude with smoothing."""
        if self._smoothing_samples_remaining > 0:
            envelope, self._current_amplitude, self._smoothing_samples_remaining = (
                consume_linear_ramp(
                    self._current_amplitude,
                    self._target_amplitude,
                    self._smoothing_samples_remaining,
                    1,
                )
            )
            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude
            return float(envelope[0])
        return self._current_amplitude

    # Properties matching Oscillator API
    @property
    def frequency(self) -> float:
        """Current oscillator frequency in Hz."""
        return self._f

    @frequency.setter
    def frequency(self, value: float):
        """Set oscillator frequency."""
        self._f = value
        self._increment = (value / self.sample_rate) % 1.0

    @property
    def amplitude(self) -> float:
        """Current amplitude (linear scale)."""
        return self._a

    @amplitude.setter
    def amplitude(self, value: float):
        """Set amplitude with smooth transition."""
        self._target_amplitude = value
        self._smoothing_samples_remaining = self._smoothing_samples_duration_total
        self._a = value

    @property
    def gain_db(self) -> float:
        """Current gain in decibels."""
        return 20.0 * np.log10(max(self._a, 1e-10))

    @gain_db.setter
    def gain_db(self, value: float):
        """Set gain in decibels."""
        new_amplitude = 10.0 ** (value / 20.0)
        self._target_amplitude = new_amplitude
        self._smoothing_samples_remaining = self._smoothing_samples_duration_total
        self._a = new_amplitude

    @property
    def phase(self) -> float:
        """Current phase in degrees."""
        return self._p

    @phase.setter
    def phase(self, value: float):
        """Set phase in degrees."""
        self._p = value
        self._phase_normalized = (value % 360.0) / 360.0

    @property
    def wave_range(self) -> tuple[float, float]:
        """Current wave range (min, max)."""
        return self._wave_range

    @wave_range.setter
    def wave_range(self, value: tuple[float, float]):
        """Set wave range and recompute conversion."""
        self._wave_range = value
        self._update_range_conversion()

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
    def pulsewidth(self, value: float):
        """Set pulse width for square wave.

        Args:
            value: Pulse width between 0.0 and 1.0

        Raises:
            ValueError: If value is outside valid range
        """
        if not 0.0 < value < 1.0:
            raise ValueError(f"pulsewidth must be between 0.0 and 1.0, got {value}")
        self._pulsewidth = value

    @property
    def init_freq(self) -> float:
        """Initial frequency supplied at construction."""
        return self._freq

    @property
    def init_amp(self) -> float:
        """Initial amplitude supplied at construction."""
        return self._initial_amp

    @property
    def init_phase(self) -> float:
        """Initial phase supplied at construction."""
        return self._phase_degrees

    # Iterator interface

    def __iter__(self):
        """Initialize iteration."""
        self.frequency = self._freq
        self.phase = self._phase_degrees
        self.amplitude = self._initial_amp
        return self

    def __next__(self) -> float:
        """Generate next sample (iterator interface)."""
        # Get current phase
        phase = self._phase_normalized

        # Generate waveform value
        if self.wave_shape == WaveShape.SINE:
            value = PolyBLEPWaveforms.sine(phase)
        elif self.wave_shape == WaveShape.SQUARE:
            value = PolyBLEPWaveforms.square(phase, self._increment, self._pulsewidth)
        elif self.wave_shape == WaveShape.SAWTOOTH_UP:
            value = PolyBLEPWaveforms.sawtooth(phase, self._increment)
        elif self.wave_shape == WaveShape.SAWTOOTH_DOWN:
            value = -PolyBLEPWaveforms.sawtooth(phase, self._increment)
        elif self.wave_shape == WaveShape.TRIANGLE:
            # Triangle via integration
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

        # Apply wave range conversion
        value = self._apply_range_conversion(value)

        # Apply amplitude (with smoothing)
        amp = self._get_current_amplitude()
        value *= amp

        # Advance phase
        self._phase_normalized = (self._phase_normalized + self._increment) % 1.0

        return float(value)

    # Vectorized generation

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using vectorization (high performance).

        Args:
            n: Number of samples to generate

        Returns:
            Array of n samples
        """
        n = validate_sample_count(n)
        # Generate phase array
        phases: np.ndarray | None = (
            self._phase_normalized + self._increment * np.arange(n)
        ) % 1.0

        # Generate waveform
        if self.wave_shape == WaveShape.SINE:
            assert phases is not None
            samples = np.sin(2.0 * np.pi * phases)
        elif self.wave_shape == WaveShape.SQUARE:
            assert phases is not None
            samples = self._generate_square_vectorized(phases)
        elif self.wave_shape == WaveShape.SAWTOOTH_UP:
            assert phases is not None
            samples = self._generate_sawtooth_vectorized(phases)
        elif self.wave_shape == WaveShape.SAWTOOTH_DOWN:
            assert phases is not None
            samples = -self._generate_sawtooth_vectorized(phases)
        elif self.wave_shape == WaveShape.TRIANGLE:
            samples = self._generate_triangle_vectorized(n)
            phases = None  # Triangle updates phase internally
        else:
            samples = np.zeros(n, dtype=np.float32)

        # Apply wave range conversion
        if self._needs_range_conversion:
            samples = samples * self._range_scale + self._range_offset

        # Apply amplitude with smoothing
        if self._smoothing_samples_remaining > 0:
            amp_envelope, self._current_amplitude, self._smoothing_samples_remaining = (
                consume_linear_ramp(
                    self._current_amplitude,
                    self._target_amplitude,
                    self._smoothing_samples_remaining,
                    n,
                )
            )
            samples *= amp_envelope
            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude
        else:
            samples *= self._current_amplitude

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
        # Generate naive square with specified pulsewidth
        output = np.where(phases < self._pulsewidth, -1.0, 1.0).astype(np.float64)

        # Apply PolyBLEP corrections at rising edge (phase = 0)
        mask1 = phases < self._increment
        if np.any(mask1):
            p = phases[mask1] / self._increment
            output[mask1] -= (p + p) - (p * p) - 1.0

        mask2 = phases > 1.0 - self._increment
        if np.any(mask2):
            p = (phases[mask2] - 1.0) / self._increment
            output[mask2] -= (p + p) + (p * p) + 1.0

        # Apply PolyBLEP corrections at falling edge (phase = pulsewidth)
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
        """Generate triangle via integration (sample-by-sample)."""
        samples = np.zeros(n, dtype=np.float32)

        for i in range(n):
            phase = self._phase_normalized
            square = PolyBLEPWaveforms.square(phase, self._increment, self._pulsewidth)

            self._triangle_accumulator = (
                self._increment * square
                + self._triangle_accumulator * (1.0 - (0.25 * self._increment))
            )

            samples[i] = self._triangle_accumulator * 4.0
            self._phase_normalized = (self._phase_normalized + self._increment) % 1.0

        return samples

    # Standard generation methods

    def get_samples_iterator(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False
    ) -> np.ndarray:
        """Generate n samples using iterator (slower but flexible).

        Args:
            n: Number of samples
            reset: Reset to initial state before generating

        Returns:
            Array of samples
        """
        n = validate_sample_count(n)
        if reset:
            iter(self)
        return np.array([next(self) for _ in range(n)], dtype=np.float32)

    def get_samples(
        self,
        n: int = DEFAULT_SAMPLE_RATE,
        reset: bool = False,
        mode: SampleMode = "auto",
    ) -> np.ndarray:
        """Generate n samples using specified method.

        Args:
            n: Number of samples
            reset: Reset to initial state before generating
            mode: Generation mode ("auto", "iterator", "vectorized")
                  auto: vectorized for n >= 512, iterator otherwise

        Returns:
            Array of samples

        Raises:
            ValueError: If mode is invalid
        """
        n = validate_sample_count(n)
        if mode not in VALID_SAMPLE_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'auto', 'iterator', or 'vectorized'."
            )

        if mode == "auto":
            mode = "vectorized" if n >= 512 else "iterator"

        if mode == "iterator":
            return self.get_samples_iterator(n, reset=reset)

        if reset:
            iter(self)
        return self.get_samples_vectorized(n)


# Convenience functions
def generate_sine(
    frequency: float,
    duration: float,
    sample_rate: float = 44100.0,
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
    sample_rate: float = 44100.0,
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
    sample_rate: float = 44100.0,
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
    sample_rate: float = 44100.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a triangle wave."""
    n_samples = int(duration * sample_rate)
    osc = PolyBLEPOscillator(
        frequency, amplitude, None, 0, sample_rate, wave_shape=WaveShape.TRIANGLE
    )
    return osc.get_samples(n_samples)
