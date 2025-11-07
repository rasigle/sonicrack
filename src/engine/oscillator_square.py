"""Square wave generation strategies.

This module provides different algorithms for generating square waves,
from ideal (aliased) to bandlimited (antialiased) versions.

The strategy pattern allows easy switching between formulations while
maintaining clean, testable code.
"""
import logging
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.oscillator import SineOscillator
from src.engine.audio_component_registry import (
    register_component,
    ComponentDescriptor,
    ComponentCategory,
)
from src.utils.utils import track_provided_args, filter_provided_args

# Type alias for square wave modes
SquareWaveMode = Literal["ideal", "ideal_smooth","soft"]


class SquareWaveStrategy(ABC):
    """Abstract base class for square wave generation strategies.

    All strategies must implement both sample-by-sample and vectorized generation.
    """

    @abstractmethod
    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        """Generate a single square wave sample.

        Args:
            phase: Current phase in radians (0 to 2π)
            pulsewidth_threshold: Phase threshold for pulse width (0 to 2π)
            low_value: Value when phase >= threshold
            high_value: Value when phase < threshold

        Returns:
            Single sample value
        """
        pass

    @abstractmethod
    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        """Generate multiple square wave samples (vectorized).

        Args:
            phases: Array of phases in radians (0 to 2π)
            pulsewidth_threshold: Phase threshold for pulse width (0 to 2π)
            low_value: Value when phase >= threshold
            high_value: Value when phase < threshold

        Returns:
            Array of sample values
        """
        pass


class IdealSquareStrategy(SquareWaveStrategy):
    """Ideal (aliased) square wave - instant transitions.

    This is the traditional square wave with instantaneous transitions,
    which creates aliasing at high frequencies but has minimal CPU overhead.

    Best for: Low frequencies, retro sounds, CPU efficiency
    """

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        return high_value if phase < pulsewidth_threshold else low_value

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        return np.where(phases < pulsewidth_threshold, high_value, low_value).astype(np.float32)


class IdealSquareStrategySmoothing(SquareWaveStrategy):
    """Ideal (aliased) square wave with amplitude smoothing.

    This variant of the ideal square wave includes amplitude smoothing
    to reduce clicks when changing amplitude or other parameters.

    Best for: Low frequencies, retro sounds, CPU efficiency,
    with reduced clicks on parameter changes.
    """

    def __init__(self, smoothing_time_ms: float = 5.0, sample_rate: float = 44100):
        """Initialize ideal square strategy with amplitude smoothing.

        Args:
            smoothing_time_ms: Time in milliseconds for amplitude transitions
            sample_rate: Sample rate for calculating smoothing samples
        """
        self.smoothing_time_ms = smoothing_time_ms
        self.sample_rate = sample_rate

        # Smoothing state
        self._current_amplitude = 1.0
        self._target_amplitude = 1.0
        self._smoothing_samples_remaining = 0

    def set_amplitude(self, amplitude: float):
        """Set target amplitude with smoothing.

        Args:
            amplitude: New target amplitude
        """
        if abs(amplitude - self._current_amplitude) > 0.001:
            self._target_amplitude = amplitude
            self._smoothing_samples_remaining = int(
                self.smoothing_time_ms * self.sample_rate / 1000
            )

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        # Generate ideal square value
        val = high_value if phase < pulsewidth_threshold else low_value

        # Apply smoothing if active
        if self._smoothing_samples_remaining > 0:
            # Calculate smoothing factor for this sample
            progress = 1.0 - (self._smoothing_samples_remaining /
                            (self.smoothing_time_ms * self.sample_rate / 1000))
            current_amp = (self._current_amplitude +
                          (self._target_amplitude - self._current_amplitude) * progress)

            self._smoothing_samples_remaining -= 1
            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude

            return val * current_amp

        return val * self._current_amplitude

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        n = len(phases)

        # Generate ideal square wave
        val = np.where(phases < pulsewidth_threshold, high_value, low_value)

        # Apply amplitude with smoothing if transitioning (prevents clicks!)
        if self._smoothing_samples_remaining > 0:
            # Calculate how many samples to smooth in this buffer
            smooth_count = min(n, self._smoothing_samples_remaining)

            # Create smooth amplitude envelope (linear ramp)
            amp_envelope = np.linspace(
                self._current_amplitude, self._target_amplitude, smooth_count
            )

            # Apply smoothed amplitude to first part
            samples = np.zeros(n, dtype=np.float32)
            samples[:smooth_count] = val[:smooth_count] * amp_envelope

            # Apply target amplitude to rest (if any)
            if smooth_count < n:
                samples[smooth_count:] = val[smooth_count:] * self._target_amplitude

            # Update state
            self._smoothing_samples_remaining -= smooth_count
            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude

            return samples
        else:
            # No smoothing needed - just apply current amplitude
            return (val * self._current_amplitude).astype(np.float32)


class SoftSquareStrategy(SquareWaveStrategy):
    """Soft/clipped square wave with smooth transitions.

    Uses tanh() or other soft clipping to create smooth transitions
    instead of instant jumps. Reduces aliasing while maintaining
    square-ish character.

    Best for: Warmer, smoother sounds; analog-style synthesis
    """

    def __init__(self, smoothness: float = 10.0):
        """Initialize soft square strategy.

        Args:
            smoothness: Controls transition steepness (higher = sharper)
                Range: 1.0 (very soft) to 100.0 (nearly ideal)
                Default: 10.0 (good balance)
        """
        self.smoothness = smoothness

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        # Create smooth square wave using tanh-based transitions
        # Standard square: HIGH from 0 to pulsewidth_threshold, LOW after

        # For a smooth square wave, we want:
        # - Phase < pulsewidth_threshold: output = high_value
        # - Phase >= pulsewidth_threshold: output = low_value
        # - Smooth transitions using tanh

        # Create a smooth step function that transitions from high to low
        # at pulsewidth_threshold
        # tanh maps: large negative → -1, large positive → +1
        # We want: before threshold → +1 (high), after threshold → -1 (low)

        dist_from_threshold = phase - pulsewidth_threshold
        # Negate to get correct polarity: before threshold gives negative (→ +1 after negation)
        smooth_step = np.tanh(-dist_from_threshold * self.smoothness)

        # smooth_step is now: +1 before threshold, -1 after threshold
        # Map from [-1, 1] to [low_value, high_value]
        # +1 should map to high_value, -1 should map to low_value
        return low_value + (high_value - low_value) * (smooth_step + 1) / 2

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        # Create smooth square wave using tanh-based transitions
        # Standard square: HIGH from 0 to pulsewidth_threshold, LOW after

        # Create a smooth step function that transitions from high to low
        # at pulsewidth_threshold
        dist_from_threshold = phases - pulsewidth_threshold
        smooth_step = np.tanh(-dist_from_threshold * self.smoothness)

        # smooth_step is: +1 before threshold, -1 after threshold
        # Map from [-1, 1] to [low_value, high_value]
        return low_value + (high_value - low_value) * (smooth_step + 1) / 2


class SquareWaveFactory:
    """Factory for creating square wave strategy instances.

    Provides a clean interface for switching between different
    square wave generation algorithms.
    """

    _strategies = {
        "ideal": IdealSquareStrategy,
        "ideal_smooth": IdealSquareStrategySmoothing,
        "soft": SoftSquareStrategy,
    }

    @classmethod
    def create(
        cls,
        mode: SquareWaveMode = "ideal",
        **kwargs
    ) -> SquareWaveStrategy:
        """Create a square wave strategy instance.

        Args:
            mode: Type of square wave generation
                - "ideal": Traditional square wave (instant transitions)
                - "ideal_smooth": Ideal square with amplitude smoothing (reduces clicks)
                - "soft": Smooth transitions using tanh()
            **kwargs: Strategy-specific parameters
                For "ideal_smooth": smoothing_time_ms (default 5.0), sample_rate
                For "soft": smoothness (1.0-100.0)

        Returns:
            SquareWaveStrategy instance

        Raises:
            ValueError: If mode is not recognized

        Examples:
            >>> # Ideal square wave
            >>> strategy = SquareWaveFactory.create("ideal")

            >>> # Ideal square with amplitude smoothing
            >>> strategy = SquareWaveFactory.create("ideal_smooth",
            ...                                      smoothing_time_ms=10.0,
            ...                                      sample_rate=48000)

            >>> # Soft square with custom smoothness
            >>> strategy = SquareWaveFactory.create("soft", smoothness=20.0)
        """
        if mode not in cls._strategies:
            raise ValueError(
                f"Unknown square wave mode: {mode}. "
                f"Available modes: {list(cls._strategies.keys())}"
            )

        logging.debug(f"Creating square wave strategy with mode: {mode}")
        strategy_class = cls._strategies[mode]
        return strategy_class(**kwargs)

    @classmethod
    def get_available_modes(cls) -> list[str]:
        """Get list of available square wave modes.

        Returns:
            List of mode names
        """
        return list(cls._strategies.keys())

    @classmethod
    def register_strategy(
        cls,
        name: str,
        strategy_class: type[SquareWaveStrategy]
    ):
        """Register a custom square wave strategy.

        Args:
            name: Name for the strategy
            strategy_class: Strategy class (must inherit from SquareWaveStrategy)

        Raises:
            TypeError: If strategy_class doesn't inherit from SquareWaveStrategy
        """
        if not issubclass(strategy_class, SquareWaveStrategy):
            raise TypeError(
                f"{strategy_class} must inherit from SquareWaveStrategy"
            )
        cls._strategies[name] = strategy_class


@register_component()
class SquareOscillator(SineOscillator):
    """Square/Pulse wave generator with variable pulse width.

    The square wave uses phase comparison to generate pulses with configurable
    duty cycle. When the phase is below the pulsewidth threshold, the output is
    `wave_range[1]`, otherwise `wave_range[0]`.

    Pulse width is specified as a fraction of the period (0.0 to 1.0):
    - 0.5 = traditional square wave (50% duty cycle)
    - 0.1 = narrow pulse (10% high, 90% low)
    - 0.9 = wide pulse (90% high, 10% low)
    """

    descriptor = ComponentDescriptor(
        name="Square",
        category=ComponentCategory.OSCILLATOR,
        description="Square/Pulse wave oscillator with variable pulse width",
        tags=["basic", "oscillator", "square", "pulse"],
        fluent_api_name="square",
        config_params=[
            "frequency",
            "gain_db",
            "amplitude",
            "phase",
            "sample_rate",
            "wave_range",
            "pulsewidth",
        ],
    )

    @track_provided_args
    def __init__(
        self,
        frequency: float = 440,
        amplitude: float = 1.0,
        gain_db: float | None = DEFAULT_GAIN_DB,
        phase: float = 0.0,
        sample_rate: int | float = DEFAULT_SAMPLE_RATE,
        wave_range: tuple[float, float] = (-1, 1),
        pulsewidth: float = 0.5,
        mode: SquareWaveMode = "ideal",
        **mode_kwargs,
    ):
        """Construct a square/pulse oscillator with variable pulse width.

        Args:
            frequency: Initial frequency in Hz.
            amplitude: Linear amplitude (0.0 to 1.0+). Default: 1.0
                Note: Ignored if gain_db is specified.
            gain_db: Gain in decibels. Default: -20.0 (safe for mixing)
                Overrides amplitude if provided.
                Set to None to use amplitude parameter instead.
            phase: Initial phase in degrees. Defaults to 0.0.
            sample_rate: Samples per second. Defaults to `DEFAULT_SAMPLE_RATE`.
            wave_range: Tuple specifying value range (min, max) of raw waveform before
                amplitude scaling. Defaults to (-1, 1).
                Advanced feature - most users should leave as default.
            pulsewidth: Pulse width as fraction of period (0.0 to 1.0).
                0.5 = traditional square wave (50% duty cycle)
                0.1 = narrow pulse (10% high, 90% low)
                0.9 = wide pulse (90% high, 10% low)
                Default: 0.5
            mode: Square wave generation algorithm. Options:
                - "ideal": Traditional square wave (instant transitions, aliasing)
                - "ideal_smooth": Ideal square with amplitude smoothing (reduces clicks)
                - "soft": Smooth transitions using tanh() (warm, reduced aliasing)
                Default: "ideal"
            **mode_kwargs: Algorithm-specific parameters:
                - For "ideal_smooth": smoothing_time_ms (default 5.0), sample_rate
                - For "soft": smoothness (1.0-100.0, default 10.0)

        Examples:
            >>> # Standard square wave (50% duty cycle, ideal algorithm)
            >>> osc = SquareOscillator(frequency=440)

            >>> # Soft square with custom smoothness
            >>> osc = SquareOscillator(frequency=220, mode="soft", smoothness=20.0)

        """
        # Filter to pass only arguments explicitly provided by user
        kwargs = filter_provided_args(
            self._provided_args,  # noqa
            frequency=frequency,
            amplitude=amplitude,
            gain_db=gain_db,
            phase=phase,
            wave_range=wave_range,
        )
        super().__init__(**kwargs)

        # Validate pulsewidth
        if not 0.0 <= pulsewidth <= 1.0:
            raise ValueError(
                f"pulsewidth must be between 0.0 and 1.0, got {pulsewidth}"
            )

        self._pulsewidth = pulsewidth
        self._mode = mode

        # Convert pulsewidth to phase threshold
        # pulsewidth of 0.5 = threshold of π (50% duty cycle)
        # pulsewidth of 0.25 = threshold of π/2 (25% of cycle is high)
        self._pulsewidth_threshold = pulsewidth * 2 * np.pi

        self._strategy = SquareWaveFactory.create(mode, **mode_kwargs)
        self._mode_kwargs = mode_kwargs


    @property
    def pulsewidth(self) -> float:
        """float: Current pulse width (0.0 to 1.0).

        Returns the duty cycle as a fraction of the period.
        0.5 = traditional square wave (50% duty cycle)
        0.1 = narrow pulse (10% high)
        0.9 = wide pulse (90% high)
        """
        return self._pulsewidth

    @pulsewidth.setter
    def pulsewidth(self, value: float):
        """Set pulse width and update internal threshold.

        Args:
            value: Pulse width between 0.0 and 1.0

        Raises:
            ValueError: If value is outside the range [0.0, 1.0]
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"pulsewidth must be between 0.0 and 1.0, got {value}")
        self._pulsewidth = value
        self._pulsewidth_threshold = value * 2 * np.pi

    @property
    def mode(self) -> SquareWaveMode:
        """str: Current square wave generation mode.

        Available modes:
        - "ideal": Traditional square wave (instant transitions)
        - "bandlimited": MinBLEP antialiased square wave
        - "soft": Smooth transitions using tanh()
        - "comparator": Sine comparator with hysteresis
        """
        return self._mode

    def set_mode(self, mode: SquareWaveMode, **mode_kwargs):
        """Change the square wave generation algorithm.

        This allows runtime switching between different square wave formulations
        without recreating the oscillator.

        Args:
            mode: New generation algorithm
            **mode_kwargs: Algorithm-specific parameters
                - For "soft": smoothness (1.0-100.0)

        Examples:
            >>> osc = SquareOscillator(frequency=440, mode="ideal")
            >>> osc.set_mode("soft", smoothness=15.0)  # Soft with custom smoothness
            >>> osc.set_mode("comparator", hysteresis=0.03)  # Comparator mode
        """
        # Update mode
        self._mode = mode
        self._mode_kwargs = mode_kwargs

        # Create new strategy
        self._strategy = SquareWaveFactory.create(mode, **mode_kwargs)

    @classmethod
    def get_available_modes(cls) -> list[str]:
        """Get list of available square wave generation modes.

        Returns:
            List of mode names

        Example:
            >>> modes = SquareOscillator.get_available_modes()
            >>> print(modes)
            ['ideal', 'soft']
        """
        return SquareWaveFactory.get_available_modes()

    def __next__(self):
        """Return next square sample and advance internal phase.

        Returns:
            float: Next square sample scaled by amplitude.
        """
        # Get current phase (wrapped to 0-2π)
        current_phase = (self._i + self._p) % (2 * np.pi)

        # Use strategy to generate square wave value
        val = self._strategy.generate_sample(
            phase=current_phase,
            pulsewidth_threshold=self._pulsewidth_threshold,
            low_value=self._wave_range[0],
            high_value=self._wave_range[1],
        )

        # Optimized phase wrapping: only wrap when needed (10-15% faster)
        self._i += self._step
        if self._i >= 2 * np.pi:
            self._i -= 2 * np.pi

        return val * self._a

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using NumPy vectorization.

        Returns:
            np.ndarray: Array of square samples (float32).
        """
        # Generate phase values for all samples (optimized: pre-add phase offset)
        phases = (self._i + self._p) + self._step * np.arange(n)

        # Wrap phases to 0-2π for pulse width comparison
        wrapped_phases = phases % (2 * np.pi)

        # Use strategy to generate square wave values
        val = self._strategy.generate_samples(
            phases=wrapped_phases,
            pulsewidth_threshold=self._pulsewidth_threshold,
            low_value=self._wave_range[0],
            high_value=self._wave_range[1],
        )

        samples = val * self._a

        # Update internal state with phase wrapping
        self._i = (self._i + self._step * n) % (2 * np.pi)

        return samples.astype(np.float32)
