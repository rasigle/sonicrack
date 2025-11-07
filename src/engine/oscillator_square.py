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
SquareWaveMode = Literal["ideal", "bandlimited", "soft", "comparator"]


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


class BandlimitedSquareStrategy(SquareWaveStrategy):
    """Bandlimited square wave using MinBLEP (Minimum-phase Band-Limited Step).

    Reduces aliasing by applying a bandlimited step function at transitions.
    This creates a cleaner, more professional sound at high frequencies.

    Best for: High-quality synthesis, anti-aliasing required
    Note: Higher CPU cost than ideal square
    """

    def __init__(self, sample_rate: float = 44100):
        """Initialize bandlimited strategy.

        Args:
            sample_rate: Sample rate in Hz (for BLEP kernel generation)
        """
        self.sample_rate = sample_rate
        # MinBLEP kernel parameters
        self.blep_length = 64  # Samples in BLEP kernel
        self.oversampling = 16  # BLEP table oversampling
        self._generate_blep_table()

        # State for edge detection
        self.last_phase = 0.0

    def _generate_blep_table(self):
        """Generate MinBLEP lookup table.

        The MinBLEP kernel is a minimum-phase bandlimited step function
        used to replace discontinuous transitions.
        """
        import sys
        print("[BLEP] Starting table generation...", file=sys.stderr, flush=True)

        # Create bandlimited step using sinc interpolation
        n = self.blep_length * self.oversampling
        print(f"[BLEP] n={n}", file=sys.stderr, flush=True)

        t = np.arange(n) / self.oversampling - self.blep_length / 2
        print(f"[BLEP] t created, len={len(t)}", file=sys.stderr, flush=True)

        # Sinc function with Blackman window
        sinc = np.sinc(t)
        print(f"[BLEP] sinc created", file=sys.stderr, flush=True)

        window = np.blackman(n)
        print(f"[BLEP] window created", file=sys.stderr, flush=True)

        # Integrate to get step function (BLEP is integral of BLAMP)
        blep = np.cumsum(sinc * window)
        print(f"[BLEP] cumsum done", file=sys.stderr, flush=True)

        blep = blep - blep[0]  # Start at 0
        blep = blep / blep[-1]  # End at 1
        print(f"[BLEP] normalized", file=sys.stderr, flush=True)

        # Ensure monotonically increasing and clamp to [0, 1]
        blep = np.maximum.accumulate(blep)  # Force monotonic increasing
        blep = np.clip(blep, 0, 1)  # Ensure [0, 1] range
        print(f"[BLEP] clamped", file=sys.stderr, flush=True)

        self.blep_table = blep.astype(np.float32)
        print(f"[BLEP] Table generation complete, len={len(self.blep_table)}", file=sys.stderr, flush=True)

    def _apply_blep(self, phase: float, output: float, last_phase: float) -> float:
        """Apply BLEP correction at phase discontinuities.

        Args:
            phase: Current phase
            output: Current output value
            last_phase: Previous phase

        Returns:
            Corrected output value
        """
        # Detect edge crossing (phase wraps or crosses pulsewidth threshold)
        # Simplified for single sample - full implementation would track edges
        return output  # Placeholder - full BLEP requires state tracking

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        # Ideal square as base
        ideal = high_value if phase < pulsewidth_threshold else low_value

        # Apply BLEP correction (simplified - full version needs edge buffer)
        corrected = self._apply_blep(phase, ideal, self.last_phase)
        self.last_phase = phase

        return corrected

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        # Generate ideal square wave (convert to float64 for BLEP corrections)
        output = np.where(phases < pulsewidth_threshold, high_value, low_value).astype(np.float64)

        # Detect transitions (where output changes)
        transitions = np.diff(output, prepend=output[0])
        transition_indices = np.where(transitions != 0)[0]

        # DEBUG: Log transition detection
        import sys
        if len(transition_indices) > 0:
            print(f"[BLEP DEBUG] Found {len(transition_indices)} transitions at indices: {transition_indices[:5]}", file=sys.stderr, flush=True)
            print(f"[BLEP DEBUG] Transition heights: {transitions[transition_indices][:5]}", file=sys.stderr, flush=True)
            print(f"[BLEP DEBUG] BLEP table length: {len(self.blep_table)}, downsampled: {len(self.blep_table[::self.oversampling])}", file=sys.stderr, flush=True)

        # Apply BLEP at each transition
        corrections_applied = 0
        for idx in transition_indices:
            # Apply BLEP kernel centered at transition
            blep_start = max(0, idx - self.blep_length // 2)
            blep_end = min(len(output), idx + self.blep_length // 2)
            blep_range = blep_end - blep_start

            if blep_range > 0 and blep_range <= self.blep_length:
                # Get downsampled BLEP table
                blep_downsampled = self.blep_table[::self.oversampling]

                # Calculate which part of the BLEP table to use
                # If we're at the start of the array, use latter part of BLEP
                # If we're at the end, use earlier part
                offset = idx - blep_start
                table_start = self.blep_length // 2 - offset
                table_end = table_start + blep_range

                # Clamp to valid range
                table_start = max(0, min(len(blep_downsampled) - blep_range, table_start))
                table_end = table_start + blep_range

                if table_end <= len(blep_downsampled):
                    kernel_slice = blep_downsampled[table_start:table_end]
                    transition_height = transitions[idx]

                    # Apply BLEP residual correction
                    # BLEP table goes from 0 to 1, representing the smooth step
                    # We center it at 0.5 and apply the full transition height
                    # This creates smooth transitions that may slightly overshoot,
                    # which is normal and necessary for proper bandlimiting
                    correction = (kernel_slice - 0.5) * transition_height

                    # DEBUG
                    if corrections_applied == 0:
                        print(f"[BLEP DEBUG] First correction at idx {idx}:", file=sys.stderr, flush=True)
                        print(f"  kernel_slice range: [{kernel_slice.min():.4f}, {kernel_slice.max():.4f}]", file=sys.stderr, flush=True)
                        print(f"  transition_height: {transition_height:.4f}", file=sys.stderr, flush=True)
                        print(f"  correction range: [{correction.min():.4f}, {correction.max():.4f}]", file=sys.stderr, flush=True)
                        print(f"  applying to output[{blep_start}:{blep_end}]", file=sys.stderr, flush=True)

                    output[blep_start:blep_end] += correction
                    corrections_applied += 1

        # DEBUG: Summary
        if len(transition_indices) > 0:
            print(f"[BLEP DEBUG] Applied {corrections_applied}/{len(transition_indices)} corrections", file=sys.stderr, flush=True)
            print(f"[BLEP DEBUG] Output after BLEP: min={output.min():.4f}, max={output.max():.4f}", file=sys.stderr, flush=True)

        # NOTE: Do NOT clamp here! The BLEP correction intentionally creates smooth
        # transitions that may slightly exceed the output range. Clamping would
        # flatten these transitions and destroy the antialiasing effect.
        # The small overshoot (typically < 10%) is normal for BLEP and represents
        # the Gibbs phenomenon being minimized through bandlimiting.

        return output



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


class ComparatorSquareStrategy(SquareWaveStrategy):
    """Comparator-based square wave (sine comparison with hysteresis).

    Simulates a hardware comparator circuit by comparing a sine wave
    against a threshold, with optional hysteresis for stability.

    Best for: Analog emulation, circuit modeling
    """

    def __init__(self, hysteresis: float = 0.01):
        """Initialize comparator strategy.

        Args:
            hysteresis: Hysteresis amount (0.0 to 0.1)
                0.0 = no hysteresis (instant switching)
                0.1 = 10% hysteresis (more stable)
        """
        self.hysteresis = hysteresis
        self.last_state = 1  # Start high

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        # Convert phase to sine value for comparison
        sine_value = np.sin(phase)

        # Calculate threshold from pulsewidth
        # pulsewidth_threshold is in phase (0-2π)
        # Convert to sine threshold
        threshold = np.cos(pulsewidth_threshold / 2)

        # Apply hysteresis
        if self.last_state == 1:  # Currently high
            switch_threshold = threshold - self.hysteresis
            if sine_value < switch_threshold:
                self.last_state = 0
        else:  # Currently low
            switch_threshold = threshold + self.hysteresis
            if sine_value > switch_threshold:
                self.last_state = 1

        return high_value if self.last_state == 1 else low_value

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        # Generate sine wave for comparison
        sine_values = np.sin(phases)

        # Calculate threshold from pulsewidth
        threshold = np.cos(pulsewidth_threshold / 2)

        # Apply hysteresis (simplified - full version would track state)
        # Convert to float32 to allow hysteresis modifications
        output = np.where(sine_values > threshold, high_value, low_value).astype(np.float32)

        # Apply hysteresis smoothing
        if self.hysteresis > 0:
            # Create transition regions
            lower_threshold = threshold - self.hysteresis
            upper_threshold = threshold + self.hysteresis

            # Find transition zones
            in_lower = (sine_values >= lower_threshold) & (sine_values <= threshold)
            in_upper = (sine_values >= threshold) & (sine_values <= upper_threshold)

            # Smooth transitions in hysteresis zones
            output[in_lower] = low_value + (high_value - low_value) * (
                (sine_values[in_lower] - lower_threshold) / (2 * self.hysteresis)
            )
            output[in_upper] = low_value + (high_value - low_value) * (
                0.5 + (sine_values[in_upper] - threshold) / (2 * self.hysteresis)
            )

        return output


class SquareWaveFactory:
    """Factory for creating square wave strategy instances.

    Provides a clean interface for switching between different
    square wave generation algorithms.
    """

    _strategies = {
        "ideal": IdealSquareStrategy,
        "bandlimited": BandlimitedSquareStrategy,
        "soft": SoftSquareStrategy,
        "comparator": ComparatorSquareStrategy,
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
                - "bandlimited": MinBLEP antialiased square wave
                - "soft": Smooth transitions using tanh()
                - "comparator": Sine comparator with hysteresis
            **kwargs: Strategy-specific parameters
                For "bandlimited": sample_rate
                For "soft": smoothness (1.0-100.0)
                For "comparator": hysteresis (0.0-0.1)

        Returns:
            SquareWaveStrategy instance

        Raises:
            ValueError: If mode is not recognized

        Examples:
            >>> # Ideal square wave
            >>> strategy = SquareWaveFactory.create("ideal")

            >>> # Bandlimited square
            >>> strategy = SquareWaveFactory.create("bandlimited", sample_rate=48000)

            >>> # Soft square with custom smoothness
            >>> strategy = SquareWaveFactory.create("soft", smoothness=20.0)

            >>> # Comparator with hysteresis
            >>> strategy = SquareWaveFactory.create("comparator", hysteresis=0.05)
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
                - "bandlimited": MinBLEP antialiased square wave (clean, CPU intensive)
                - "soft": Smooth transitions using tanh() (warm, reduced aliasing)
                - "comparator": Sine comparator with hysteresis (analog emulation)
                Default: "ideal"
            **mode_kwargs: Algorithm-specific parameters:
                - For "bandlimited": No additional parameters (uses sample_rate)
                - For "soft": smoothness (1.0-100.0, default 10.0)
                - For "comparator": hysteresis (0.0-0.1, default 0.01)

        Examples:
            >>> # Standard square wave (50% duty cycle, ideal algorithm)
            >>> osc = SquareOscillator(frequency=440)

            >>> # Narrow pulse with bandlimited algorithm (antialiased)
            >>> osc = SquareOscillator(frequency=880, pulsewidth=0.2, mode="bandlimited")

            >>> # Soft square with custom smoothness
            >>> osc = SquareOscillator(frequency=220, mode="soft", smoothness=20.0)

            >>> # Comparator mode with hysteresis (analog emulation)
            >>> osc = SquareOscillator(frequency=110, mode="comparator", hysteresis=0.05)
        """
        # Filter to pass only arguments explicitly provided by user
        kwargs = filter_provided_args(
            self._provided_args,  # noqa
            frequency=frequency,
            amplitude=amplitude,
            gain_db=gain_db,
            phase=phase,
            sample_rate=sample_rate,
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

        # Create square wave generation strategy
        # Pass sample_rate for bandlimited mode
        if mode == "bandlimited" and "sample_rate" not in mode_kwargs:
            mode_kwargs["sample_rate"] = sample_rate

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
                - For "bandlimited": (uses existing sample_rate)
                - For "soft": smoothness (1.0-100.0)
                - For "comparator": hysteresis (0.0-0.1)

        Examples:
            >>> osc = SquareOscillator(frequency=440, mode="ideal")
            >>> osc.set_mode("bandlimited")  # Switch to antialiased
            >>> osc.set_mode("soft", smoothness=15.0)  # Soft with custom smoothness
            >>> osc.set_mode("comparator", hysteresis=0.03)  # Comparator mode
        """
        # Update mode
        self._mode = mode
        self._mode_kwargs = mode_kwargs

        # Pass sample_rate for bandlimited mode if not provided
        if mode == "bandlimited" and "sample_rate" not in mode_kwargs:
            mode_kwargs["sample_rate"] = self._sample_rate

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
            ['ideal', 'bandlimited', 'soft', 'comparator']
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

        # # Apply amplitude with smoothing if transitioning (prevents clicks!)
        # if self._smoothing_samples_remaining > 0:
        #     # ...existing smoothing code...
        #     # Calculate how many samples to smooth in this buffer
        #     smooth_count = min(n, self._smoothing_samples_remaining)
        #
        #     # Create smooth amplitude envelope (linear ramp)
        #     amp_envelope = np.linspace(
        #         self._current_amplitude, self._target_amplitude, smooth_count
        #     )
        #
        #     # Apply smoothed amplitude to first part
        #     samples = np.zeros(n, dtype=np.float32)
        #     samples[:smooth_count] = val[:smooth_count] * amp_envelope
        #
        #     # Apply target amplitude to rest (if any)
        #     if smooth_count < n:
        #         samples[smooth_count:] = val[smooth_count:] * self._target_amplitude
        #
        #     # Update state
        #     self._smoothing_samples_remaining -= smooth_count
        #     if self._smoothing_samples_remaining <= 0:
        #         self._current_amplitude = self._target_amplitude
        # else:
        #     # No smoothing needed - direct multiplication

        samples = val * self._a

        # Update internal state with phase wrapping
        self._i = (self._i + self._step * n) % (2 * np.pi)

        return samples.astype(np.float32)
