"""Modulated oscillators for expressive synthesis.

This module provides the ModulatedOscillator class, which combines basic
oscillators with modulators (like ADSR envelopes) to create time-varying
synthesis. This is a fundamental technique in subtractive synthesis for
creating natural-sounding, expressive audio.

Classes:
    ModulatedOscillator: Combines an oscillator with modulators for dynamic synthesis.

Example:
    >>> from src.engine import SineOscillator, ADSREnvelope
    >>>
    >>> # Create oscillator and envelope
    >>> osc = SineOscillator(frequency=440, amplitude=1.0)
    >>> env = ADSREnvelope(
    ...     attack_duration=0.1,
    ...     decay_duration=0.2,
    ...     sustain_level=0.7,
    ...     release_duration=0.3
    ... )
    >>>
    >>> # Combine them with amplitude modulation
    >>> mod_osc = ModulatedOscillator(
    ...     osc,
    ...     env,
    ...     amp_mod=lambda base_amp, env_val: base_amp * env_val
    ... )
    >>>
    >>> # Generate modulated audio
    >>> samples = mod_osc.get_samples(1000)
    >>>
    >>> # Trigger note release
    >>> mod_osc.trigger_release()

Modulation Types:
    - amp_mod: Modulate oscillator amplitude (common for ADSR envelopes)
    - freq_mod: Modulate oscillator frequency (vibrato, FM synthesis)
    - phase_mod: Modulate oscillator phase (phase modulation synthesis)

Note:
    ModulatedOscillator maintains the ended state of its modulators,
    making it suitable for voice management in polyphonic synthesizers.
"""

import numpy as np

from src.engine.audio_component import ComponentDescriptor, Generator
from src.engine.audio_component_registry import register_component, ComponentCategory
from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.oscillator import Oscillator


@register_component()
class ModulatedOscillator(Generator):
    """Creates a modulated oscillator by using a plain oscillator along with modulators,
    the `[parameter]_mod` functions of the signature (float, float) -> float are used
    to decide the method of modulation.

    Has `.trigger_release()` implemented to trigger the release stage of the
    modulators. Similarly, has `.ended` to indicate the end of signal generator of the
    modulators if the generation is meant to be finite.

    The ModulatedOscillator internal values are set by calling __init__ and then
    __next__ to generate the sequence of values.
    """

    descriptor = ComponentDescriptor(
        name="ModulatedOscillator",
        category=ComponentCategory.OSCILLATOR,
        description="Oscillator with modulation support (amplitude, frequency, phase)",
        tags=["oscillator", "modulated", "advanced"],
        config_params=["gain_db", "frequency", "phase"],  # Forwarded to underlying oscillator
    )

    def __init__(
        self, oscillator, *modulators, amp_mod=None, freq_mod=None, phase_mod=None
    ):
        """Initialize the ModulatedOscillator.

        Args:
            oscillator : Instance of `Oscillator`, a component that generates a
                periodic signal of a given frequency.

            modulators : Components that generate a signal that can be used to modify
                the  internal parameters of the oscillator. The number of modulators
                should be between 1 and 3. If only 1 is passed then the same modulator
                is used for all the parameters.

            amp_mod : Any function that takes in the initial oscillator amplitude
                value and the modulator value and returns the modified value.
                If set the first modulator is used for the values.

            freq_mod : Any function that takes in the initial oscillator frequency
                value and the modulator value and returns the modified value.
                If set the second modulator of the last modulator is used for the
                values.

            phase_mod : Any function that takes in the initial oscillator phase
                value and the modulator value and returns the modified value.
                If set the third modulator of the last modulator is used for the values.
        """

        super().__init__()
        if not isinstance(oscillator, Oscillator):
            raise TypeError(
                f"Oscillator should be an instance of Oscillator. "
                f"Given: {type(oscillator)}"
            )

        self.oscillator = oscillator
        self.modulators = modulators

        self.amp_mod = amp_mod
        self.freq_mod = freq_mod
        self.phase_mod = phase_mod
        self._modulators_count = len(modulators)

        # Initialize all components to avoid issues when get_samples is called
        # without reset
        iter(self)

    def __iter__(self):
        iter(self.oscillator)
        [iter(modulator) for modulator in self.modulators]
        return self

    def _modulate(self, mod_vals):
        if self.amp_mod is not None:
            new_amp = self.amp_mod(self.oscillator.init_amp, mod_vals[0])
            self.oscillator.amplitude = new_amp

        if self.freq_mod is not None:
            mod_val = mod_vals[1 if self._modulators_count == 2 else 0]
            new_freq = self.freq_mod(self.oscillator.init_freq, mod_val)
            self.oscillator.frequency = new_freq

        if self.phase_mod is not None:
            mod_val = mod_vals[2 if self._modulators_count == 3 else -1]
            new_phase = self.phase_mod(self.oscillator.init_phase, mod_val)
            self.oscillator.phase = new_phase

    def trigger_release(self):
        tr = "trigger_release"
        for modulator in self.modulators:
            if hasattr(modulator, tr):
                modulator.trigger_release()

        if isinstance(self.oscillator, ModulatedOscillator) and hasattr(
            self.oscillator, tr
        ):
            self.oscillator.trigger_release()

    @property
    def ended(self):
        e = "ended"
        ended = []
        for modulator in self.modulators:
            if hasattr(modulator, e):
                ended.append(modulator.ended)

        if isinstance(self.oscillator, ModulatedOscillator) and hasattr(
            self.oscillator, e
        ):
            ended.append(self.oscillator.ended)
        return all(ended)

    # Parameter forwarding for hot-swap support
    @property
    def gain_db(self) -> float:
        """Get gain_db from underlying oscillator."""
        return self.oscillator.gain_db

    @gain_db.setter
    def gain_db(self, value: float):
        """Set gain_db on underlying oscillator."""
        self.oscillator.gain_db = value

    @property
    def frequency(self) -> float:
        """Get frequency from underlying oscillator."""
        return self.oscillator.frequency

    @frequency.setter
    def frequency(self, value: float):
        """Set frequency on underlying oscillator."""
        self.oscillator.frequency = value

    @property
    def phase(self) -> float:
        """Get phase from underlying oscillator."""
        return self.oscillator.phase

    @phase.setter
    def phase(self, value: float):
        """Set phase on underlying oscillator."""
        self.oscillator.phase = value

    def __next__(self):
        mod_vals = [next(modulator) for modulator in self.modulators]
        self._modulate(mod_vals)
        return next(self.oscillator)

    def get_samples_iterator(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False
    ) -> np.ndarray:
        """Generate n samples using Python iterator (slower but flexible).

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            reset: If True, reset the modulated oscillator to initial state.

        Returns:
            list[float]: List of `n` consecutive samples produced by calling
            `next(self)` repeatedly.
        """
        if reset:
            iter(self)
        return np.array([next(self) for _ in range(n)], dtype=np.float32)

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Generate n samples using fully vectorized operations.

        This is a high-performance implementation that uses NumPy vectorization
        and phase accumulation to generate all samples at once, avoiding the
        Python loop overhead. Expected speedup: 20-40x compared to iterator approach.

        Args:
            n: Number of samples to produce.

        Returns:
            np.ndarray: Array of `n` consecutive samples.

        Note:
            This uses phase accumulation with time-varying frequency, allowing
            true vectorization even when frequency/amplitude change per sample.

        Performance:
            - Iterator approach: ~258K samples/sec
            - Vectorized approach: ~5-10M samples/sec (20-40x faster)
        """
        # Import oscillator types for type checking
        from src.engine.oscillator import (
            SineOscillator,
            SquareOscillator,
            TriangleOscillator,
            SawtoothOscillator,
        )

        # Step 1: Generate all modulator values in bulk (vectorized)
        mod_arrays = []
        for modulator in self.modulators:
            if hasattr(modulator, "get_samples"):
                mod_vals = modulator.get_samples(n, reset=False, mode="vectorized")
            else:
                # Fallback to iterator for modulators without get_samples
                mod_vals = np.array(
                    [next(modulator) for _ in range(n)], dtype=np.float32
                )
            mod_arrays.append(mod_vals)

        # Step 2: Compute modulated parameters for ALL samples at once

        # Get base values
        base_freq = self.oscillator.init_freq
        base_amp = self.oscillator.init_amp
        sample_rate = self.oscillator.sample_rate

        # Compute frequencies for all samples (vectorized)
        if self.freq_mod is not None:
            mod_idx = 1 if self._modulators_count == 2 else 0
            mod_vals = mod_arrays[mod_idx]

            # Check if freq_mod can be vectorized
            try:
                # Try vectorized call
                freqs = self.freq_mod(np.full(n, base_freq, dtype=np.float32), mod_vals)
            except (TypeError, ValueError):
                # Fallback to element-wise if function doesn't support arrays
                freqs = np.array(
                    [self.freq_mod(base_freq, mod_vals[i]) for i in range(n)],
                    dtype=np.float32,
                )
        else:
            # Constant frequency
            freqs = np.full(n, base_freq, dtype=np.float32)

        # Compute amplitudes for all samples (vectorized)
        if self.amp_mod is not None:
            mod_vals = mod_arrays[0]

            # Check if amp_mod can be vectorized
            try:
                # Try vectorized call
                amps = self.amp_mod(np.full(n, base_amp, dtype=np.float32), mod_vals)
            except (TypeError, ValueError):
                # Fallback to element-wise if function doesn't support arrays
                amps = np.array(
                    [self.amp_mod(base_amp, mod_vals[i]) for i in range(n)],
                    dtype=np.float32,
                )
        else:
            # Constant amplitude
            amps = np.full(n, base_amp, dtype=np.float32)

        # Step 3: Phase accumulation with time-varying frequency
        # This is the KEY optimization - accumulate phase changes
        phase_increments = 2.0 * np.pi * freqs / sample_rate
        phases = np.cumsum(phase_increments) + self.oscillator._p


        # Step 4: Generate waveform based on oscillator type
        # Use optimized NumPy operations for each waveform

        if isinstance(self.oscillator, SineOscillator):
            # Sine wave: simple sin function
            waveform = np.sin(phases)

        elif isinstance(self.oscillator, SquareOscillator):
            # Square wave: sign of sin
            waveform = np.sign(np.sin(phases))

        elif isinstance(self.oscillator, TriangleOscillator):
            # Triangle wave: 2*arcsin(sin(x))/π
            waveform = (2.0 / np.pi) * np.arcsin(np.sin(phases))

        elif isinstance(self.oscillator, SawtoothOscillator):
            # Sawtooth wave: phase modulo 2π, normalized to [-1, 1]
            waveform = 2.0 * ((phases / (2.0 * np.pi)) % 1.0) - 1.0

        else:
            # Unknown oscillator type - fall back to sample-by-sample
            # This maintains compatibility with custom oscillators
            return self._get_samples_fallback(n, mod_arrays)

        # Step 5: Apply amplitude modulation
        samples = waveform * amps

        # Step 6: Update oscillator state for continuous phase
        # This ensures phase continuity between calls
        self.oscillator._p = phases[-1] % (2.0 * np.pi)

        return samples.astype(np.float32)

    def _get_samples_fallback(self, n: int, mod_arrays: list) -> np.ndarray:
        """Fallback to sample-by-sample generation for unknown oscillator types.

        This maintains compatibility with custom oscillators that aren't
        recognized by the vectorized implementation.

        Args:
            n: Number of samples
            mod_arrays: Pre-computed modulator arrays

        Returns:
            np.ndarray: Generated samples
        """
        samples = np.zeros(n, dtype=np.float32)

        for i in range(n):
            # Get modulator values for this sample
            mod_vals = [mod_arr[i] for mod_arr in mod_arrays]

            # Apply modulation
            self._modulate(mod_vals)

            # Generate sample
            samples[i] = next(self.oscillator)

        return samples

    def get_samples(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False, mode: str = "auto"
    ) -> np.ndarray:
        """Generate n samples using the specified method.

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            reset: If True, reset the modulated oscillator to initial state.
            mode: Generation mode. Options:
                - "auto": Automatically choose best method (vectorized for n >= 512)
                - "iterator": Use Python iterator (returns list)
                - "vectorized": Convert to NumPy array (returns ndarray)

        Returns:
            np.ndarray: Generated samples as NumPy array.

        Raises:
            ValueError: If mode is not one of "auto", "iterator", or "vectorized".

        Examples:
            >>> from src.engine import SineOscillator, ADSREnvelope, ModulatedOscillator
            >>>
            >>> osc = SineOscillator(440)
            >>> env = ADSREnvelope(0.1, 0.2, 0.7, 0.3)
            >>> mod_osc = ModulatedOscillator(osc, env, amp_mod=lambda a, e: a * e)
            >>> samples1 = mod_osc.get_samples(1000)  # Auto mode
            >>> samples2 = mod_osc.get_samples(100, mode="iterator", reset=True)
        """
        if mode not in ("auto", "iterator", "vectorized"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'auto', 'iterator', or 'vectorized'."
            )

        if mode == "auto":
            mode = "vectorized" if n >= 512 else "iterator"

        if mode == "iterator":
            samples_list = self.get_samples_iterator(n, reset=reset)
            return np.array(samples_list, dtype=np.float32)

        # mode == "vectorized"
        if reset:
            iter(self)
        return self.get_samples_vectorized(n)
