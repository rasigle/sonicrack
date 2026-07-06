"""Modulated oscillators for expressive synthesis.

This module provides the ModulatedOscillator class, which combines basic
oscillators with modulators (like ADSR envelopes) to create time-varying
synthesis. This is a fundamental technique in subtractive synthesis for
creating natural-sounding, expressive audio.

Classes:
    ModulatedOscillator: Combines an oscillator with modulators for dynamic synthesis.
    ModulatedFrequency: Specialized class for frequency modulation (vibrato,
        FM synthesis).

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

Example - Frequency Modulation (Vibrato):
    >>> from src.engine import ModulatedFrequency
    >>>
    >>> # Carrier oscillator
    >>> carrier = SineOscillator(frequency=440, amplitude=0.5)
    >>>
    >>> # LFO for vibrato (5 Hz, ±50 Hz depth)
    >>> lfo = SineOscillator(frequency=5.0, amplitude=50.0)
    >>>
    >>> # Create frequency-modulated oscillator
    >>> vibrato = ModulatedFrequency(carrier, lfo)
    >>>
    >>> # Generate samples
    >>> samples = vibrato.get_samples(44100)

Modulation Types:
    - amp_mod: Modulate oscillator amplitude (common for ADSR envelopes)
    - freq_mod: Modulate oscillator frequency (vibrato, FM synthesis)
    - phase_mod: Modulate oscillator phase (phase modulation synthesis)

Note:
    ModulatedOscillator maintains the ended state of its modulators,
    making it suitable for voice management in polyphonic synthesizers.
"""

from enum import StrEnum

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    ComponentDescriptor,
    Generator,
    ParameterDescriptor,
)
from src.engine.core.registry import ComponentCategory, register_component
from src.engine.core.sample_mode import VALID_SAMPLE_MODES, SampleMode
from src.engine.generators.oscillators.oscillator import Oscillator
from src.engine.utils.validation import validate_sample_count


class ModulationLane(StrEnum):
    """Internal modulation lanes supported by ModulatedOscillator."""

    AMP = "amp"
    FREQ = "freq"
    FM = "fm"
    PHASE = "phase"


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
        fluent_api_name="oscillator_mod",
        description="Oscillator with modulation support (amplitude, frequency, phase)",
        tags=["oscillator", "modulated", "advanced"],
        parameters={
            "oscillator": ParameterDescriptor(
                name="oscillator",
                default=None,
                description="Carrier oscillator component.",
            ),
            "modulators": ParameterDescriptor(
                name="modulators",
                default=(),
                description=(
                    "Modulator components used for amplitude, frequency, or phase "
                    "modulation."
                ),
            ),
            "amp_mod": ParameterDescriptor(
                name="amp_mod",
                default=None,
                description="Optional amplitude modulation function.",
            ),
            "freq_mod": ParameterDescriptor(
                name="freq_mod",
                default=None,
                description="Optional frequency modulation function.",
            ),
            "fm_mod": ParameterDescriptor(
                name="fm_mod",
                default=None,
                description="Optional linear FM modulation function.",
            ),
            "phase_mod": ParameterDescriptor(
                name="phase_mod",
                default=None,
                description="Optional phase modulation function.",
            ),
        },
    )

    def __init__(
        self,
        oscillator: Oscillator,
        *modulators,
        amp_mod=None,
        freq_mod=None,
        fm_mod=None,
        phase_mod=None,
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

            fm_mod : Any function that takes in the frequency after pitch modulation
                and the modulator value and returns the linear-FM frequency value.

            phase_mod : Any function that takes in the initial oscillator phase
                value and the modulator value and returns the modified value.
                If set the third modulator of the last modulator is used for the values.
        """
        super().__init__()
        if not isinstance(oscillator, Oscillator):
            raise TypeError(
                f"Oscillator should be an instance of Oscillator. Given: {oscillator}"
            )

        self.oscillator: Oscillator = oscillator
        self.modulators = modulators

        self.amp_mod = amp_mod
        self.freq_mod = freq_mod
        self.fm_mod = fm_mod
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
        frequency = self.oscillator.init_freq

        if self.amp_mod is not None:
            mod_val = mod_vals[self._get_modulation_index(ModulationLane.AMP)]
            new_amp = self.amp_mod(self.oscillator.init_amp, mod_val)
            self.oscillator.amplitude = new_amp

        if self.freq_mod is not None:
            mod_val = mod_vals[self._get_modulation_index(ModulationLane.FREQ)]
            frequency = self.freq_mod(self.oscillator.init_freq, mod_val)

        if self.fm_mod is not None:
            mod_val = mod_vals[self._get_modulation_index(ModulationLane.FM)]
            frequency = self.fm_mod(frequency, mod_val)

        if self.freq_mod is not None or self.fm_mod is not None:
            self.oscillator.frequency = frequency

        if self.phase_mod is not None:
            mod_val = mod_vals[self._get_modulation_index(ModulationLane.PHASE)]
            new_phase = self.phase_mod(self.oscillator.init_phase, mod_val)
            self.oscillator.phase = new_phase

    def _get_modulation_index(self, target: ModulationLane) -> int:
        """Return the modulator index for a modulation lane."""
        if self.fm_mod is None:
            if target is ModulationLane.AMP:
                return 0
            if target is ModulationLane.FREQ:
                return 1 if self._modulators_count == 2 else 0
            if target is ModulationLane.PHASE:
                return 2 if self._modulators_count == 3 else -1

        active_targets: list[ModulationLane] = []
        if self.amp_mod is not None:
            active_targets.append(ModulationLane.AMP)
        if self.freq_mod is not None:
            active_targets.append(ModulationLane.FREQ)
        if self.fm_mod is not None:
            active_targets.append(ModulationLane.FM)
        if self.phase_mod is not None:
            active_targets.append(ModulationLane.PHASE)

        if self._modulators_count <= 1:
            return 0

        try:
            index = active_targets.index(target)
        except ValueError:
            return -1

        return min(index, self._modulators_count - 1)

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
        n = validate_sample_count(n)
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
        n = validate_sample_count(n)
        if n == 0:
            return np.empty(0, dtype=np.float32)

        # Step 1: Generate all modulator values in bulk (vectorized)
        mod_arrays = []
        for modulator in self.modulators:
            if hasattr(modulator, "get_samples"):
                mod_vals = modulator.get_samples(n, mode="vectorized")
            else:
                # Fallback to iterator for modulators without get_samples
                mod_vals = np.array(
                    [next(modulator) for _ in range(n)], dtype=np.float32
                )
            mod_arrays.append(mod_vals)

        # Step 2: Compute modulated parameters for ALL samples at once
        base_freq = self.oscillator.init_freq
        base_amp = self.oscillator.init_amp
        base_phase = self.oscillator.init_phase

        freqs = self._evaluate_modulation_array(
            n=n,
            base_value=base_freq,
            mod_arrays=mod_arrays,
            mod_func=self.freq_mod,
            mod_index=self._get_modulation_index(ModulationLane.FREQ),
        )
        if freqs is None:
            freqs = np.full(n, base_freq, dtype=np.float64)

        fm_freqs = self._evaluate_modulation_array(
            n=n,
            base_value=0.0,
            mod_arrays=mod_arrays,
            mod_func=(
                None
                if self.fm_mod is None
                else lambda _base, mod_val: self.fm_mod(freqs, mod_val)
            ),
            mod_index=self._get_modulation_index(ModulationLane.FM),
        )
        if fm_freqs is not None:
            freqs = fm_freqs

        amps = self._evaluate_modulation_array(
            n=n,
            base_value=base_amp,
            mod_arrays=mod_arrays,
            mod_func=self.amp_mod,
            mod_index=self._get_modulation_index(ModulationLane.AMP),
        )
        if amps is None:
            amps = np.full(n, base_amp, dtype=np.float64)

        phase_offsets_deg = self._evaluate_modulation_array(
            n=n,
            base_value=base_phase,
            mod_arrays=mod_arrays,
            mod_func=self.phase_mod,
            mod_index=self._get_modulation_index(ModulationLane.PHASE),
        )

        if not (
            hasattr(self.oscillator, "render_modulated_waveform")
            and hasattr(self.oscillator, "commit_modulated_phase_state")
        ):
            return self._get_samples_fallback(n, mod_arrays)

        # Step 3: Generate waveform through the carrier-owned phase/state API.
        waveform, phase_state = self.oscillator.render_modulated_waveform(
            freqs=freqs,
            phase_offsets_deg=phase_offsets_deg,
        )

        # Step 4: Apply amplitude modulation.
        samples = np.asarray(waveform * amps, dtype=np.float32)

        # Step 5: Update oscillator parameter state to match the final modulated sample.
        if mod_arrays:
            self._modulate([mod_arr[-1] for mod_arr in mod_arrays])

        self.oscillator.commit_modulated_phase_state(phase_state)

        return samples

    @staticmethod
    def _coerce_modulation_result(result, n: int) -> np.ndarray:
        """Normalize a scalar or array-like modulation result to a 1-D float array."""
        result_array = np.asarray(result, dtype=np.float64)
        if result_array.shape == ():
            return np.full(n, float(result_array), dtype=np.float64)
        return np.array(
            np.broadcast_to(result_array, (n,)), dtype=np.float64, copy=False
        )

    def _evaluate_modulation_array(
        self,
        n: int,
        base_value: float,
        mod_arrays: list[np.ndarray],
        mod_func,
        mod_index: int,
    ) -> np.ndarray | None:
        """Evaluate a modulation function for an entire vectorized buffer."""
        if mod_func is None:
            return None

        mod_vals = mod_arrays[mod_index]
        base_array = np.full(n, base_value, dtype=np.float64)

        try:
            result = mod_func(base_array, mod_vals)
        except (TypeError, ValueError):
            result = [mod_func(base_value, mod_vals[i]) for i in range(n)]

        return self._coerce_modulation_result(result, n)

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
        self,
        n: int = DEFAULT_SAMPLE_RATE,
        reset: bool = False,
        mode: SampleMode = "auto",
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
            >>> samples2 = mod_osc.get_samples(100,mode="iterator")
        """
        n = validate_sample_count(n)
        if mode not in VALID_SAMPLE_MODES:
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


@register_component()
class ModulatedFrequency(ModulatedOscillator):
    """Frequency-modulated oscillator for vibrato and FM synthesis.

    This is a specialized version of ModulatedOscillator that specifically handles
    frequency modulation. It provides a simpler interface for the common case where
    you want to modulate only the frequency of an oscillator.

    This class actually modulates the FREQUENCY of the oscillator (creating vibrato
    or FM synthesis effects), not the amplitude. For amplitude modulation (tremolo),
    use ModulatedVolume or ModulatedOscillator with amp_mod.

    Args:
        oscillator: The carrier oscillator whose frequency will be modulated.
        modulator: Generator that produces frequency modulation values.
                  The modulator output is ADDED to the base frequency.
                  Example: LFO with amplitude=50 creates ±50 Hz variation.
        freq_mod_func: Optional custom frequency modulation function.
                      Defaults to: lambda base_freq, mod_val: base_freq + mod_val

    Example - Vibrato (LFO modulates frequency):
        >>> from src.engine.oscillator import SineOscillator
        >>>
        >>> # Carrier: 440 Hz sine wave
        >>> carrier = SineOscillator(frequency=440, amplitude=0.5)
        >>>
        >>> # LFO: 5 Hz sine with ±50 Hz range
        >>> lfo = SineOscillator(frequency=5.0, amplitude=50.0)
        >>>
        >>> # Create frequency-modulated oscillator (vibrato)
        >>> vibrato = ModulatedFrequency(carrier, lfo)
        >>>
        >>> # Generate samples
        >>> samples = vibrato.get_samples(44100)
        >>> # Result: 440 Hz tone with 5 Hz vibrato, ±50 Hz depth

    Example - FM Synthesis (audio-rate modulation):
        >>> # Carrier: 440 Hz
        >>> carrier = SineOscillator(frequency=440, amplitude=0.3)
        >>>
        >>> # Modulator: 220 Hz (half the carrier frequency)
        >>> # with large amplitude for strong FM effect
        >>> fm_modulator = SineOscillator(frequency=220, amplitude=200.0)
        >>>
        >>> # Create FM oscillator
        >>> fm_synth = ModulatedFrequency(carrier, fm_modulator)
        >>>
        >>> # Generate samples
        >>> samples = fm_synth.get_samples(44100)
        >>> # Result: Rich harmonic content from FM synthesis

    Example - Custom modulation function:
        >>> # Custom function for exponential frequency modulation
        >>> def exp_freq_mod(base_freq, mod_val):
        ...     # Convert linear modulation to exponential (semitones)
        ...     semitones = mod_val / 100.0  # mod_val in cents
        ...     return base_freq * (2.0 ** (semitones / 12.0))
        >>>
        >>> carrier = SineOscillator(frequency=440)
        >>> # ±100 cents = ±1 semitone
        >>> lfo = SineOscillator(frequency=5.0, amplitude=100.0)
        >>> vibrato = ModulatedFrequency(carrier, lfo, freq_mod_func=exp_freq_mod)

    Attributes:
        oscillator: The carrier oscillator being modulated.
        modulators: Tuple containing the frequency modulator(s).
        freq_mod: The frequency modulation function being used.

    Note:
        The modulator output is ADDED to the base frequency by default.
        For ±50 Hz vibrato around 440 Hz:
        - Base frequency: 440 Hz
        - LFO amplitude: 50 Hz
        - Result: frequency sweeps from 390 Hz to 490 Hz

    See Also:
        - ModulatedOscillator: For general-purpose modulation (amp, freq, phase)
        - ModulatedVolume: For amplitude modulation (tremolo)
    """

    descriptor = ComponentDescriptor(
        name="Modulated Frequency",
        category=ComponentCategory.OSCILLATOR,
        fluent_api_name="frequency_mod",
        description="Frequency-modulated oscillator (vibrato, FM synthesis)",
        tags=["oscillator", "modulated", "frequency", "vibrato", "fm"],
        parameters={
            "oscillator": ParameterDescriptor(
                name="oscillator",
                default=None,
                description="Carrier oscillator component.",
            ),
            "modulator": ParameterDescriptor(
                name="modulator",
                default=None,
                description="Frequency modulator component.",
            ),
            "freq_mod_func": ParameterDescriptor(
                name="freq_mod_func",
                default=None,
                description="Optional custom frequency modulation function.",
            ),
        },
    )

    def __init__(self, oscillator, modulator, freq_mod_func=None):
        """Initialize frequency-modulated oscillator.

        Args:
            oscillator: Carrier oscillator whose frequency will be modulated.
                       Must be an instance of Oscillator.
            modulator: Frequency modulator (LFO or audio-rate oscillator).
                      Output is added to base frequency by default.
            freq_mod_func: Optional custom frequency modulation function with
                          signature: (base_freq: float, mod_val: float) -> float
                          Default: lambda base_freq, mod_val: base_freq + mod_val

        Raises:
            TypeError: If oscillator is not an Oscillator instance.
            TypeError: If modulator doesn't support iteration.

        Example:
            >>> from src.engine import SineOscillator
            >>> carrier = SineOscillator(frequency=440, amplitude=0.5)
            >>> lfo = SineOscillator(frequency=5.0, amplitude=50.0)
            >>> fm_osc = ModulatedFrequency(carrier, lfo)
        """

        def default_freq_mod(base_freq, mod_val):
            return base_freq + mod_val

        # Default frequency modulation: add modulator output to base frequency
        if freq_mod_func is None:
            freq_mod_func = default_freq_mod

        # Initialize parent ModulatedOscillator with only freq_mod
        # Note: modulator is passed as positional argument (*modulators in parent)
        super().__init__(
            oscillator,
            modulator,  # Positional argument for *modulators
            freq_mod=freq_mod_func,
            amp_mod=None,  # No amplitude modulation
            phase_mod=None,  # No phase modulation
        )

    def __repr__(self):
        """Return string representation."""
        return (
            f"ModulatedFrequency("
            f"oscillator={self.oscillator.__class__.__name__}, "
            f"modulator={self.modulators[0].__class__.__name__})"
        )
