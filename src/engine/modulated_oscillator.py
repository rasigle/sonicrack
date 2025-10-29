from constants import DEFAULT_SAMPLE_RATE
from engine.oscillator import Oscillator


class ModulatedOscillator:
    """Creates a modulated oscillator by using a plain oscillator along with modulators,
    the `[parameter]_mod` functions of the signature (float, float) -> float are used
    to decide the method of modulation.

    Has `.trigger_release()` implemented to trigger the release stage of the
    modulators. Similarly, has `.ended` to indicate the end of signal generator of the
    modulators if the generation is meant to be finite.

    The ModulatedOscillator internal values are set by calling __init__ and then
    __next__ to generate the sequence of values.
    """

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
        if not isinstance(oscillator, Oscillator):
            raise TypeError(
                f"Oscillator should be an instance of Oscillator. "
                f"Given: {type(oscillator)}"
            )
        # if not all([isinstance(m, Modulator) for m in modulators]):
        #     raise TypeError(
        #         f"All given modulators should be instances of Modulator. "
        #         f"Given: {[type(mod) for mod in modulators]}"
        #     )
        self.oscillator = oscillator
        self.modulators = modulators

        self.amp_mod = amp_mod
        self.freq_mod = freq_mod
        self.phase_mod = phase_mod
        self._modulators_count = len(modulators)

    def __iter__(self):
        iter(self.oscillator)
        [iter(modulator) for modulator in self.modulators]
        return self

    def _modulate(self, mod_vals):
        if self.amp_mod is not None:
            new_amp = self.amp_mod(self.oscillator.init_amp, mod_vals[0])
            self.oscillator.amp = new_amp

        if self.freq_mod is not None:
            mod_val = mod_vals[1 if self._modulators_count == 2 else 0]
            new_freq = self.freq_mod(self.oscillator.init_freq, mod_val)
            self.oscillator.freq = new_freq

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

    def __next__(self):
        mod_vals = [next(modulator) for modulator in self.modulators]
        self._modulate(mod_vals)
        return next(self.oscillator)

    def get_samples(self, n: int = DEFAULT_SAMPLE_RATE, it: bool = False):
        """Return the next *n* samples from this generator.

        Args:
            n: Number of samples to produce. Defaults to `DEFAULT_SAMPLE_RATE`.
            it: If True, return an iterator instead of a list.

        Returns:
            list[float]: List of `n` consecutive samples produced by calling
            `next(self)` repeatedly.

        Note:
            If `it` is True, the method returns an iterator instead of a list.
        """
        if it:
            iter(self)
        return [next(self) for _ in range(n)]
