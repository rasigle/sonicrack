"""Shared oscillator infrastructure and amplitude utilities."""

import logging
from abc import abstractmethod

import numpy as np

from src.constants import (
    AUTO_MODE_VECTORIZE_THRESHOLD,
    DEFAULT_GAIN_DB,
    DEFAULT_SAMPLE_RATE,
)
from src.engine.core.component import Generator, ParameterDescriptor
from src.engine.core.parameter import RuntimeParameter, SmoothingPolicy
from src.engine.core.sample_mode import VALID_SAMPLE_MODES, SampleMode
from src.engine.utils.decorators import track_provided_args
from src.engine.utils.math import db_to_linear, linear_to_db
from src.engine.utils.validation import validate_sample_count, validate_sample_rate

DEFAULT_TIME_AMPLITUDE_SMOOTHING_MS = 10
"""Default duration for amplitude smoothing to prevent clicks."""


class Oscillator(Generator):
    """Base class for all signal generators with gain and sample generation helpers."""

    _provided_args: set[str]
    ended: bool = False

    @track_provided_args
    def __init__(
        self,
        frequency: float = 440,
        amplitude: float = 1.0,
        gain_db: float | None = DEFAULT_GAIN_DB,
        phase: float = 0.0,
        sample_rate: int | float = DEFAULT_SAMPLE_RATE,
        wave_range: tuple[float, float] = (-1, 1),
    ):
        sample_rate = validate_sample_rate(sample_rate)
        self._sample_rate = sample_rate
        super().__init__(sample_rate=sample_rate)

        self._freq = frequency
        self._phase = phase
        self._wave_range = wave_range
        self._initial_amp = _derive_amplitude_from_init(
            self._provided_args,
            amplitude,
            gain_db,
        )

        self._i: float = 0.0
        self._step: float = 0.0

        self._f = frequency
        self._a = self._initial_amp
        self._p = self._phase

        # Create RuntimeParameter for amplitude with LINEAR smoothing
        amplitude_descriptor = ParameterDescriptor(
            name="amplitude",
            default=self._initial_amp,
            minimum=0.0,
            smoothing_policy=SmoothingPolicy.LINEAR,
            smoothing_duration_ms=DEFAULT_TIME_AMPLITUDE_SMOOTHING_MS,
        )
        self._amplitude_param = RuntimeParameter(amplitude_descriptor, sample_rate)

        self._needs_range_conversion: bool = False
        self._range_scale: float = 1.0
        self._range_offset: float = 0.0
        self._update_range_conversion()

        iter(self)

    @property
    def init_freq(self):
        return self._freq

    @property
    def init_amp(self):
        return self._initial_amp

    @property
    def init_phase(self):
        return self._phase

    @property
    def wave_range(self):
        return self._wave_range

    @wave_range.setter
    def wave_range(self, value: tuple[float, float]):
        self._wave_range = value
        self._update_range_conversion()

    def _update_range_conversion(self):
        self._needs_range_conversion = self._wave_range != (-1, 1)
        if self._needs_range_conversion:
            self._range_scale = (self._wave_range[1] - self._wave_range[0]) / 2.0
            self._range_offset = (self._wave_range[1] + self._wave_range[0]) / 2.0
        else:
            self._range_scale = 1.0
            self._range_offset = 0.0

    @property
    def frequency(self):
        return self._f

    @frequency.setter
    def frequency(self, value):
        self._f = value
        self._post_freq_set()

    @property
    def amplitude(self):
        return self._a

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude_param.value = value
        self._a = value
        self._post_amp_set()

    @property
    def gain_db(self) -> float:
        return linear_to_db(self._a)

    @gain_db.setter
    def gain_db(self, value: float):
        new_amplitude = float(db_to_linear(value))
        self._amplitude_param.value = new_amplitude
        self._a = new_amplitude
        self._post_amp_set()

    @property
    def phase(self):
        return self._p

    @phase.setter
    def phase(self, value):
        self._p = value
        self._post_phase_set()

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        value = validate_sample_rate(value)
        if value != self._sample_rate:
            self._sample_rate = value
            self._amplitude_param.sample_rate = value
            self._post_sample_rate_set()

    def reset(self):
        """Reset oscillator to its initial constructor state."""
        self.frequency = self._freq
        self.phase = self._phase
        self.amplitude = self._initial_amp

        # Important: avoid a smoothed ramp after reset.
        self._amplitude_param._current_value = self._initial_amp
        self._amplitude_param._target_value = self._initial_amp
        self._amplitude_param._smoothing_samples_remaining = 0

        self._i = 0.0
        self.ended = False

        self._initialize_osc()
        return self

    def _post_freq_set(self):
        pass

    def _post_amp_set(self):
        pass

    def _post_phase_set(self):
        pass

    def _post_sample_rate_set(self):
        self._post_freq_set()

    def _initialize_osc(self):
        pass

    def _apply_wave_range_value(self, value: float) -> float:
        if self._needs_range_conversion:
            return value * self._range_scale + self._range_offset
        return value

    def _apply_wave_range_values(self, values: np.ndarray) -> np.ndarray:
        if self._needs_range_conversion:
            return values * self._range_scale + self._range_offset
        return values

    def _apply_amplitude_to_buffer(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        n = len(values)

        if n == 0:
            return np.empty(0, dtype=np.float32)

        # Get smoothed amplitude envelope from RuntimeParameter
        amp_envelope = self._amplitude_param.get_interpolated_buffer(n)
        return (values * amp_envelope).astype(np.float32)

    # Backward-compatible properties for tests and legacy code
    @property
    def _target_amplitude(self) -> float:
        """Backward compatibility: delegate to RuntimeParameter."""
        return self._amplitude_param._target_value

    @_target_amplitude.setter
    def _target_amplitude(self, value: float):
        """Backward compatibility: delegate to RuntimeParameter."""
        self._amplitude_param.value = value

    @property
    def _current_amplitude(self) -> float:
        """Backward compatibility: delegate to RuntimeParameter."""
        return self._amplitude_param._current_value

    @_current_amplitude.setter
    def _current_amplitude(self, value: float):
        """Backward compatibility: delegate to RuntimeParameter."""
        self._amplitude_param._current_value = value

    @property
    def _smoothing_samples_remaining(self) -> int:
        """Backward compatibility: delegate to RuntimeParameter."""
        return self._amplitude_param._smoothing_samples_remaining

    @_smoothing_samples_remaining.setter
    def _smoothing_samples_remaining(self, value: int):
        """Backward compatibility: delegate to RuntimeParameter."""
        self._amplitude_param._smoothing_samples_remaining = value

    @property
    def _smoothing_samples_duration_total(self) -> int:
        """Backward compatibility: delegate to RuntimeParameter."""
        return self._amplitude_param._smoothing_duration_samples

    @_smoothing_samples_duration_total.setter
    def _smoothing_samples_duration_total(self, value: int):
        """Backward compatibility: delegate to RuntimeParameter."""
        self._amplitude_param._smoothing_duration_samples = value

    def process_frequency_buffer(self, frequencies: np.ndarray) -> np.ndarray:
        """Generate samples with per-sample frequency modulation (bulk API).

        This is a high-performance method for processing frequency-modulated
        oscillators. Instead of calling next() 512 times with frequency changes,
        this processes an entire buffer at once with vectorized operations.

        Args:
            frequencies: Array of frequencies (Hz), one per output sample

        Returns:
            Array of audio samples with modulated frequency

        Example:
            >>> from engine import SineOscillator
            >>>
            >>> osc = SineOscillator(frequency=440)
            >>> # Generate 512 samples with frequency sweep 440-880 Hz
            >>> freqs = np.linspace(440, 880, 512)
            >>> samples = osc.process_frequency_buffer(freqs)
        """
        frequencies = np.asarray(frequencies, dtype=np.float64)

        # Calculate phase increments for each frequency
        phase_increments = (2.0 * np.pi * frequencies) / self._sample_rate

        # Accumulate phase (cumulative sum maintains phase continuity)
        phases = np.cumsum(phase_increments) + self._p

        # Update internal phase for next call (maintain continuity)
        self._p = phases[-1] % (2.0 * np.pi)

        # Generate waveform from phase array (calls subclass implementation)
        raw_samples = self._generate_waveform_from_phases(phases)

        # Apply amplitude smoothing if active
        smoothed_samples = self._apply_amplitude_to_buffer(raw_samples)

        # Apply wave range conversion if needed
        return self._apply_wave_range_values(smoothed_samples).astype(np.float32)

    def _generate_waveform_from_phases(self, phases: np.ndarray) -> np.ndarray:
        """Generate waveform from array of phase values.

        Subclasses should override this to provide their specific waveform.
        Default implementation calls the existing _generate_waveform method.

        Args:
            phases: Array of phase values (radians)

        Returns:
            Array of raw waveform samples
        """
        # Default: call the existing _generate_waveform method if available
        if hasattr(self, "_generate_waveform"):
            return self._generate_waveform(phases)

        # Fallback: generate samples one by one (slow, but works)
        samples = np.empty(len(phases), dtype=np.float32)
        for i, phase in enumerate(phases):
            old_phase = self._p
            self._p = phase
            samples[i] = next(self)
            self._p = old_phase
        return samples

    def __next__(self):
        raise StopIteration

    def __iter__(self):
        return self.reset()

    def get_samples_iterator(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False
    ) -> np.ndarray:
        n = validate_sample_count(n)
        if reset:
            self.reset()
        return np.array([next(self) for _ in range(n)], dtype=np.float32)

    @abstractmethod
    def get_samples_vectorized(self, n: int) -> np.ndarray:
        pass

    def get_samples(
        self,
        n: int = DEFAULT_SAMPLE_RATE,
        reset: bool = False,
        mode: SampleMode = "auto",
    ) -> np.ndarray:
        n = validate_sample_count(n)
        if mode not in VALID_SAMPLE_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'auto', 'iterator', or 'vectorized'."
            )

        if mode == "auto":
            mode = "vectorized" if n >= AUTO_MODE_VECTORIZE_THRESHOLD else "iterator"

        if mode == "iterator":
            samples_list = self.get_samples_iterator(n, reset=reset)
            return np.array(samples_list, dtype=np.float32)

        if reset:
            self.reset()
        return self.get_samples_vectorized(n)

    # --- Modulation API (for ModulatedOscillator vectorized path) ---

    def render_modulated_waveform(
        self,
        freqs: np.ndarray,
        phase_offsets_deg: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Render unamplified waveform for per-sample modulation.

        Subclasses should override this for optimal performance. The default
        implementation uses the iterator fallback path.

        Args:
            freqs: Array of frequencies (Hz) per sample
            phase_offsets_deg: Optional phase offsets in degrees per sample

        Returns:
            Tuple of (waveform_samples, phase_state_dict)
        """
        n = len(freqs)
        samples = np.empty(n, dtype=np.float32)

        # Save and restore phase state
        saved_i = self._i
        saved_p = self._p

        for i in range(n):
            self._f = float(freqs[i])
            self._post_freq_set()
            if phase_offsets_deg is not None:
                self._p = float(phase_offsets_deg[i])
                self._post_phase_set()
            samples[i] = next(self)

        final_state = {
            "carrier_phase": self._i,
            "sample_index": float(n),
        }

        # Restore original state
        self._i = saved_i
        self._p = saved_p
        self._post_freq_set()
        self._post_phase_set()

        return samples, final_state

    def commit_modulated_phase_state(self, state: dict[str, float]) -> None:
        """Commit phase state produced by ``render_modulated_waveform``.

        Subclasses should override this to properly restore internal state.
        """
        self._i = float(state.get("carrier_phase", 0.0))


def _derive_amplitude_from_init(
    given_args: set[str], amplitude: float | None, gain_db: float | None
) -> float:
    """Determine the initial linear amplitude from constructor arguments."""
    if amplitude is not None and not isinstance(amplitude, (int, float, np.number)):
        raise TypeError(f"Amplitude must be number, got {type(amplitude).__name__}")
    if amplitude is not None and amplitude < 0.0:
        raise ValueError(f"Amplitude must be non-negative, got {amplitude}")
    if gain_db is not None and not isinstance(gain_db, (int, float, np.number)):
        raise TypeError(f"Gain_db must be a number, got {type(gain_db).__name__}")

    gain_db_set = "gain_db" in given_args
    amplitude_set = "amplitude" in given_args

    if gain_db_set and gain_db is not None:
        expected_amp = float(db_to_linear(gain_db))
        if (
            amplitude_set
            and amplitude is not None
            and not np.isclose(amplitude, expected_amp)
        ):
            logging.warning(
                f"Both gain_db={gain_db} and amplitude={amplitude} were specified. "
                f"Using gain_db, which results in an amplitude of {expected_amp:.3f}."
            )
        return expected_amp

    if amplitude_set and amplitude is not None:
        return amplitude

    if gain_db is not None:
        return float(db_to_linear(gain_db))

    return 1.0
