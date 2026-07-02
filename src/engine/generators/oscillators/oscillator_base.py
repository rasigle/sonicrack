"""Shared oscillator infrastructure and amplitude utilities."""

import logging
from abc import abstractmethod

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.core.component import Generator
from src.engine.core.sample_mode import SampleMode, VALID_SAMPLE_MODES
from src.engine.utils.decorators import track_provided_args
from src.engine.utils.math import db_to_linear, linear_to_db
from src.engine.utils.ramping import consume_linear_ramp, duration_ms_to_samples
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
            gain_db,  # noqa
        )

        self._i: float = 0.0
        self._step: float = 0.0

        self._f = frequency
        self._a = self._initial_amp
        self._p = self._phase

        self._target_amplitude = self._initial_amp
        self._current_amplitude = self._initial_amp
        self._smoothing_samples_remaining = 0
        self._smoothing_samples_duration_total = duration_ms_to_samples(
            sample_rate, DEFAULT_TIME_AMPLITUDE_SMOOTHING_MS
        )

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
        self._target_amplitude = value
        self._smoothing_samples_remaining = self._smoothing_samples_duration_total
        self._a = value
        self._post_amp_set()

    @property
    def gain_db(self) -> float:
        return linear_to_db(self._a)

    @gain_db.setter
    def gain_db(self, value: float):
        new_amplitude = float(db_to_linear(value))
        self._target_amplitude = new_amplitude
        self._smoothing_samples_remaining = self._smoothing_samples_duration_total
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
            self._smoothing_samples_duration_total = duration_ms_to_samples(
                value, DEFAULT_TIME_AMPLITUDE_SMOOTHING_MS
            )
            self._post_sample_rate_set()

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

        if self._smoothing_samples_remaining > 0:
            smooth_count = min(n, self._smoothing_samples_remaining)
            amp_envelope, self._current_amplitude, self._smoothing_samples_remaining = (
                consume_linear_ramp(
                    self._current_amplitude,
                    self._target_amplitude,
                    self._smoothing_samples_remaining,
                    smooth_count,
                )
            )

            samples = np.empty(n, dtype=np.float32)
            samples[:smooth_count] = values[:smooth_count] * amp_envelope

            if smooth_count < n:
                samples[smooth_count:] = values[smooth_count:] * self._target_amplitude

            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude

            return samples.astype(np.float32)

        self._current_amplitude = self._a
        return np.asarray(values * self._a, dtype=np.float32)

    def __next__(self):
        return None

    def __iter__(self):
        self.frequency = self._freq
        self.phase = self._phase
        self.amplitude = self._initial_amp
        self._initialize_osc()
        return self

    def get_samples_iterator(
        self, n: int = DEFAULT_SAMPLE_RATE, reset: bool = False
    ) -> np.ndarray:
        n = validate_sample_count(n)
        if reset:
            iter(self)
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
            mode = "vectorized" if n >= 512 else "iterator"

        if mode == "iterator":
            samples_list = self.get_samples_iterator(n, reset=reset)
            return np.array(samples_list, dtype=np.float32)

        if reset:
            iter(self)
        return self.get_samples_vectorized(n)


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
