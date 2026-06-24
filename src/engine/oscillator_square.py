"""Square-wave strategies and oscillator implementation."""

import logging
import math
from abc import ABC, abstractmethod
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.audio_component import (
    ComponentDescriptor,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.audio_component_registry import ComponentCategory, register_component
from src.engine.oscillator_base import Oscillator
from src.engine.validation import validate_sample_count, validate_sample_rate
from src.utils.utils import filter_provided_args, track_provided_args

SquareWaveMode = Literal[
    "ideal", "ideal_smooth", "bandlimited", "vcv", "soft", "comparator"
]
TWO_PI = 2 * np.pi
VCV_MINBLEP_ZERO_CROSSINGS = 16
VCV_MINBLEP_OVERSAMPLE = 16
logger = logging.getLogger(__name__)


@runtime_checkable
class _FrequencyAwareStrategy(Protocol):
    def set_frequency(self, frequency: float) -> None:
        pass


@runtime_checkable
class _SampleRateAwareStrategy(Protocol):
    def set_sample_rate(self, sample_rate: float) -> None:
        pass


@runtime_checkable
class _ResettableStrategy(Protocol):
    def reset_state(self) -> None:
        pass


class SquareWaveStrategy(ABC):
    """Abstract base class for square wave generation strategies."""

    @abstractmethod
    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        pass

    @abstractmethod
    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        pass


class IdealSquareStrategy(SquareWaveStrategy):
    """Ideal aliased square wave with instant transitions."""

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
        return np.where(phases < pulsewidth_threshold, high_value, low_value).astype(
            np.float32
        )


class IdealSquareStrategySmoothing(SquareWaveStrategy):
    """Ideal square wave with internal amplitude smoothing."""

    def __init__(
        self, smoothing_time_ms: float = 5.0, sample_rate: float = 44100
    ) -> None:
        if smoothing_time_ms < 0:
            raise ValueError(
                f"smoothing_time_ms must be non-negative, got {smoothing_time_ms}"
            )
        self.smoothing_time_ms = smoothing_time_ms
        self.sample_rate = validate_sample_rate(sample_rate)
        self._current_amplitude = 1.0
        self._target_amplitude = 1.0
        self._start_amplitude = 1.0
        self._smoothing_samples_total = 0
        self._smoothing_samples_remaining = 0

    def _smoothing_sample_count(self) -> int:
        return int(self.smoothing_time_ms * self.sample_rate / 1000)

    def set_sample_rate(self, sample_rate: float) -> None:
        self.sample_rate = validate_sample_rate(sample_rate)

    def reset_amplitude(self, amplitude: float) -> None:
        self._current_amplitude = amplitude
        self._target_amplitude = amplitude
        self._start_amplitude = amplitude
        self._smoothing_samples_total = 0
        self._smoothing_samples_remaining = 0

    def set_amplitude(self, amplitude: float) -> None:
        if abs(amplitude - self._current_amplitude) > 0.001:
            smoothing_samples = self._smoothing_sample_count()
            self._start_amplitude = self._current_amplitude
            self._target_amplitude = amplitude
            self._smoothing_samples_total = smoothing_samples
            self._smoothing_samples_remaining = smoothing_samples
            if smoothing_samples == 0:
                self.reset_amplitude(amplitude)

    def _consume_amplitude_envelope(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty(0, dtype=np.float32)
        if self._smoothing_samples_remaining <= 0:
            return np.full(n, self._current_amplitude, dtype=np.float32)

        smooth_count = min(n, self._smoothing_samples_remaining)
        completed = self._smoothing_samples_total - self._smoothing_samples_remaining
        positions = completed + np.arange(1, smooth_count + 1, dtype=np.float64)
        progress = positions / self._smoothing_samples_total
        envelope = (
            self._start_amplitude
            + (self._target_amplitude - self._start_amplitude) * progress
        ).astype(np.float32)

        self._smoothing_samples_remaining -= smooth_count
        self._current_amplitude = float(envelope[-1])
        if self._smoothing_samples_remaining <= 0:
            self.reset_amplitude(self._target_amplitude)

        if smooth_count == n:
            return envelope

        tail = np.full(n - smooth_count, self._target_amplitude, dtype=np.float32)
        return np.concatenate((envelope, tail))

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        val = high_value if phase < pulsewidth_threshold else low_value
        return float(val * self._consume_amplitude_envelope(1)[0])

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        n = len(phases)
        val = np.where(phases < pulsewidth_threshold, high_value, low_value)
        return (val * self._consume_amplitude_envelope(n)).astype(np.float32)


class SoftSquareStrategy(SquareWaveStrategy):
    """Square wave with tanh-smoothed transitions."""

    def __init__(self, smoothness: float = 10.0) -> None:
        if smoothness <= 0:
            raise ValueError(f"smoothness must be positive, got {smoothness}")
        self.smoothness = smoothness

    def _smooth_level(
        self, phases: float | np.ndarray, pulsewidth_threshold: float
    ) -> float | np.ndarray:
        if pulsewidth_threshold <= 0.0:
            return np.zeros_like(phases, dtype=np.float32)
        if pulsewidth_threshold >= TWO_PI:
            return np.ones_like(phases, dtype=np.float32)

        phases_wrapped = np.asarray(phases) % TWO_PI
        inside_high = phases_wrapped < pulsewidth_threshold
        distance_inside = np.minimum(
            phases_wrapped, pulsewidth_threshold - phases_wrapped
        )
        distance_outside = np.minimum(
            phases_wrapped - pulsewidth_threshold,
            TWO_PI - phases_wrapped,
        )
        signed_distance = np.where(inside_high, distance_inside, -distance_outside)
        return (np.tanh(signed_distance * self.smoothness) + 1.0) / 2.0

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        level = self._smooth_level(phase, pulsewidth_threshold)
        return float(low_value + (high_value - low_value) * level)

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        level = self._smooth_level(phases, pulsewidth_threshold)
        return np.asarray(
            low_value + (high_value - low_value) * level, dtype=np.float32
        )


class BandlimitedSquareStrategy(SquareWaveStrategy):
    """PolyBLEP antialiased square wave with reduced edge aliasing."""

    def __init__(self, sample_rate: float = 44100, frequency: float = 440) -> None:
        self.sample_rate = validate_sample_rate(sample_rate)
        self.frequency = frequency

    def set_sample_rate(self, sample_rate: float) -> None:
        self.sample_rate = validate_sample_rate(sample_rate)

    def set_frequency(self, frequency: float) -> None:
        self.frequency = frequency

    @property
    def _increment(self) -> float:
        return min(abs(self.frequency / self.sample_rate), 0.5)

    @staticmethod
    def _polyblep(phases: np.ndarray, increment: float) -> np.ndarray:
        correction = np.zeros_like(phases, dtype=np.float64)
        if increment <= 0.0:
            return correction

        start_mask = phases < increment
        if np.any(start_mask):
            p = phases[start_mask] / increment
            correction[start_mask] = (p + p) - (p * p) - 1.0

        end_mask = phases > 1.0 - increment
        if np.any(end_mask):
            p = (phases[end_mask] - 1.0) / increment
            correction[end_mask] = (p + p) + (p * p) + 1.0

        return correction

    def _generate_normalized(
        self, phases: np.ndarray, pulsewidth_threshold: float
    ) -> np.ndarray:
        pulsewidth = pulsewidth_threshold / TWO_PI
        if pulsewidth <= 0.0:
            return np.full(len(phases), -1.0, dtype=np.float32)
        if pulsewidth >= 1.0:
            return np.full(len(phases), 1.0, dtype=np.float32)

        normalized_phases = (phases % TWO_PI) / TWO_PI
        output = np.where(normalized_phases < pulsewidth, 1.0, -1.0).astype(np.float64)
        increment = self._increment
        output += self._polyblep(normalized_phases, increment)
        shifted_phases = (normalized_phases - pulsewidth + 1.0) % 1.0
        output -= self._polyblep(shifted_phases, increment)
        return output.astype(np.float32)

    @staticmethod
    def _scale_from_normalized(
        values: np.ndarray, low_value: float, high_value: float
    ) -> np.ndarray:
        midpoint = (high_value + low_value) / 2.0
        scale = (high_value - low_value) / 2.0
        return np.asarray(midpoint + values * scale, dtype=np.float32)

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        normalized = self._generate_normalized(
            np.asarray([phase], dtype=np.float64), pulsewidth_threshold
        )
        return float(self._scale_from_normalized(normalized, low_value, high_value)[0])

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        normalized = self._generate_normalized(phases, pulsewidth_threshold)
        return self._scale_from_normalized(normalized, low_value, high_value)


def _blackman_harris(position: np.ndarray) -> np.ndarray:
    return (
        0.35875
        - 0.48829 * np.cos(TWO_PI * position)
        + 0.14128 * np.cos(2 * TWO_PI * position)
        - 0.01168 * np.cos(3 * TWO_PI * position)
    )


def _minimum_phase_minblep_table(
    zero_crossings: int = VCV_MINBLEP_ZERO_CROSSINGS,
    oversample: int = VCV_MINBLEP_OVERSAMPLE,
) -> np.ndarray:
    n = 2 * zero_crossings * oversample
    positions = np.arange(n, dtype=np.float64) / oversample - zero_crossings
    impulse = np.sinc(positions)
    impulse *= _blackman_harris(np.arange(n, dtype=np.float64) / (n - 1))

    spectrum = np.fft.fft(impulse)
    log_magnitude = np.log(np.maximum(np.abs(spectrum), np.exp(-10.0)))
    cepstrum = np.fft.ifft(log_magnitude)
    cepstrum[1 : n // 2] *= 2.0
    cepstrum[n // 2 :] = 0.0
    minimum_phase = np.fft.ifft(np.exp(np.fft.fft(cepstrum))).real

    step = np.cumsum(minimum_phase) / oversample
    step = np.concatenate(([0.0], step[:-1]))
    step /= step[-1] + minimum_phase[-1] / oversample
    return np.asarray(step - 1.0, dtype=np.float32)


class VCVRackSquareStrategy(SquareWaveStrategy):
    """VCV Rack Fundamental-style minBLEP square wave with DC blocking."""

    _minblep_table = _minimum_phase_minblep_table()

    def __init__(
        self,
        sample_rate: float = 44100,
        frequency: float = 440,
        dc_block: bool = True,
    ) -> None:
        self.sample_rate = validate_sample_rate(sample_rate)
        self.frequency = frequency
        self.dc_block = dc_block
        self._buffer = np.zeros(2 * VCV_MINBLEP_ZERO_CROSSINGS, dtype=np.float32)
        self._prev_phase: float | None = None
        self._last_square_state = 1.0
        self._last_pulsewidth = 0.5
        self._dc_lowpass_state = 0.0
        self._update_dc_alpha()

    def set_sample_rate(self, sample_rate: float) -> None:
        self.sample_rate = validate_sample_rate(sample_rate)
        self._update_dc_alpha()

    def set_frequency(self, frequency: float) -> None:
        self.frequency = frequency

    def reset_state(self) -> None:
        self._buffer.fill(0.0)
        self._prev_phase = None
        self._last_square_state = 1.0
        self._last_pulsewidth = 0.5
        self._dc_lowpass_state = 0.0

    def _update_dc_alpha(self) -> None:
        cutoff = min(0.4, 20.0 / self.sample_rate)
        w = TWO_PI * cutoff
        self._dc_alpha = w / (1.0 + w)

    @staticmethod
    def _square_state(phase: float, pulsewidth: float) -> float:
        return 1.0 if phase < pulsewidth else -1.0

    @staticmethod
    def _crossing_subsample(
        threshold: float, start_phase: float, end_phase: float
    ) -> float | None:
        delta = end_phase - start_phase
        if delta == 0.0:
            return None
        diff = threshold - start_phase
        if delta >= 0.0:
            threshold -= math.floor(diff)
        else:
            threshold -= math.ceil(diff)
        subsample = (threshold - start_phase) / delta
        if 0.0 < subsample <= 1.0:
            return float(subsample)
        return None

    def _insert_discontinuity(self, subsample: float, magnitude: float) -> None:
        if not 0.0 < subsample <= 1.0 or magnitude == 0.0:
            return
        table = self._minblep_table
        extended_table = np.concatenate((table, np.zeros(1, dtype=np.float32)))
        offset = (1.0 - subsample) * VCV_MINBLEP_OVERSAMPLE
        for index in range(len(self._buffer)):
            position = index * VCV_MINBLEP_OVERSAMPLE + offset
            lower = int(position)
            fraction = position - lower
            value = extended_table[lower] + fraction * (
                extended_table[lower + 1] - extended_table[lower]
            )
            self._buffer[index] += magnitude * value

    def _shift_buffer(self) -> float:
        value = float(self._buffer[0])
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = 0.0
        return value

    def _process_dc_filter(self, value: float) -> float:
        if not self.dc_block:
            return value
        self._dc_lowpass_state += self._dc_alpha * (value - self._dc_lowpass_state)
        return value - self._dc_lowpass_state

    def _process_normalized_sample(
        self, phase: float, pulsewidth_threshold: float
    ) -> float:
        pulsewidth = pulsewidth_threshold / TWO_PI
        if pulsewidth < 0.01:
            pulsewidth = 0.01
        elif pulsewidth > 0.99:
            pulsewidth = 0.99
        current_phase = (phase % TWO_PI) / TWO_PI
        if self._prev_phase is None:
            self._prev_phase = (current_phase - self.frequency / self.sample_rate) % 1.0
            self._last_square_state = self._square_state(self._prev_phase, pulsewidth)
            self._last_pulsewidth = pulsewidth

        if pulsewidth != self._last_pulsewidth:
            changed_state = self._square_state(self._prev_phase, pulsewidth)
            magnitude = changed_state - self._last_square_state
            if magnitude != 0.0:
                self._insert_discontinuity(1e-6, magnitude)
                self._last_square_state = changed_state
            self._last_pulsewidth = pulsewidth

        start_phase = self._prev_phase
        delta = current_phase - start_phase
        if delta < -0.5:
            current_phase_unwrapped = current_phase + 1.0
        elif delta > 0.5:
            current_phase_unwrapped = current_phase - 1.0
        else:
            current_phase_unwrapped = current_phase

        wrap_subsample = self._crossing_subsample(
            1.0, start_phase, current_phase_unwrapped
        )
        if wrap_subsample is not None:
            self._insert_discontinuity(wrap_subsample, 2.0)

        pulse_subsample = self._crossing_subsample(
            pulsewidth, start_phase, current_phase_unwrapped
        )
        if pulse_subsample is not None:
            self._insert_discontinuity(pulse_subsample, -2.0)

        normalized = self._square_state(current_phase, pulsewidth)
        self._last_square_state = normalized
        self._prev_phase = current_phase
        normalized += self._shift_buffer()
        return self._process_dc_filter(normalized)

    @staticmethod
    def _scale_from_normalized(
        value: float | np.ndarray, low_value: float, high_value: float
    ) -> float | np.ndarray:
        midpoint = (high_value + low_value) / 2.0
        scale = (high_value - low_value) / 2.0
        return midpoint + value * scale

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        normalized = self._process_normalized_sample(phase, pulsewidth_threshold)
        return float(self._scale_from_normalized(normalized, low_value, high_value))

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        normalized = np.asarray(
            [
                self._process_normalized_sample(float(phase), pulsewidth_threshold)
                for phase in phases
            ],
            dtype=np.float32,
        )
        return np.asarray(
            self._scale_from_normalized(normalized, low_value, high_value),
            dtype=np.float32,
        )


class ComparatorSquareStrategy(SquareWaveStrategy):
    """Sine-comparator square wave with optional hysteresis."""

    def __init__(self, hysteresis: float = 0.0) -> None:
        if not 0.0 <= hysteresis <= 1.0:
            raise ValueError(
                f"hysteresis must be between 0.0 and 1.0, got {hysteresis}"
            )
        self.hysteresis = hysteresis
        self._state_high = False
        self._state_initialized = False

    def reset_state(self) -> None:
        self._state_high = False
        self._state_initialized = False

    @staticmethod
    def _comparator_level(phase: float, pulsewidth_threshold: float) -> float:
        pulsewidth = pulsewidth_threshold / TWO_PI
        if pulsewidth <= 0.0:
            return -1.0
        if pulsewidth >= 1.0:
            return 1.0

        center = pulsewidth * np.pi
        threshold = np.cos(center)
        return float(np.cos((phase % TWO_PI) - center) - threshold)

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        level = self._comparator_level(phase, pulsewidth_threshold)
        if not self._state_initialized:
            self._state_high = level >= 0.0
            self._state_initialized = True
        elif self._state_high and level < -self.hysteresis:
            self._state_high = False
        elif not self._state_high and level > self.hysteresis:
            self._state_high = True

        return high_value if self._state_high else low_value

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        samples = np.empty(len(phases), dtype=np.float32)
        for index, phase in enumerate(phases):
            samples[index] = self.generate_sample(
                float(phase), pulsewidth_threshold, low_value, high_value
            )
        return samples


class SquareWaveFactory:
    """Factory for square wave strategy instances."""

    _strategies: dict[str, type[SquareWaveStrategy]] = {
        "ideal": IdealSquareStrategy,
        "ideal_smooth": IdealSquareStrategySmoothing,
        "bandlimited": BandlimitedSquareStrategy,
        "vcv": VCVRackSquareStrategy,
        "soft": SoftSquareStrategy,
        "comparator": ComparatorSquareStrategy,
    }

    @classmethod
    def create(cls, mode: str = "ideal", **kwargs) -> SquareWaveStrategy:
        if mode not in cls._strategies:
            raise ValueError(
                f"Unknown square wave mode: {mode}. "
                f"Available modes: {list(cls._strategies.keys())}"
            )
        logger.debug("Creating square wave strategy with mode: %s", mode)
        strategy_class = cls._strategies[mode]
        return strategy_class(**kwargs)

    @classmethod
    def get_available_modes(cls) -> list[str]:
        return list(cls._strategies.keys())

    @classmethod
    def register_strategy(
        cls, name: str, strategy_class: type[SquareWaveStrategy]
    ) -> None:
        if not issubclass(strategy_class, SquareWaveStrategy):
            raise TypeError(f"{strategy_class} must inherit from SquareWaveStrategy")
        cls._strategies[name] = strategy_class


@register_component()
class SquareOscillator(Oscillator):
    """Square or pulse wave oscillator with selectable generation strategy."""

    descriptor = ComponentDescriptor(
        name="Square",
        category=ComponentCategory.OSCILLATOR,
        description="Square/Pulse wave oscillator with variable pulse width",
        tags=["basic", "oscillator", "square", "pulse"],
        fluent_api_name="square",
        parameters=make_parameter_descriptors(
            "frequency",
            "gain_db",
            "amplitude",
            "phase",
            "sample_rate",
            "wave_range",
            "pulsewidth",
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
        frequency: float = 440,
        amplitude: float = 1.0,
        gain_db: float | None = DEFAULT_GAIN_DB,
        phase: float = 0.0,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        wave_range: tuple[float, float] = (-1, 1),
        pulsewidth: float = 0.5,
        mode: SquareWaveMode = "ideal",
        **mode_kwargs,
    ) -> None:
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
        if not 0.0 <= pulsewidth <= 1.0:
            raise ValueError(
                f"pulsewidth must be between 0.0 and 1.0, got {pulsewidth}"
            )
        self._pulsewidth = pulsewidth
        self._mode = mode
        self._pulsewidth_threshold = pulsewidth * TWO_PI
        self._strategy = SquareWaveFactory.create(mode, **mode_kwargs)
        self._mode_kwargs = mode_kwargs
        self._sync_strategy_runtime()
        self._sync_smoothing_strategy(initial=True)

    @property
    def pulsewidth(self) -> float:
        return self._pulsewidth

    @pulsewidth.setter
    def pulsewidth(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"pulsewidth must be between 0.0 and 1.0, got {value}")
        self._pulsewidth = value
        self._pulsewidth_threshold = value * TWO_PI

    def _post_freq_set(self) -> None:
        self._step = (TWO_PI * self._f) / self._sample_rate
        self._sync_strategy_runtime()

    def _post_amp_set(self) -> None:
        if self._strategy_handles_amplitude():
            strategy = self._strategy
            assert isinstance(strategy, IdealSquareStrategySmoothing)
            strategy.set_amplitude(self._target_amplitude)
            self._smoothing_samples_remaining = 0
            self._current_amplitude = self._target_amplitude

    def _post_phase_set(self) -> None:
        self._p = np.deg2rad(self._p)

    def _post_sample_rate_set(self) -> None:
        self._post_freq_set()
        self._sync_strategy_runtime()

    def _initialize_osc(self) -> None:
        self._i = 0.0
        strategy = getattr(self, "_strategy", None)
        if isinstance(strategy, _ResettableStrategy):
            strategy.reset_state()

    @property
    def mode(self) -> SquareWaveMode:
        return self._mode

    def set_mode(self, mode: SquareWaveMode, **mode_kwargs) -> None:
        self._mode = mode
        self._mode_kwargs = mode_kwargs
        self._strategy = SquareWaveFactory.create(mode, **mode_kwargs)
        self._sync_strategy_runtime()
        self._sync_smoothing_strategy(initial=False)

    @classmethod
    def get_available_modes(cls) -> list[str]:
        return SquareWaveFactory.get_available_modes()

    def _strategy_handles_amplitude(self) -> bool:
        return isinstance(
            getattr(self, "_strategy", None), IdealSquareStrategySmoothing
        )

    def _sync_smoothing_strategy(self, *, initial: bool) -> None:
        if not self._strategy_handles_amplitude():
            return

        strategy = self._strategy
        assert isinstance(strategy, IdealSquareStrategySmoothing)
        strategy.set_sample_rate(self._sample_rate)
        if initial:
            strategy.set_amplitude(self._target_amplitude)
        else:
            strategy.reset_amplitude(self._current_amplitude)
            strategy.set_amplitude(self._target_amplitude)

        self._smoothing_samples_remaining = 0
        self._current_amplitude = self._target_amplitude

    def _sync_strategy_runtime(self) -> None:
        strategy = getattr(self, "_strategy", None)
        if isinstance(strategy, _SampleRateAwareStrategy):
            strategy.set_sample_rate(self._sample_rate)
        if isinstance(strategy, _FrequencyAwareStrategy):
            strategy.set_frequency(self._f)

    def __next__(self) -> float:
        current_phase = (self._i + self._p) % TWO_PI
        val = self._strategy.generate_sample(
            phase=current_phase,
            pulsewidth_threshold=self._pulsewidth_threshold,
            low_value=self._wave_range[0],
            high_value=self._wave_range[1],
        )
        self._i += self._step
        if self._i >= TWO_PI:
            self._i -= TWO_PI
        if self._strategy_handles_amplitude():
            return float(val)
        return float(self._apply_amplitude_to_buffer(np.asarray([val]))[0])

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        n = validate_sample_count(n)
        phases = (self._i + self._p) + self._step * np.arange(n)
        wrapped_phases = phases % TWO_PI
        val = self._strategy.generate_samples(
            phases=wrapped_phases,
            pulsewidth_threshold=self._pulsewidth_threshold,
            low_value=self._wave_range[0],
            high_value=self._wave_range[1],
        )
        if self._strategy_handles_amplitude():
            samples = val
        else:
            samples = self._apply_amplitude_to_buffer(val)
        self._i = (self._i + self._step * n) % TWO_PI
        return samples.astype(np.float32)
