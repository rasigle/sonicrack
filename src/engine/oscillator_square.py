"""Square-wave strategies and oscillator implementation."""

import logging
from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.audio_component import ComponentDescriptor
from src.engine.audio_component_registry import ComponentCategory, register_component
from src.engine.oscillator_sine import SineOscillator
from src.utils.utils import filter_provided_args, track_provided_args

SquareWaveMode = Literal["ideal", "ideal_smooth", "soft"]


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

    def __init__(self, **kwargs):
        pass

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

    def __init__(self, smoothing_time_ms: float = 5.0, sample_rate: float = 44100):
        self.smoothing_time_ms = smoothing_time_ms
        self.sample_rate = sample_rate
        self._current_amplitude = 1.0
        self._target_amplitude = 1.0
        self._smoothing_samples_remaining = 0

    def set_amplitude(self, amplitude: float):
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
        val = high_value if phase < pulsewidth_threshold else low_value
        if self._smoothing_samples_remaining > 0:
            progress = 1.0 - (
                self._smoothing_samples_remaining
                / (self.smoothing_time_ms * self.sample_rate / 1000)
            )
            current_amp = (
                self._current_amplitude
                + (self._target_amplitude - self._current_amplitude) * progress
            )
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
        val = np.where(phases < pulsewidth_threshold, high_value, low_value)
        if self._smoothing_samples_remaining > 0:
            smooth_count = min(n, self._smoothing_samples_remaining)
            amp_envelope = np.linspace(
                self._current_amplitude, self._target_amplitude, smooth_count
            )
            samples = np.zeros(n, dtype=np.float32)
            samples[:smooth_count] = val[:smooth_count] * amp_envelope
            if smooth_count < n:
                samples[smooth_count:] = val[smooth_count:] * self._target_amplitude
            self._smoothing_samples_remaining -= smooth_count
            if self._smoothing_samples_remaining <= 0:
                self._current_amplitude = self._target_amplitude
            return samples
        return (val * self._current_amplitude).astype(np.float32)


class SoftSquareStrategy(SquareWaveStrategy):
    """Square wave with tanh-smoothed transitions."""

    def __init__(self, smoothness: float = 10.0):
        self.smoothness = smoothness

    def generate_sample(
        self,
        phase: float,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> float:
        dist_from_threshold = phase - pulsewidth_threshold
        smooth_step = np.tanh(-dist_from_threshold * self.smoothness)
        return low_value + (high_value - low_value) * (smooth_step + 1) / 2

    def generate_samples(
        self,
        phases: np.ndarray,
        pulsewidth_threshold: float,
        low_value: float,
        high_value: float,
    ) -> np.ndarray:
        dist_from_threshold = phases - pulsewidth_threshold
        smooth_step = np.tanh(-dist_from_threshold * self.smoothness)
        return low_value + (high_value - low_value) * (smooth_step + 1) / 2


class SquareWaveFactory:
    """Factory for square wave strategy instances."""

    _strategies = {
        "ideal": IdealSquareStrategy,
        "ideal_smooth": IdealSquareStrategySmoothing,
        "soft": SoftSquareStrategy,
    }

    @classmethod
    def create(cls, mode: SquareWaveMode = "ideal", **kwargs) -> SquareWaveStrategy:
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
        return list(cls._strategies.keys())

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type[SquareWaveStrategy]):
        if not issubclass(strategy_class, SquareWaveStrategy):
            raise TypeError(f"{strategy_class} must inherit from SquareWaveStrategy")
        cls._strategies[name] = strategy_class


@register_component()
class SquareOscillator(SineOscillator):
    """Square or pulse wave oscillator with selectable generation strategy."""

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
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        wave_range: tuple[float, float] = (-1, 1),
        pulsewidth: float = 0.5,
        mode: SquareWaveMode = "ideal",
        **mode_kwargs,
    ):
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
        self._pulsewidth_threshold = pulsewidth * 2 * np.pi
        self._strategy = SquareWaveFactory.create(mode, **mode_kwargs)
        self._mode_kwargs = mode_kwargs

    @property
    def pulsewidth(self) -> float:
        return self._pulsewidth

    @pulsewidth.setter
    def pulsewidth(self, value: float):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"pulsewidth must be between 0.0 and 1.0, got {value}")
        self._pulsewidth = value
        self._pulsewidth_threshold = value * 2 * np.pi

    @property
    def mode(self) -> SquareWaveMode:
        return self._mode

    def set_mode(self, mode: SquareWaveMode, **mode_kwargs):
        self._mode = mode
        self._mode_kwargs = mode_kwargs
        self._strategy = SquareWaveFactory.create(mode, **mode_kwargs)

    @classmethod
    def get_available_modes(cls) -> list[str]:
        return SquareWaveFactory.get_available_modes()

    def __next__(self):
        current_phase = (self._i + self._p) % (2 * np.pi)
        val = self._strategy.generate_sample(
            phase=current_phase,
            pulsewidth_threshold=self._pulsewidth_threshold,
            low_value=self._wave_range[0],
            high_value=self._wave_range[1],
        )
        self._i += self._step
        if self._i >= 2 * np.pi:
            self._i -= 2 * np.pi
        return val * self._a

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        phases = (self._i + self._p) + self._step * np.arange(n)
        wrapped_phases = phases % (2 * np.pi)
        val = self._strategy.generate_samples(
            phases=wrapped_phases,
            pulsewidth_threshold=self._pulsewidth_threshold,
            low_value=self._wave_range[0],
            high_value=self._wave_range[1],
        )
        samples = self._apply_amplitude_to_buffer(val)
        self._i = (self._i + self._step * n) % (2 * np.pi)
        return samples.astype(np.float32)
