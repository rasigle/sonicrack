"""Ramp-based oscillators such as sawtooth and triangle."""

from typing import Literal

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.audio_component import (
    ComponentDescriptor,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.audio_component_registry import ComponentCategory, register_component
from src.engine.oscillator_base import Oscillator
from src.engine.validation import validate_sample_count
from src.utils.utils import filter_provided_args, track_provided_args


@register_component()
class SawtoothOscillator(Oscillator):
    """Sawtooth wave generator with optional analog coloration."""

    descriptor = ComponentDescriptor(
        name="Sawtooth",
        category=ComponentCategory.OSCILLATOR,
        description="Sawtooth wave oscillator with analog mode",
        fluent_api_name="sawtooth",
        parameters=make_parameter_descriptors(
            "frequency",
            "gain_db",
            "amplitude",
            "phase",
            "sample_rate",
            "wave_range",
            "mode",
            mode=ParameterDescriptor(
                name="mode",
                default="pure",
                choices=("pure", "analog"),
                description="Ramp oscillator generation mode.",
            ),
        ),
        tags=["basic", "oscillator", "sawtooth", "analog"],
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
        mode: Literal["pure", "analog"] = "pure",
    ):
        self._mode = mode
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
        self._phase_degrees = self._p

    def _post_freq_set(self):
        old_period = getattr(self, "_period", None)
        self._period = self._sample_rate / self._f
        if not hasattr(self, "_phase_degrees"):
            self._phase_degrees = 0.0
        self._p = (self._phase_degrees / 360) * self._period
        if old_period is not None and old_period > 0 and self._period > 0:
            phase_fraction = (self._i % old_period) / old_period
            self._i = phase_fraction * self._period

    def _post_phase_set(self):
        if not hasattr(self, "_phase_degrees"):
            self._phase_degrees = 0.0
        self._phase_degrees = self._p
        self._p = (self._p / 360) * self._period

    def _initialize_osc(self):
        self._i = 0

    @property
    def mode(self) -> Literal["pure", "analog"]:
        return self._mode

    @mode.setter
    def mode(self, value: Literal["pure", "analog"]):
        if value not in ["pure", "analog"]:
            raise ValueError(f"Invalid mode '{value}'. Must be 'pure' or 'analog'")
        self._mode = value

    @classmethod
    def get_available_modes(cls) -> list[str]:
        return ["pure", "analog"]

    def _apply_analog_character(
        self, val: float | np.ndarray, sample_indices: np.ndarray | None = None
    ) -> float | np.ndarray:
        analog = np.tanh(val * 1.15)
        if isinstance(val, np.ndarray):
            if sample_indices is None:
                sample_indices = np.arange(len(val), dtype=np.float64)
            phase_mod = np.sin(sample_indices * 0.1) * 0.03
            analog = analog * (1.0 + phase_mod)
        else:
            analog = analog * 1.02
        return analog * 0.92

    def __next__(self):
        div = (self._i + self._p) / self._period if self._period != 0 else 0
        val = 2 * (div - np.floor(0.5 + div))
        if self._mode == "analog":
            val = self._apply_analog_character(val)
        self._i = self._i + 1
        val = self._apply_wave_range_value(val)
        return val * self._a

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        n = validate_sample_count(n)
        indices = np.arange(n, dtype=np.float32) + self._i
        if self._period != 0:
            div = (indices + self._p) / self._period
            val = 2 * (div - np.floor(0.5 + div))
        else:
            val = np.zeros(n, dtype=np.float32)
        if self._mode == "analog":
            val = self._apply_analog_character(val, indices)
        val = self._apply_wave_range_values(val)
        samples = self._apply_amplitude_to_buffer(val)
        self._i += n
        return samples.astype(np.float32)


@register_component()
class TriangleOscillator(SawtoothOscillator):
    """Triangle wave generator with optional analog coloration."""

    descriptor = ComponentDescriptor(
        name="Triangle",
        category=ComponentCategory.OSCILLATOR,
        description="Triangle wave oscillator with analog mode",
        tags=["basic", "oscillator", "triangle", "analog"],
        fluent_api_name="triangle",
        parameters=make_parameter_descriptors(
            "frequency",
            "gain_db",
            "amplitude",
            "phase",
            "sample_rate",
            "wave_range",
            "mode",
            mode=ParameterDescriptor(
                name="mode",
                default="pure",
                choices=("pure", "analog"),
                description="Triangle oscillator generation mode.",
            ),
        ),
    )

    def _apply_analog_character_triangle(
        self, val: float | np.ndarray, sample_indices: np.ndarray | None = None
    ) -> float | np.ndarray:
        analog = np.tanh(val * 1.08)
        if isinstance(val, np.ndarray):
            if sample_indices is None:
                sample_indices = np.arange(len(val), dtype=np.float64)
            phase_mod = np.sin(sample_indices * 0.08) * 0.02
            analog = analog * (1.0 + phase_mod)
        else:
            analog = analog * 1.01
        return analog * 0.95

    def __next__(self):
        div = (self._i + self._p) / self._period if self._period != 0 else 0
        val = 2 * (div - np.floor(0.5 + div))
        val = (abs(val) - 0.5) * 2
        if self._mode == "analog":
            val = self._apply_analog_character_triangle(val)
        self._i = self._i + 1
        val = self._apply_wave_range_value(val)
        return val * self._a

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        n = validate_sample_count(n)
        indices = np.arange(n, dtype=np.float32) + self._i
        if self._period != 0:
            div = (indices + self._p) / self._period
            val = 2 * (div - np.floor(0.5 + div))
            val = (np.abs(val) - 0.5) * 2
        else:
            val = np.zeros(n, dtype=np.float32)
        if self._mode == "analog":
            val = self._apply_analog_character_triangle(val, indices)
        val = self._apply_wave_range_values(val)
        samples = self._apply_amplitude_to_buffer(val)
        self._i += n
        return samples.astype(np.float32)
