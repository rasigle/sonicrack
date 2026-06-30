"""Ramp-based oscillators such as sawtooth and triangle."""

import math
from typing import Literal

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.audio_component import (
    ComponentDescriptor,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.audio_component_registry import ComponentCategory, register_component
from src.engine.generator.oscillator_base import Oscillator
from src.engine.generator.oscillator_minblep import (
    TWO_PI,
    VCV_MINBLEP_OVERSAMPLE,
    VCV_MINBLEP_ZERO_CROSSINGS,
    minimum_phase_minblep_table,
)
from src.engine.validation import validate_sample_count
from src.utils.utils import filter_provided_args, track_provided_args

SawtoothMode = Literal["pure", "analog", "vcv"]


@register_component()
class SawtoothOscillator(Oscillator):
    """Sawtooth wave generator with optional analog/minBLEP coloration."""

    _minblep_table = minimum_phase_minblep_table()

    descriptor = ComponentDescriptor(
        name="Sawtooth",
        category=ComponentCategory.OSCILLATOR,
        description="Sawtooth wave oscillator with analog and VCV-style modes",
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
                choices=("pure", "analog", "vcv"),
                description="Ramp oscillator generation mode.",
            ),
        ),
        tags=["basic", "oscillator", "sawtooth", "analog", "vcv"],
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
        mode: SawtoothMode = "pure",
        dc_block: bool = True,
    ):
        self.dc_block = dc_block
        self._vcv_buffer = np.zeros(2 * VCV_MINBLEP_ZERO_CROSSINGS, dtype=np.float32)
        self._vcv_prev_phase: float | None = None
        self._dc_lowpass_state = 0.0
        self.set_mode(mode)
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
        self._update_dc_alpha()

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
        self._vcv_buffer.fill(0.0)
        self._vcv_prev_phase = None
        self._dc_lowpass_state = 0.0

    @property
    def mode(self) -> SawtoothMode:
        return self._mode

    @mode.setter
    def mode(self, value: SawtoothMode):
        self.set_mode(value)

    def set_mode(self, mode: SawtoothMode) -> None:
        if mode not in self.get_available_modes():
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'pure', 'analog', or 'vcv'"
            )
        self._mode = mode

    @classmethod
    def get_available_modes(cls) -> list[str]:
        return ["pure", "analog", "vcv"]

    def _post_sample_rate_set(self):
        self._post_freq_set()
        self._update_dc_alpha()

    def _update_dc_alpha(self) -> None:
        cutoff = min(0.4, 20.0 / self._sample_rate)
        w = TWO_PI * cutoff
        self._dc_alpha = w / (1.0 + w)

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

    def _insert_vcv_discontinuity(self, subsample: float, magnitude: float) -> None:
        if not 0.0 < subsample <= 1.0 or magnitude == 0.0:
            return
        table = self._minblep_table
        extended_table = np.concatenate((table, np.zeros(1, dtype=np.float32)))
        offset = (1.0 - subsample) * VCV_MINBLEP_OVERSAMPLE
        for index in range(len(self._vcv_buffer)):
            position = index * VCV_MINBLEP_OVERSAMPLE + offset
            lower = int(position)
            fraction = position - lower
            value = extended_table[lower] + fraction * (
                extended_table[lower + 1] - extended_table[lower]
            )
            self._vcv_buffer[index] += magnitude * value

    def _shift_vcv_buffer(self) -> float:
        value = float(self._vcv_buffer[0])
        self._vcv_buffer[:-1] = self._vcv_buffer[1:]
        self._vcv_buffer[-1] = 0.0
        return value

    def _process_vcv_dc_filter(self, value: float) -> float:
        if not self.dc_block:
            return value
        self._dc_lowpass_state += self._dc_alpha * (value - self._dc_lowpass_state)
        return value - self._dc_lowpass_state

    @staticmethod
    def _saw_state(phase: float) -> float:
        return float(2.0 * (phase - np.floor(0.5 + phase)))

    def _process_vcv_normalized_sample(self, phase: float) -> float:
        current_phase = phase % 1.0
        if self._vcv_prev_phase is None:
            increment = self._f / self._sample_rate
            self._vcv_prev_phase = (current_phase - increment) % 1.0

        start_phase = self._vcv_prev_phase
        delta = current_phase - start_phase
        if delta < -0.5:
            current_phase_unwrapped = current_phase + 1.0
        elif delta > 0.5:
            current_phase_unwrapped = current_phase - 1.0
        else:
            current_phase_unwrapped = current_phase

        phase_delta = current_phase_unwrapped - start_phase
        wrap_subsample = self._crossing_subsample(
            0.5, start_phase, current_phase_unwrapped
        )
        if wrap_subsample is not None:
            self._insert_vcv_discontinuity(
                wrap_subsample,
                -2.0 if phase_delta > 0.0 else 2.0,
            )

        normalized = self._saw_state(current_phase)
        self._vcv_prev_phase = current_phase
        normalized += self._shift_vcv_buffer()
        return self._process_vcv_dc_filter(normalized)

    def _generate_vcv_from_cycles(self, cycles: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self._process_vcv_normalized_sample(float(cycle)) for cycle in cycles],
            dtype=np.float32,
        )

    def __next__(self):
        div = (self._i + self._p) / self._period if self._period != 0 else 0
        if self._mode == "vcv":
            val = self._process_vcv_normalized_sample(div)
        else:
            val = 2 * (div - np.floor(0.5 + div))
        if self._mode == "analog":
            val = self._apply_analog_character(val)
        self._i = self._i + 1
        val = self._apply_wave_range_value(val)
        return val * self._a

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        n = validate_sample_count(n)
        indices = np.arange(n, dtype=np.float64) + self._i
        if self._period != 0:
            div = (indices + self._p) / self._period
            if self._mode == "vcv":
                val = self._generate_vcv_from_cycles(div)
            else:
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

    def set_mode(self, mode: Literal["pure", "analog"]) -> None:
        if mode not in self.get_available_modes():
            raise ValueError(f"Invalid mode '{mode}'. Must be 'pure' or 'analog'")
        self._mode = mode

    @classmethod
    def get_available_modes(cls) -> list[str]:
        return ["pure", "analog"]

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
        indices = np.arange(n, dtype=np.float64) + self._i
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
