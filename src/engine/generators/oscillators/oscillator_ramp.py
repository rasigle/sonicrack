"""Ramp-based oscillators such as sawtooth and triangle."""

from typing import Literal, cast

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    ComponentDescriptor,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.core.registry import ComponentCategory, register_component
from src.engine.generators.oscillators.oscillator_base import Oscillator
from src.engine.generators.oscillators.oscillator_minblep import (
    VCV_MINBLEP_ZERO_CROSSINGS,
    compute_dc_alpha,
    crossing_subsample,
    insert_minblep_discontinuity,
    minimum_phase_minblep_table,
    process_dc_filter,
    shift_minblep_buffer,
)
from src.engine.utils.decorators import filter_provided_args, track_provided_args
from src.engine.utils.validation import validate_sample_count

SawtoothMode = Literal["pure", "analog", "vcv"]
TriangleMode = Literal["pure", "analog"]


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

        self._mode: SawtoothMode = "pure"
        self.set_mode(mode)
        self._phase_degrees = self._p
        self._update_dc_alpha()

    def _post_freq_set(self):
        old_period = cast(float | None, getattr(self, "_period", None))
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
        self._dc_alpha = compute_dc_alpha(self._sample_rate)

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

    def _apply_analog_character_buffer(
        self, values: np.ndarray, sample_indices: np.ndarray | None = None
    ) -> np.ndarray:
        return np.asarray(
            self._apply_analog_character(values, sample_indices), dtype=np.float32
        )

    def _insert_vcv_discontinuity(self, subsample: float, magnitude: float) -> None:
        insert_minblep_discontinuity(
            self._vcv_buffer, self._minblep_table, subsample, magnitude
        )

    def _shift_vcv_buffer(self) -> float:
        return shift_minblep_buffer(self._vcv_buffer)

    def _process_vcv_dc_filter(self, value: float) -> float:
        result, self._dc_lowpass_state = process_dc_filter(
            value, self._dc_lowpass_state, self._dc_alpha, self.dc_block
        )
        return result

    @staticmethod
    def _saw_state(phase: float) -> float:
        return float(2.0 * (phase - np.floor(0.5 + phase)))

    def _process_vcv_normalized_sample(self, phase: float) -> float:
        current_phase = phase % 1.0
        if self._vcv_prev_phase is None:
            increment = self._f / self._sample_rate
            self._vcv_prev_phase = (current_phase - increment) % 1.0

        start_phase = cast(float, self._vcv_prev_phase)
        delta = current_phase - start_phase
        if delta < -0.5:
            current_phase_unwrapped = current_phase + 1.0
        elif delta > 0.5:
            current_phase_unwrapped = current_phase - 1.0
        else:
            current_phase_unwrapped = current_phase

        phase_delta = current_phase_unwrapped - start_phase
        wrap_subsample = crossing_subsample(0.5, start_phase, current_phase_unwrapped)
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
            val = float(2 * (div - np.floor(0.5 + div)))

        if self._mode == "analog":
            val = self._apply_analog_character(val)
        val = float(val)
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
        val = np.asarray(val, dtype=np.float32)
        if self._mode == "analog":
            val = self._apply_analog_character_buffer(val, indices)
        val = self._apply_wave_range_values(val)
        samples = self._apply_amplitude_to_buffer(val)
        self._i += n
        return samples.astype(np.float32)

    def render_modulated_waveform(
        self,
        freqs: np.ndarray,
        phase_offsets_deg: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Render an unamplified waveform for per-sample modulation."""
        increments = freqs / self.sample_rate
        start_cycle = self._i / self._period if self._period != 0 else 0.0
        phase_offsets = np.concatenate(
            ([0.0], np.cumsum(increments[:-1], dtype=np.float64))
        )
        carrier_cycles = start_cycle + phase_offsets
        carrier_end = float(start_cycle + float(np.sum(increments, dtype=np.float64)))

        if phase_offsets_deg is None:
            offset_cycles = np.full(
                len(freqs),
                self._p / self._period if self._period != 0 else 0.0,
                dtype=np.float64,
            )
        else:
            offset_cycles = phase_offsets_deg / 360.0

        cycles = carrier_cycles + offset_cycles
        if self.mode == "vcv":
            waveform = self._generate_vcv_from_cycles(cycles)
        else:
            waveform = 2 * (cycles - np.floor(0.5 + cycles))

        if self.mode == "analog":
            sample_indices = self._i + np.arange(len(freqs), dtype=np.float64)
            waveform = np.asarray(
                self._apply_analog_character(waveform, sample_indices),
                dtype=np.float64,
            )

        waveform = np.asarray(self._apply_wave_range_values(waveform), dtype=np.float64)
        return waveform, {"carrier_cycle": carrier_end}

    def commit_modulated_phase_state(self, state: dict[str, float]) -> None:
        """Commit phase state produced by ``render_modulated_waveform``."""
        carrier_cycle = float(state["carrier_cycle"]) % 1.0
        self._i = carrier_cycle * self._period if self._period != 0 else 0.0


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

    @track_provided_args
    def __init__(
        self,
        frequency: float = 440,
        amplitude: float = 1.0,
        gain_db: float | None = DEFAULT_GAIN_DB,
        phase: float = 0.0,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        wave_range: tuple[float, float] = (-1, 1),
        mode: TriangleMode = "pure",
    ):
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
        Oscillator.__init__(self, **kwargs)

        self._mode: TriangleMode = "pure"
        self.set_mode(mode)
        self._phase_degrees = self._p

    def _initialize_osc(self):
        self._i = 0

    @property
    def mode(self) -> TriangleMode:
        return self._mode

    @mode.setter
    def mode(self, value: TriangleMode):
        self.set_mode(value)

    def set_mode(self, mode: TriangleMode) -> None:
        if mode not in self.get_available_modes():
            raise ValueError(f"Invalid mode '{mode}'. Must be 'pure' or 'analog'")
        self._mode = mode

    @classmethod
    def get_available_modes(cls) -> list[str]:
        return ["pure", "analog"]

    def _post_sample_rate_set(self):
        self._post_freq_set()

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

    def _apply_analog_character_triangle_buffer(
        self, values: np.ndarray, sample_indices: np.ndarray | None = None
    ) -> np.ndarray:
        return np.asarray(
            self._apply_analog_character_triangle(values, sample_indices),
            dtype=np.float32,
        )

    def __next__(self):
        div = (self._i + self._p) / self._period if self._period != 0 else 0
        val = 2 * (div - np.floor(0.5 + div))
        val = (abs(val) - 0.5) * 2
        if self._mode == "analog":
            val = self._apply_analog_character_triangle(val)
        val = float(val)
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
        val = np.asarray(val, dtype=np.float32)
        if self._mode == "analog":
            val = self._apply_analog_character_triangle_buffer(val, indices)
        val = self._apply_wave_range_values(val)
        samples = self._apply_amplitude_to_buffer(val)
        self._i += n
        return samples.astype(np.float32)

    def render_modulated_waveform(
        self,
        freqs: np.ndarray,
        phase_offsets_deg: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Render an unamplified waveform for per-sample modulation."""
        increments = freqs / self.sample_rate
        start_cycle = self._i / self._period if self._period != 0 else 0.0
        phase_offsets = np.concatenate(
            ([0.0], np.cumsum(increments[:-1], dtype=np.float64))
        )
        carrier_cycles = start_cycle + phase_offsets
        carrier_end = float(start_cycle + float(np.sum(increments, dtype=np.float64)))

        if phase_offsets_deg is None:
            offset_cycles = np.full(
                len(freqs),
                self._p / self._period if self._period != 0 else 0.0,
                dtype=np.float64,
            )
        else:
            offset_cycles = phase_offsets_deg / 360.0

        cycles = carrier_cycles + offset_cycles
        waveform = 2 * (cycles - np.floor(0.5 + cycles))
        waveform = (np.abs(waveform) - 0.5) * 2
        if self.mode == "analog":
            sample_indices = self._i + np.arange(len(freqs), dtype=np.float64)
            waveform = np.asarray(
                self._apply_analog_character_triangle(waveform, sample_indices),
                dtype=np.float64,
            )

        waveform = np.asarray(self._apply_wave_range_values(waveform), dtype=np.float64)
        return waveform, {"carrier_cycle": carrier_end}
