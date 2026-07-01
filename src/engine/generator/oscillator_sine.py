"""Sine-family oscillator implementations."""

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
from src.engine.utils import filter_provided_args, track_provided_args
from src.engine.validation import validate_sample_count


@register_component()
class SineOscillator(Oscillator):
    """Sine wave generator with multiple harmonic modes."""

    descriptor = ComponentDescriptor(
        name="Sine",
        category=ComponentCategory.OSCILLATOR,
        description="Pure sine wave oscillator with harmonic modes",
        tags=["basic", "oscillator", "sine", "harmonics", "analog"],
        fluent_api_name="sine",
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
                choices=("pure", "warm", "bright", "analog"),
                description="Sine oscillator harmonic mode.",
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
        mode: Literal["pure", "warm", "bright", "analog"] = "pure",
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
        super().__init__(**kwargs)

    def _post_freq_set(self):
        self._step = (2 * np.pi * self._f) / self._sample_rate

    def _post_phase_set(self):
        self._p = np.deg2rad(self._p)

    def _initialize_osc(self):
        self._i = 0
        self._sample_index = 0

    @property
    def mode(self) -> Literal["pure", "warm", "bright", "analog"]:
        return self._mode

    @mode.setter
    def mode(self, value: Literal["pure", "warm", "bright", "analog"]):
        self.set_mode(value)

    def set_mode(self, mode: Literal["pure", "warm", "bright", "analog"]) -> None:
        if mode not in self.get_available_modes():
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'pure', 'warm', 'bright', or 'analog'"
            )
        self._mode = mode

    @classmethod
    def get_available_modes(cls) -> list[str]:
        return ["pure", "warm", "bright", "analog"]

    def _generate_waveform(
        self,
        phase: float | np.ndarray,
        sample_indices: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        if self._mode == "pure":
            return self._generate_pure_sine(phase)
        if self._mode == "warm":
            return self._generate_warm_sine(phase)
        if self._mode == "bright":
            return self._generate_bright_sine(phase)
        return self._generate_analog_sine(phase, sample_indices)

    def _generate_pure_sine(self, phase: float | np.ndarray) -> float | np.ndarray:
        return np.sin(phase)

    def _generate_warm_sine(self, phase: float | np.ndarray) -> float | np.ndarray:
        fundamental = np.sin(phase)
        harmonic_2 = np.sin(2 * phase) * 0.30
        harmonic_4 = np.sin(4 * phase) * 0.15
        harmonic_6 = np.sin(6 * phase) * 0.08
        warm = fundamental + harmonic_2 + harmonic_4 + harmonic_6
        warm = np.tanh(warm * 0.95)
        return warm * 0.90

    def _generate_bright_sine(self, phase: float | np.ndarray) -> float | np.ndarray:
        fundamental = np.sin(phase)
        harmonic_2 = np.sin(2 * phase) * 0.18
        harmonic_3 = np.sin(3 * phase) * 0.25
        harmonic_5 = np.sin(5 * phase) * 0.15
        harmonic_7 = np.sin(7 * phase) * 0.10
        harmonic_9 = np.sin(9 * phase) * 0.06
        bright = (
            fundamental + harmonic_2 + harmonic_3 + harmonic_5 + harmonic_7 + harmonic_9
        )
        bright = bright + (bright**3) * 0.08
        return bright * 0.75

    def _generate_analog_sine(
        self,
        phase: float | np.ndarray,
        sample_indices: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        fundamental = np.sin(phase)
        second_harmonic = np.sin(2 * phase) * 0.08
        third_harmonic = np.sin(3 * phase) * 0.04
        vcv = fundamental + second_harmonic + third_harmonic
        vcv = np.tanh(vcv * 1.1)
        if sample_indices is not None:
            phase_mod = np.sin(sample_indices * 0.5) * 0.02
        else:
            phase_mod = np.sin(phase * 0.5) * 0.02
        vcv = vcv * (1.0 + phase_mod)
        return vcv * 0.88

    def __next__(self):
        current_phase = self._i + self._p
        val = self._generate_waveform(current_phase, self._sample_index)
        self._i += self._step
        if self._i >= 2 * np.pi:
            self._i -= 2 * np.pi
        self._sample_index += 1
        val = self._apply_wave_range_value(val)
        return val * self._a

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        n = validate_sample_count(n)
        sample_indices = np.arange(n, dtype=np.float64)
        phases = (self._i + self._p) + self._step * sample_indices
        absolute_indices = self._sample_index + sample_indices
        val = np.asarray(self._generate_waveform(phases, absolute_indices))
        val = self._apply_wave_range_values(val)
        samples = self._apply_amplitude_to_buffer(val)
        self._i = (self._i + self._step * n) % (2 * np.pi)
        self._sample_index += n
        return samples.astype(np.float32)
