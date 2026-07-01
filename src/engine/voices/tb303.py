"""Composable TB-303 style mono voice."""

from __future__ import annotations

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.dsp.filters.acid_303 import AcidResonantFilter
from src.engine.dsp.modulators import DecayEnvelope
from src.engine.generators.oscillators.oscillator import (
    SawtoothOscillator,
    SquareOscillator,
)
from src.engine.sequencing import AccentProcessor, SlideProcessor
from src.engine.utils.validation import validate_sample_count, validate_sample_rate


class TB303Voice:
    """Convenience acid-bass voice composed from reusable primitives."""

    def __init__(
        self,
        *,
        waveform: str = "Sawtooth",
        tuning: float = 1.0,
        pulsewidth: float = 0.5,
        cutoff: float = 700.0,
        resonance: float = 8.0,
        env_amount: float = 2.5,
        decay: float = 0.18,
        accent: float = 0.7,
        slide_time: float = 0.08,
        drive_db: float = 6.0,
        volume: float = 0.8,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.sample_rate = validate_sample_rate(sample_rate)
        self.waveform = waveform
        self.tuning = float(tuning)
        self.pulsewidth = float(pulsewidth)
        self.cutoff = float(cutoff)
        self.resonance = float(resonance)
        self.env_amount = float(env_amount)
        self.decay = float(decay)
        self.accent = float(accent)
        self.slide_time = float(slide_time)
        self.drive_db = float(drive_db)
        self.volume = float(volume)
        self._previous_gate = 0.0
        self._oscillator = self._create_oscillator(110.0)
        self._slide = SlideProcessor(time=slide_time, sample_rate=sample_rate)
        self._envelope = DecayEnvelope(decay_duration=decay, sample_rate=sample_rate)
        self._accent = AccentProcessor(
            amount=accent,
            amp_depth=0.35,
            cutoff_depth=0.7,
            envelope_depth=0.5,
            sample_rate=sample_rate,
        )
        self._filter = AcidResonantFilter(sample_rate=sample_rate)

    def reset_state(self) -> None:
        """Reset oscillator-adjacent processors and envelope/filter state."""
        self._previous_gate = 0.0
        self._slide.reset()
        self._envelope.reset()
        self._accent.reset()
        self._filter.reset_state()

    def process(
        self,
        *,
        frequency: np.ndarray,
        gate: np.ndarray,
        accent: np.ndarray | None = None,
        slide: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render one mono audio buffer."""
        num_samples = validate_sample_count(len(frequency))
        frequencies = self._fit_signal(frequency, num_samples)
        gates = self._fit_signal(gate, num_samples)
        accents = self._fit_signal(accent, num_samples)
        slides = self._fit_signal(slide, num_samples)

        self._sync_parameters()
        pitch = self._slide.process(frequencies * self.tuning, slides)
        oscillator = self._render_oscillator(pitch)
        envelope = self._render_envelope(gates)
        accent_frame = self._accent.process(accents)
        filtered = self._filter.process_modulated(
            oscillator,
            env_cv=np.clip(envelope + accent_frame.envelope, 0.0, 1.0),
            accent_cv=accent_frame.cutoff,
        )
        amplitude = np.clip(envelope + accent_frame.amp, 0.0, 1.0)
        return (filtered * amplitude * self.volume).astype(np.float32)

    def _sync_parameters(self) -> None:
        if self._oscillator_shape() != self._current_oscillator_shape():
            frequency = getattr(self._oscillator, "frequency", 110.0)
            self._oscillator = self._create_oscillator(frequency)
        if isinstance(self._oscillator, SquareOscillator):
            self._oscillator.pulsewidth = self.pulsewidth
        self._slide.time = self.slide_time
        self._envelope.decay_duration = self.decay
        self._envelope.amount = 1.0
        self._accent.amount = self.accent
        self._filter.cutoff = self.cutoff
        self._filter.resonance = self.resonance
        self._filter.env_amount = self.env_amount
        self._filter.accent_amount = 1.0
        self._filter.drive_db = self.drive_db

    def _render_oscillator(self, frequencies: np.ndarray) -> np.ndarray:
        output = np.empty(len(frequencies), dtype=np.float32)
        for index, frequency in enumerate(frequencies):
            self._oscillator.frequency = float(np.clip(frequency, 1.0, 20000.0))
            output[index] = next(self._oscillator)
        return output

    def _render_envelope(self, gate: np.ndarray) -> np.ndarray:
        output = np.zeros(len(gate), dtype=np.float32)
        start = 0
        for index, value in enumerate(gate):
            current_gate = float(value)
            note_on = self._previous_gate < 0.3 and current_gate > 0.7
            if note_on:
                if index > start:
                    output[start:index] = self._envelope.get_samples(index - start)
                self._envelope.trigger_note_on()
                start = index
            self._previous_gate = current_gate
        if start < len(gate):
            output[start:] = self._envelope.get_samples(len(gate) - start)
        return output

    def _create_oscillator(self, frequency: float):
        if self.waveform == "Square":
            return SquareOscillator(
                frequency,
                gain_db=0.0,
                pulsewidth=self.pulsewidth,
                sample_rate=self.sample_rate,
                mode="vcv",
            )
        return SawtoothOscillator(frequency, gain_db=0.0, sample_rate=self.sample_rate)

    def _oscillator_shape(self) -> tuple[str, float]:
        return (self.waveform, self.pulsewidth)

    def _current_oscillator_shape(self) -> tuple[str, float]:
        return (
            "Square" if isinstance(self._oscillator, SquareOscillator) else "Sawtooth",
            getattr(self._oscillator, "pulsewidth", self.pulsewidth),
        )

    @staticmethod
    def _fit_signal(values: np.ndarray | None, num_samples: int) -> np.ndarray:
        if values is None:
            return np.zeros(num_samples, dtype=np.float32)
        signal = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(signal) == num_samples:
            return signal
        if len(signal) > num_samples:
            return signal[:num_samples]
        padded = np.zeros(num_samples, dtype=np.float32)
        padded[: len(signal)] = signal
        if len(signal):
            padded[len(signal) :] = signal[-1]
        return padded
