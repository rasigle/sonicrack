"""Monophonic step sequencer primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.sequencing.clock import StepClock
from src.engine.utils.cv import midi_note_to_pitch_cv
from src.engine.utils.validation import validate_sample_count, validate_sample_rate


@dataclass(slots=True)
class StepEvent:
    """One monophonic sequencer step."""

    note: int | None
    gate: bool = True
    accent: bool = False
    slide: bool = False
    tie: bool = False
    gate_length: float = 0.8


@dataclass(slots=True)
class SequencerFrame:
    """Rendered CV outputs for one audio buffer."""

    # Kept as ``frequency`` for compatibility with existing module port names.
    # Values are 1V/oct pitch CV, not Hz.
    frequency: np.ndarray
    gate: np.ndarray
    accent: np.ndarray
    slide: np.ndarray


class StepSequencer:
    """Render monophonic pitch, gate, accent, and slide CV from a pattern."""

    def __init__(
        self,
        pattern: list[StepEvent] | None = None,
        *,
        bpm: float = 120.0,
        division: str = "1/16",
        swing: float = 0.0,
        transpose: int = 0,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.sample_rate = validate_sample_rate(sample_rate)
        self.pattern = pattern or [StepEvent(36)]
        self.transpose = int(transpose)
        self.clock = StepClock(
            bpm=bpm,
            division=division,
            swing=swing,
            sample_rate=self.sample_rate,
        )
        self._active_step = -1
        self._samples_in_step = 0
        self._current_frequency = midi_note_to_pitch_cv(36)
        self._current_event = StepEvent(36)

    def reset(self) -> None:
        """Reset pattern and clock state."""
        self.clock.reset()
        self._active_step = -1
        self._samples_in_step = 0
        self._current_frequency = midi_note_to_pitch_cv(36)
        self._current_event = StepEvent(36)

    @property
    def step_samples(self) -> int:
        return self.clock.step_samples

    def process(
        self, num_samples: int, clock_pulses: np.ndarray | None = None
    ) -> SequencerFrame:
        """Render sequencer CV outputs for one buffer."""
        num_samples = validate_sample_count(num_samples)
        if clock_pulses is None:
            clock_pulses = self.clock.process(num_samples)
        else:
            clock_pulses = self._fit_signal(clock_pulses, num_samples)

        frequency = np.empty(num_samples, dtype=np.float32)
        gate = np.zeros(num_samples, dtype=np.float32)
        accent = np.zeros(num_samples, dtype=np.float32)
        slide = np.zeros(num_samples, dtype=np.float32)

        for index in range(num_samples):
            if clock_pulses[index] > 0.5 or self._active_step < 0:
                self._advance_step()

            event = self._current_event
            frequency[index] = self._current_frequency
            if event.note is not None and event.gate:
                gate_limit = max(1, int(self.step_samples * event.gate_length))
                is_tied = event.tie or event.slide
                gate[index] = (
                    1.0 if is_tied or self._samples_in_step < gate_limit else 0.0
                )
            accent[index] = 1.0 if event.accent else 0.0
            slide[index] = 1.0 if event.slide else 0.0
            self._samples_in_step += 1

        return SequencerFrame(
            frequency=frequency,
            gate=gate,
            accent=accent,
            slide=slide,
        )

    def configure_clock(self, *, bpm: float, division: str, swing: float) -> None:
        """Update clock parameters while preserving phase."""
        self.clock.bpm = float(bpm)
        self.clock.division = division
        self.clock.swing = float(swing)

    def _advance_step(self) -> None:
        if not self.pattern:
            self.pattern = [StepEvent(None, gate=False)]

        self._active_step = (self._active_step + 1) % len(self.pattern)
        self._current_event = self.pattern[self._active_step]
        self._samples_in_step = 0

        if self._current_event.note is not None:
            note = int(np.clip(self._current_event.note + self.transpose, 0, 127))
            self._current_frequency = midi_note_to_pitch_cv(note)

    @staticmethod
    def _fit_signal(values: np.ndarray, num_samples: int) -> np.ndarray:
        signal = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(signal) == num_samples:
            return signal
        if len(signal) > num_samples:
            return signal[:num_samples]
        padded = np.zeros(num_samples, dtype=np.float32)
        padded[: len(signal)] = signal
        return padded
