"""Behringer/Roland System 100M 182-style analog sequencer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.component import (
    AudioComponent,
    ComponentCategory,
    ComponentDescriptor,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.core.registry import register_component
from src.engine.sequencing.clock import StepClock
from src.engine.utils.validation import validate_sample_count, validate_sample_rate


@dataclass(slots=True)
class Behringer182Frame:
    """Rendered outputs for one 182 sequencer audio buffer."""

    cv_a: np.ndarray
    cv_b: np.ndarray
    gate: np.ndarray
    trigger: np.ndarray
    end: np.ndarray
    step: np.ndarray


@register_component()
class Behringer182Sequencer(AudioComponent):
    """Eight-step dual-row CV/gate sequencer inspired by the 182 module."""

    descriptor = ComponentDescriptor(
        name="Behringer182Sequencer",
        category=ComponentCategory.SEQUENCER,
        description="System 100M 182-style dual CV row analog step sequencer",
        tags=["sequencer", "cv", "gate", "behringer", "182", "system-100m"],
        parameters=make_parameter_descriptors(
            "bpm",
            "division",
            "swing",
            "steps",
            "direction",
            "gate_length",
            "cv_a_range",
            "cv_b_range",
            "sample_rate",
            bpm=ParameterDescriptor(
                name="bpm",
                default=120.0,
                minimum=1.0,
                unit="BPM",
                description="Internal clock tempo.",
            ),
            division=ParameterDescriptor(
                name="division",
                default="1/16",
                choices=("1/4", "1/8", "1/16", "1/32"),
                description="Internal clock step division.",
            ),
            swing=ParameterDescriptor(
                name="swing",
                default=0.0,
                minimum=0.0,
                maximum=0.75,
                description="Internal clock swing amount.",
            ),
            steps=ParameterDescriptor(
                name="steps",
                default=8,
                minimum=1,
                maximum=8,
                description="Active sequence length.",
            ),
            direction=ParameterDescriptor(
                name="direction",
                default="forward",
                choices=("forward", "reverse", "pendulum", "random"),
                description="Step advance direction.",
            ),
            gate_length=ParameterDescriptor(
                name="gate_length",
                default=0.5,
                minimum=0.0,
                maximum=1.0,
                description="Fraction of each step where the gate is high.",
            ),
            cv_a_range=ParameterDescriptor(
                name="cv_a_range",
                default=5.0,
                minimum=0.0,
                unit="V",
                description="Output voltage range for CV row A.",
            ),
            cv_b_range=ParameterDescriptor(
                name="cv_b_range",
                default=5.0,
                minimum=0.0,
                unit="V",
                description="Output voltage range for CV row B.",
            ),
        ),
        fluent_api_name="behringer_182",
    )

    def __init__(
        self,
        cv_a: list[float] | np.ndarray | None = None,
        cv_b: list[float] | np.ndarray | None = None,
        gates: list[bool] | np.ndarray | None = None,
        *,
        bpm: float = 120.0,
        division: str = "1/16",
        swing: float = 0.0,
        steps: int = 8,
        direction: str = "forward",
        gate_length: float = 0.5,
        cv_a_range: float = 5.0,
        cv_b_range: float = 5.0,
        sample_rate: float = DEFAULT_SAMPLE_RATE,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.sample_rate = validate_sample_rate(sample_rate)
        self.cv_a = self._fit_steps(cv_a, default=0.0)
        self.cv_b = self._fit_steps(cv_b, default=0.0)
        self.gates = self._fit_gates(gates)
        self.steps = int(np.clip(steps, 1, 8))
        self.direction = direction
        self.gate_length = float(gate_length)
        self.cv_a_range = float(cv_a_range)
        self.cv_b_range = float(cv_b_range)
        self.clock = StepClock(
            bpm=bpm,
            division=division,
            swing=swing,
            sample_rate=self.sample_rate,
        )
        self._rng = np.random.default_rng(random_seed)
        self._active_step = -1
        self._samples_in_step = 0
        self._pendulum_delta = 1
        self._previous_reset = 0.0

    def reset(self) -> None:
        """Reset clock and sequence position."""
        self.clock.reset()
        self._active_step = -1
        self._samples_in_step = 0
        self._pendulum_delta = 1
        self._previous_reset = 0.0

    @property
    def step_samples(self) -> int:
        return self.clock.step_samples

    def configure_clock(self, *, bpm: float, division: str, swing: float) -> None:
        """Update internal clock parameters while preserving phase."""
        self.clock.bpm = float(bpm)
        self.clock.division = division
        self.clock.swing = float(swing)

    def process(
        self,
        num_samples: int,
        clock_pulses: np.ndarray | None = None,
        reset_pulses: np.ndarray | None = None,
        hold_signal: np.ndarray | None = None,
        run_signal: np.ndarray | None = None,
    ) -> Behringer182Frame:
        """Render CV, gate, trigger, end, and active-step lanes."""
        num_samples = validate_sample_count(num_samples)
        if clock_pulses is None:
            clock_pulses = self.clock.process(num_samples)
        else:
            clock_pulses = self._fit_signal(clock_pulses, num_samples)

        if (
            reset_pulses is None
            and hold_signal is None
            and run_signal is None
        ):
            return self._process_clocked(num_samples, clock_pulses)

        resets = self._fit_signal(reset_pulses, num_samples)
        holds = self._fit_signal(hold_signal, num_samples)
        runs = (
            np.ones(num_samples, dtype=np.float32)
            if run_signal is None
            else self._fit_signal(run_signal, num_samples)
        )

        cv_a = np.empty(num_samples, dtype=np.float32)
        cv_b = np.empty(num_samples, dtype=np.float32)
        gate = np.zeros(num_samples, dtype=np.float32)
        trigger = np.zeros(num_samples, dtype=np.float32)
        end = np.zeros(num_samples, dtype=np.float32)
        step = np.empty(num_samples, dtype=np.float32)

        for index in range(num_samples):
            reset_edge = self._previous_reset <= 0.5 and resets[index] > 0.5
            self._previous_reset = float(resets[index])
            can_advance = holds[index] <= 0.5 and runs[index] > 0.5
            should_clock = clock_pulses[index] > 0.5 or self._active_step < 0
            if reset_edge:
                self._set_step(0)
                trigger[index] = 1.0 if self.gates[self._active_step] else 0.0
                end[index] = 1.0 if self._active_step == self.steps - 1 else 0.0
            elif should_clock and can_advance:
                wrapped = self._advance_step()
                trigger[index] = 1.0 if self.gates[self._active_step] else 0.0
                end[index] = 1.0 if wrapped else 0.0

            active = max(0, self._active_step)
            cv_a[index] = self._scale_cv(self.cv_a[active], self.cv_a_range)
            cv_b[index] = self._scale_cv(self.cv_b[active], self.cv_b_range)
            if self.gates[active]:
                gate_limit = max(1, int(self.step_samples * self.gate_length))
                gate[index] = 1.0 if self._samples_in_step < gate_limit else 0.0
            step[index] = float(active)
            self._samples_in_step += 1

        return Behringer182Frame(
            cv_a=cv_a,
            cv_b=cv_b,
            gate=gate,
            trigger=trigger,
            end=end,
            step=step,
        )

    def _process_clocked(
        self, num_samples: int, clock_pulses: np.ndarray
    ) -> Behringer182Frame:
        """Fast path for the common clock-only render case."""
        cv_a = np.empty(num_samples, dtype=np.float32)
        cv_b = np.empty(num_samples, dtype=np.float32)
        gate = np.zeros(num_samples, dtype=np.float32)
        trigger = np.zeros(num_samples, dtype=np.float32)
        end = np.zeros(num_samples, dtype=np.float32)
        step = np.empty(num_samples, dtype=np.float32)

        event_indices = np.flatnonzero(clock_pulses > 0.5)
        if self._active_step < 0 and (
            len(event_indices) == 0 or event_indices[0] != 0
        ):
            event_indices = np.insert(event_indices, 0, 0)

        if len(event_indices) == 0:
            self._fill_segment(
                0,
                num_samples,
                cv_a=cv_a,
                cv_b=cv_b,
                gate=gate,
                step=step,
            )
            self._samples_in_step += num_samples
            return Behringer182Frame(
                cv_a=cv_a,
                cv_b=cv_b,
                gate=gate,
                trigger=trigger,
                end=end,
                step=step,
            )

        segment_starts = event_indices.astype(np.int64, copy=False)
        segment_ends = np.empty_like(segment_starts)
        segment_ends[:-1] = segment_starts[1:]
        segment_ends[-1] = num_samples

        if segment_starts[0] > 0:
            self._fill_segment(
                0,
                int(segment_starts[0]),
                cv_a=cv_a,
                cv_b=cv_b,
                gate=gate,
                step=step,
            )
            self._samples_in_step += int(segment_starts[0])

        for start, stop in zip(segment_starts, segment_ends, strict=False):
            start = int(start)
            stop = int(stop)
            wrapped = self._advance_step()
            active = max(0, self._active_step)
            if self.gates[active]:
                trigger[start] = 1.0
            if wrapped:
                end[start] = 1.0
            self._fill_segment(
                start,
                stop,
                cv_a=cv_a,
                cv_b=cv_b,
                gate=gate,
                step=step,
            )
            self._samples_in_step += stop - start

        return Behringer182Frame(
            cv_a=cv_a,
            cv_b=cv_b,
            gate=gate,
            trigger=trigger,
            end=end,
            step=step,
        )

    def _fill_segment(
        self,
        start: int,
        stop: int,
        *,
        cv_a: np.ndarray,
        cv_b: np.ndarray,
        gate: np.ndarray,
        step: np.ndarray,
    ) -> None:
        if stop <= start:
            return

        active = max(0, self._active_step)
        cv_a[start:stop] = self._scale_cv(self.cv_a[active], self.cv_a_range)
        cv_b[start:stop] = self._scale_cv(self.cv_b[active], self.cv_b_range)
        step[start:stop] = float(active)

        if not self.gates[active]:
            return

        gate_limit = max(1, int(self.step_samples * self.gate_length))
        length = stop - start
        offsets = self._samples_in_step + np.arange(length)
        gate[start:stop] = (offsets < gate_limit).astype(np.float32)

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        """Return CV row A for compatibility with generator-style use."""
        return self.process(n).cv_a

    def _advance_step(self) -> bool:
        if self.direction == "reverse":
            next_step = (
                self.steps - 1 if self._active_step < 0 else self._active_step - 1
            )
            wrapped = next_step < 0
            self._set_step(self.steps - 1 if wrapped else next_step)
            return wrapped

        if self.direction == "pendulum":
            if self._active_step < 0:
                self._set_step(0)
                return False
            next_step = self._active_step + self._pendulum_delta
            wrapped = next_step >= self.steps or next_step < 0
            if wrapped:
                self._pendulum_delta *= -1
                next_step = self._active_step + self._pendulum_delta
            self._set_step(int(np.clip(next_step, 0, self.steps - 1)))
            return wrapped

        if self.direction == "random":
            previous = self._active_step
            self._set_step(int(self._rng.integers(0, self.steps)))
            return previous == self.steps - 1 and self._active_step != previous

        next_step = self._active_step + 1
        wrapped = next_step >= self.steps
        self._set_step(0 if wrapped else next_step)
        return wrapped

    def _set_step(self, step: int) -> None:
        self._active_step = int(np.clip(step, 0, self.steps - 1))
        self._samples_in_step = 0

    @staticmethod
    def _fit_steps(
        values: list[float] | np.ndarray | None, default: float
    ) -> np.ndarray:
        if values is None:
            values = [default] * 8
        raw = np.asarray(values, dtype=np.float32).reshape(-1)
        fitted = np.full(8, default, dtype=np.float32)
        fitted[: min(8, len(raw))] = raw[:8]
        return fitted

    @staticmethod
    def _fit_gates(values: list[bool] | np.ndarray | None) -> np.ndarray:
        if values is None:
            values = [True] * 8
        raw = np.asarray(values, dtype=bool).reshape(-1)
        fitted = np.ones(8, dtype=bool)
        fitted[: min(8, len(raw))] = raw[:8]
        return fitted

    @staticmethod
    def _fit_signal(
        values: np.ndarray | None, num_samples: int, default: float = 0.0
    ) -> np.ndarray:
        if values is None:
            return np.full(num_samples, default, dtype=np.float32)
        signal = np.asarray(values, dtype=np.float32).reshape(-1)
        if len(signal) == num_samples:
            return signal
        if len(signal) > num_samples:
            return signal[:num_samples]
        padded = np.full(num_samples, default, dtype=np.float32)
        padded[: len(signal)] = signal
        return padded

    @staticmethod
    def _scale_cv(value: float, output_range: float) -> float:
        return float(np.clip(value, 0.0, 1.0) * max(0.0, output_range))
