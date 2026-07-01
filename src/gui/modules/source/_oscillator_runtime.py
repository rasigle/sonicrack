"""Shared runtime helpers for GUI oscillator source modules."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class RuntimeOscillator(Protocol):
    """Minimal oscillator API needed for runtime frequency ramping."""

    frequency: float

    def get_samples(self, n: int) -> np.ndarray: ...

    def __next__(self) -> float: ...


DEFAULT_FREQUENCY_SLEW_TIME_MS = 35.0


def smooth_control_signal(
    signal: np.ndarray,
    previous_value: float | None,
    sample_rate: float,
    smoothing_time_ms: float,
) -> tuple[np.ndarray, float | None]:
    """One-pole smooth a control signal while preserving buffer state."""
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if len(values) == 0:
        return values, previous_value
    if smoothing_time_ms <= 0.0:
        return values, float(values[-1])

    smoothing_samples = max(sample_rate * smoothing_time_ms / 1000.0, 1.0)
    alpha = 1.0 - math.exp(-1.0 / smoothing_samples)
    current = float(values[0] if previous_value is None else previous_value)

    smoothed = np.empty(len(values), dtype=np.float32)
    for index, target in enumerate(values):
        current += (float(target) - current) * alpha
        smoothed[index] = current

    return smoothed, current


def _frequency_slew_values(
    previous_frequency: float,
    target_frequency: float,
    num_samples: int,
    sample_rate: float,
    slew_time_ms: float,
) -> np.ndarray:
    """Build a smooth per-sample frequency curve in pitch space."""
    if num_samples <= 0:
        return np.empty(0, dtype=np.float64)
    if previous_frequency == target_frequency or slew_time_ms <= 0.0:
        return np.full(num_samples, target_frequency, dtype=np.float64)

    slew_samples = max(sample_rate * slew_time_ms / 1000.0, 1.0)
    alpha = 1.0 - math.exp(-1.0 / slew_samples)
    frequencies = np.empty(num_samples, dtype=np.float64)

    if previous_frequency > 0.0 and target_frequency > 0.0:
        current = math.log(previous_frequency)
        target = math.log(target_frequency)
        for index in range(num_samples):
            current += (target - current) * alpha
            frequencies[index] = math.exp(current)
    else:
        current = previous_frequency
        for index in range(num_samples):
            current += (target_frequency - current) * alpha
            frequencies[index] = current

    if abs(frequencies[-1] - target_frequency) < max(
        1e-6, abs(target_frequency) * 1e-6
    ):
        frequencies[-1] = target_frequency

    return frequencies


def render_with_frequency_ramp(
    oscillator: RuntimeOscillator,
    previous_frequency: float,
    target_frequency: float,
    num_samples: int,
    slew_time_ms: float = DEFAULT_FREQUENCY_SLEW_TIME_MS,
) -> tuple[np.ndarray, float]:
    """Render one buffer while slewing frequency smoothly across samples."""
    sample_rate = float(getattr(oscillator, "sample_rate", 44100.0))
    if previous_frequency == target_frequency:
        oscillator.frequency = target_frequency
        return oscillator.get_samples(num_samples), target_frequency

    frequencies = _frequency_slew_values(
        previous_frequency,
        target_frequency,
        num_samples,
        sample_rate,
        slew_time_ms,
    )
    samples = []
    for frequency in frequencies:
        oscillator.frequency = float(frequency)
        samples.append(next(oscillator))
    final_frequency = float(frequencies[-1]) if len(frequencies) else target_frequency
    return np.asarray(samples, dtype=np.float32), final_frequency


def smooth_continuity_correction(
    samples: np.ndarray,
    previous_output: float | None,
    sample_rate: float,
    smoothing_time_ms: float,
) -> np.ndarray:
    """Remove an output step by decaying a correction over a short window."""
    if previous_output is None or len(samples) == 0:
        return samples

    smoothing_samples = int(round(sample_rate * smoothing_time_ms / 1000))
    smoothing_samples = min(max(smoothing_samples, 1), len(samples))
    correction = float(previous_output - samples[0])
    if correction == 0.0:
        return samples

    smoothed = samples.copy()
    envelope = np.linspace(1.0, 0.0, smoothing_samples, endpoint=False)
    smoothed[:smoothing_samples] += correction * envelope
    return smoothed.astype(np.float32)


def render_with_clock_resets(
    oscillator: RuntimeOscillator,
    previous_frequency: float,
    target_frequency: float,
    num_samples: int,
    clock_signal: np.ndarray | None,
    previous_clock: float,
    last_output_value: float | None,
    reset_smoothing_time_ms: float,
    frequency_slew_time_ms: float = DEFAULT_FREQUENCY_SLEW_TIME_MS,
) -> tuple[np.ndarray, float, float]:
    """Render with shared pitch slew and sample-accurate clock resets."""
    if clock_signal is None:
        samples, rendered_frequency = render_with_frequency_ramp(
            oscillator,
            previous_frequency,
            target_frequency,
            num_samples,
            frequency_slew_time_ms,
        )
        return samples, rendered_frequency, 0.0

    samples = []
    start = 0
    rendered_frequency = previous_frequency
    current_previous_clock = previous_clock
    previous_output = last_output_value
    pending_reset_previous_output = None
    sample_rate = float(getattr(oscillator, "sample_rate", 44100.0))

    for index, value in enumerate(clock_signal[:num_samples]):
        current_clock = float(value)
        should_reset = current_previous_clock < 0.3 and current_clock > 0.7
        if should_reset:
            if index > start:
                chunk, rendered_frequency = render_with_frequency_ramp(
                    oscillator,
                    rendered_frequency,
                    target_frequency,
                    index - start,
                    frequency_slew_time_ms,
                )
                if pending_reset_previous_output is not None:
                    chunk = smooth_continuity_correction(
                        chunk,
                        pending_reset_previous_output,
                        sample_rate,
                        reset_smoothing_time_ms,
                    )
                    pending_reset_previous_output = None
                samples.append(chunk)
                if len(chunk) > 0:
                    previous_output = float(chunk[-1])

            reset = getattr(oscillator, "_initialize_osc", None)
            if callable(reset):
                reset()
            pending_reset_previous_output = previous_output
            start = index
        current_previous_clock = current_clock

    if start < num_samples:
        chunk, rendered_frequency = render_with_frequency_ramp(
            oscillator,
            rendered_frequency,
            target_frequency,
            num_samples - start,
            frequency_slew_time_ms,
        )
        if pending_reset_previous_output is not None:
            chunk = smooth_continuity_correction(
                chunk,
                pending_reset_previous_output,
                sample_rate,
                reset_smoothing_time_ms,
            )
        samples.append(chunk)

    if not samples:
        rendered = np.empty(0, dtype=np.float32)
    else:
        rendered = np.concatenate(samples).astype(np.float32)

    final_clock = (
        float(clock_signal[min(num_samples, len(clock_signal)) - 1])
        if num_samples > 0
        else current_previous_clock
    )
    return rendered, rendered_frequency, final_clock
