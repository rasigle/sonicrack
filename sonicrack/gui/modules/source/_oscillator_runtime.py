"""Shared runtime helpers for GUI oscillator source modules."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from scipy.signal import lfilter

from sonicrack.runtime.helpers import gate_transition_indices


class RuntimeOscillator(Protocol):
    """Minimal oscillator API needed for runtime frequency ramping."""

    frequency: float

    def get_samples(self, n: int) -> np.ndarray: ...

    def __next__(self) -> float: ...


# Increased from 35ms to prevent clicks/pops when changing frequency
# Smoothing happens in log (pitch) space for natural-sounding transitions
DEFAULT_FREQUENCY_SLEW_TIME_MS = 100.0


def smooth_control_signal(
    signal: np.ndarray,
    previous_value: float | None,
    sample_rate: float,
    smoothing_time_ms: float,
) -> tuple[np.ndarray, float | None]:
    """One-pole smooth a control signal while preserving buffer state.

    Uses a vectorized IIR (``scipy.signal.lfilter``) equivalent to the former
    per-sample Python loop:

        y[n] = y[n-1] + (x[n] - y[n-1]) * alpha
    """
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    if len(values) == 0:
        return values, previous_value
    if smoothing_time_ms <= 0.0:
        return values, float(values[-1])

    smoothing_samples = max(sample_rate * smoothing_time_ms / 1000.0, 1.0)
    alpha = 1.0 - math.exp(-1.0 / smoothing_samples)
    prev = float(values[0] if previous_value is None else previous_value)

    # y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, alpha - 1.0], dtype=np.float64)
    zi = np.array([(1.0 - alpha) * prev], dtype=np.float64)
    smoothed, _ = lfilter(b, a, values.astype(np.float64, copy=False), zi=zi)
    result = smoothed.astype(np.float32)
    return result, float(result[-1])


def _frequency_slew_values(
    previous_frequency: float,
    target_frequency: float,
    num_samples: int,
    sample_rate: float,
    slew_time_ms: float,
) -> np.ndarray:
    """Build a smooth per-sample frequency curve in pitch space.

    Closed-form one-pole trajectory (no per-sample Python loop):

        y[n] = target + (y0 - target) * (1 - alpha) ** (n + 1)
    """
    if num_samples <= 0:
        return np.empty(0, dtype=np.float64)
    if previous_frequency == target_frequency or slew_time_ms <= 0.0:
        return np.full(num_samples, target_frequency, dtype=np.float64)

    slew_samples = max(sample_rate * slew_time_ms / 1000.0, 1.0)
    alpha = 1.0 - math.exp(-1.0 / slew_samples)
    decay = 1.0 - alpha
    # exponents 1..N match the iterative loop's post-update values
    powers = decay ** np.arange(1, num_samples + 1, dtype=np.float64)

    if previous_frequency > 0.0 and target_frequency > 0.0:
        log_prev = math.log(previous_frequency)
        log_target = math.log(target_frequency)
        log_freqs = log_target + (log_prev - log_target) * powers
        frequencies = np.exp(log_freqs)
    else:
        frequencies = (
            target_frequency + (previous_frequency - target_frequency) * powers
        )

    if abs(frequencies[-1] - target_frequency) < max(
        1e-6, abs(target_frequency) * 1e-6
    ):
        frequencies[-1] = target_frequency

    return frequencies


def _render_frequency_buffer(
    oscillator: RuntimeOscillator,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Render samples for a per-sample frequency curve.

    Uses each oscillator's ``__next__`` path so ramp continuity matches the
    click-free tests across buffer splits. (``process_frequency_buffer`` uses a
    different internal cycle state model that is not continuous for all
    waveforms when ramps are split across callbacks.)

    The frequency trajectory itself is precomputed vectorially; only the
    sample loop remains scalar.
    """
    n = len(frequencies)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    samples = np.empty(n, dtype=np.float32)
    osc_next = oscillator.__next__
    for index, frequency in enumerate(frequencies):
        oscillator.frequency = float(frequency)
        samples[index] = osc_next()
    return samples


def render_with_frequency_ramp(
    oscillator: RuntimeOscillator,
    previous_frequency: float,
    target_frequency: float,
    num_samples: int,
    slew_time_ms: float = DEFAULT_FREQUENCY_SLEW_TIME_MS,
) -> tuple[np.ndarray, float]:
    """Render one buffer while slewing frequency smoothly across samples.

    Stable-frequency buffers use vectorized ``get_samples``. Changing frequency
    uses a vectorized slew curve plus a per-sample render that preserves
    oscillator phase continuity across buffer boundaries.
    """
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
    samples = _render_frequency_buffer(oscillator, frequencies)
    final_frequency = float(frequencies[-1]) if len(frequencies) else target_frequency
    return samples, final_frequency


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

    clock = np.asarray(clock_signal[:num_samples], dtype=np.float32).reshape(-1)
    note_ons, _note_offs, final_clock = gate_transition_indices(
        clock,
        previous_clock,
        low=0.3,
        high=0.7,
    )

    if note_ons.size == 0:
        samples, rendered_frequency = render_with_frequency_ramp(
            oscillator,
            previous_frequency,
            target_frequency,
            num_samples,
            frequency_slew_time_ms,
        )
        return samples, rendered_frequency, final_clock

    chunks: list[np.ndarray] = []
    start = 0
    rendered_frequency = previous_frequency
    previous_output = last_output_value
    pending_reset_previous_output = None
    sample_rate = float(getattr(oscillator, "sample_rate", 44100.0))

    for index in note_ons:
        index = int(index)
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
            chunks.append(chunk)
            if len(chunk) > 0:
                previous_output = float(chunk[-1])

        reset = getattr(oscillator, "_initialize_osc", None)
        if callable(reset):
            reset()
        pending_reset_previous_output = previous_output
        start = index

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
        chunks.append(chunk)

    if not chunks:
        rendered = np.empty(0, dtype=np.float32)
    else:
        rendered = np.concatenate(chunks).astype(np.float32, copy=False)

    return rendered, rendered_frequency, final_clock
