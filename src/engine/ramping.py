"""Shared ramp and smoothing helpers for engine and realtime IO paths."""

from __future__ import annotations

from numbers import Real

import numpy as np

from src.engine.validation import validate_sample_rate


def duration_ms_to_samples(
    sample_rate: float,
    duration_ms: float,
    *,
    name: str = "duration_ms",
    min_samples: int = 0,
) -> int:
    """Convert a finite non-negative duration in milliseconds to samples."""
    sample_rate = validate_sample_rate(sample_rate)
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, Real):
        raise TypeError(
            f"{name} must be a real number, got {type(duration_ms).__name__}"
        )

    duration_ms = float(duration_ms)
    if not np.isfinite(duration_ms) or duration_ms < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")

    return max(min_samples, int(duration_ms * sample_rate / 1000))


def fill_linear_ramp(
    start: float,
    stop: float,
    length: int,
    *,
    out: np.ndarray | None = None,
    index_buffer: np.ndarray | None = None,
) -> np.ndarray:
    """Return a float32 linear ramp, optionally reusing caller-owned buffers."""
    if length < 0:
        raise ValueError("length must be non-negative")

    if out is None:
        ramp = np.empty(length, dtype=np.float32)
    else:
        if out.shape[0] < length:
            raise ValueError("out buffer is shorter than length")
        ramp = out[:length]

    if length == 0:
        return ramp
    if length == 1:
        ramp[0] = stop
        return ramp

    if index_buffer is None:
        indices = np.arange(length, dtype=np.float32)
    else:
        if index_buffer.shape[0] < length:
            raise ValueError("index buffer is shorter than length")
        indices = index_buffer[:length]

    scale = (stop - start) / (length - 1)
    np.multiply(indices, scale, out=ramp)
    ramp += start
    return ramp


def consume_linear_ramp(
    current: float,
    target: float,
    remaining_samples: int,
    count: int,
) -> tuple[np.ndarray, float, int]:
    """Consume a chunk from a linear ramp and return envelope, current, remaining."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.empty(0, dtype=np.float32), current, remaining_samples
    if remaining_samples <= 0:
        return np.full(count, target, dtype=np.float32), target, 0

    ramp_count = min(count, remaining_samples)
    positions = np.arange(1, ramp_count + 1, dtype=np.float64)
    envelope = np.asarray(
        current + (target - current) * (positions / remaining_samples),
        dtype=np.float32,
    )

    new_remaining = remaining_samples - ramp_count
    new_current = target if new_remaining <= 0 else float(envelope[-1])

    if ramp_count == count:
        return envelope, new_current, new_remaining

    tail = np.full(count - ramp_count, target, dtype=np.float32)
    return np.concatenate((envelope, tail)), new_current, 0
