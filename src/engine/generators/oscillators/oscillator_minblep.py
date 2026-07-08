"""Shared minBLEP utilities for bandlimited oscillator edges.

Provides common functions used by both VCVRackSquareStrategy and
SawtoothOscillator (vcv mode) to avoid code duplication.
"""

import math

import numpy as np

TWO_PI = 2 * np.pi
VCV_MINBLEP_ZERO_CROSSINGS = 16
VCV_MINBLEP_OVERSAMPLE = 16


def crossing_subsample(
    threshold: float, start_phase: float, end_phase: float
) -> float | None:
    """Calculate subsample position of a phase crossing.

    Used to interpolate minBLEP corrections between samples for
    antialiased discontinuities.

    Args:
        threshold: Phase threshold to detect crossing (e.g., 0.0 or 0.5)
        start_phase: Phase at previous sample
        end_phase: Phase at current sample

    Returns:
        Subsample position (0.0 to 1.0) if crossing detected, None otherwise
    """
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


def insert_minblep_discontinuity(
    buffer: np.ndarray,
    minblep_table: np.ndarray,
    subsample: float,
    magnitude: float,
    extended_table: np.ndarray | None = None,
) -> None:
    """Insert a minBLEP discontinuity correction into a buffer.

    Args:
        buffer: Target buffer to add the correction to
        minblep_table: MinBLEP lookup table
        subsample: Subsample position (0.0 to 1.0) of the discontinuity
        magnitude: Magnitude of the discontinuity (positive or negative)
        extended_table: Extended table to add the correction to
            (optional, defaults to minblep_table with an extra zero)
    """
    if not 0.0 < subsample <= 1.0 or magnitude == 0.0:
        return
    if extended_table is None:
        extended_table = np.concatenate((minblep_table, np.zeros(1, dtype=np.float32)))
    offset = (1.0 - subsample) * VCV_MINBLEP_OVERSAMPLE
    for index in range(len(buffer)):
        position = index * VCV_MINBLEP_OVERSAMPLE + offset
        lower = int(position)
        fraction = position - lower
        value = extended_table[lower] + fraction * (
            extended_table[lower + 1] - extended_table[lower]
        )
        buffer[index] += magnitude * value


def shift_minblep_buffer(buffer: np.ndarray) -> float:
    """Shift a minBLEP buffer left by one sample and return the oldest value.

    Args:
        buffer: Buffer to shift (modified in-place)

    Returns:
        The value that was shifted out (oldest sample)
    """
    if len(buffer) == 0:
        return 0.0
    value = float(buffer[0])
    buffer[:-1] = buffer[1:]
    buffer[-1] = 0.0
    return value


def compute_dc_alpha(sample_rate: float, cutoff_hz: float = 20.0) -> float:
    """Compute DC blocking filter alpha coefficient.

    Args:
        sample_rate: Sample rate in Hz
        cutoff_hz: Filter cutoff frequency in Hz (default: 20.0)

    Returns:
        Alpha coefficient for the DC blocking filter
    """
    cutoff = min(0.4, cutoff_hz / sample_rate)
    w = TWO_PI * cutoff
    return w / (1.0 + w)


def process_dc_filter(
    value: float, state: float, alpha: float, enabled: bool = True
) -> tuple[float, float]:
    """Apply DC blocking filter to a single sample.

    Args:
        value: Input sample value
        state: Current filter state
        alpha: Filter coefficient (from compute_dc_alpha)
        enabled: Whether filtering is enabled

    Returns:
        Tuple of (filtered_value, new_state)
    """
    if not enabled:
        return value, state
    new_state = state + alpha * (value - state)
    return value - new_state, new_state


def _blackman_harris(position: np.ndarray) -> np.ndarray:
    return (
        0.35875
        - 0.48829 * np.cos(TWO_PI * position)
        + 0.14128 * np.cos(2 * TWO_PI * position)
        - 0.01168 * np.cos(3 * TWO_PI * position)
    )


def minimum_phase_minblep_table(
    zero_crossings: int = VCV_MINBLEP_ZERO_CROSSINGS,
    oversample: int = VCV_MINBLEP_OVERSAMPLE,
) -> np.ndarray:
    n = 2 * zero_crossings * oversample
    positions = np.arange(n, dtype=np.float64) / oversample - zero_crossings
    impulse = np.sinc(positions)
    impulse *= _blackman_harris(np.arange(n, dtype=np.float64) / (n - 1))

    spectrum = np.fft.fft(impulse)
    log_magnitude = np.log(np.maximum(np.abs(spectrum), np.exp(-10.0)))
    cepstrum = np.fft.ifft(log_magnitude)
    cepstrum[1 : n // 2] *= 2.0
    cepstrum[n // 2 :] = 0.0
    minimum_phase = np.fft.ifft(np.exp(np.fft.fft(cepstrum))).real

    step = np.cumsum(minimum_phase) / oversample
    step = np.concatenate(([0.0], step[:-1]))
    step /= step[-1] + minimum_phase[-1] / oversample
    return np.asarray(step - 1.0, dtype=np.float32)
