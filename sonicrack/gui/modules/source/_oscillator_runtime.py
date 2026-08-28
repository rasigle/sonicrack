"""Shared runtime helpers for GUI oscillator source modules.

Most helpers live in ``soniclab.generators.oscillators.runtime``. This module
re-exports them and provides a realtime-optimized ``render_with_frequency_ramp``
that prefers each oscillator's bulk modulated-waveform path.

The stock soniclab ramp falls back to a per-sample ``__next__`` loop while the
frequency is slewing. On band-limited square/saw that is ~100x slower than the
vectorized path and can underrun the audio callback (audible glitches/crackles)
when the user moves oscillator frequency knobs in dense patches such as
``demo_wobble_bass``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from soniclab.generators.oscillators.runtime import (
    DEFAULT_FREQUENCY_SLEW_TIME_MS,
    RuntimeOscillator,
    _frequency_slew_values,
    _render_frequency_buffer,
    render_with_clock_resets,
    smooth_continuity_correction,
    smooth_control_signal,
)

__all__ = [
    "DEFAULT_FREQUENCY_SLEW_TIME_MS",
    "RuntimeOscillator",
    "_frequency_slew_values",
    "_render_frequency_buffer",
    "render_with_clock_resets",
    "render_with_frequency_ramp",
    "smooth_continuity_correction",
    "smooth_control_signal",
]


def _frequencies_effectively_equal(previous: float, target: float) -> bool:
    """Return True when the pitch error is negligible for audio purposes.

    Exact float equality never holds for long with the one-pole slew (it only
    snaps at ~1e-6 relative after >1s). Without a practical threshold the
    expensive ramp path keeps running after the knob has settled.
    """
    if previous == target:
        return True
    return abs(previous - target) <= max(1e-4, abs(target) * 1e-6)


def _render_bulk_frequency_buffer(
    oscillator: Any, frequencies: np.ndarray
) -> np.ndarray | None:
    """Render a frequency curve via bulk modulated-waveform API when present."""
    render_modulated = getattr(oscillator, "render_modulated_waveform", None)
    commit_state = getattr(oscillator, "commit_modulated_phase_state", None)
    if not callable(render_modulated) or not callable(commit_state):
        return None

    waveform, state = render_modulated(frequencies)
    commit_state(state)
    samples = np.asarray(waveform, dtype=np.float32)
    apply_amplitude = getattr(oscillator, "_apply_amplitude_to_buffer", None)
    if callable(apply_amplitude):
        samples = np.asarray(apply_amplitude(samples), dtype=np.float32)
    return samples.astype(np.float32, copy=False)


def render_with_frequency_ramp(
    oscillator: RuntimeOscillator,
    previous_frequency: float,
    target_frequency: float,
    num_samples: int,
    slew_time_ms: float = DEFAULT_FREQUENCY_SLEW_TIME_MS,
) -> tuple[np.ndarray, float]:
    """Render one buffer while slewing frequency smoothly across samples.

    Stable-frequency buffers use vectorized ``get_samples``. Changing frequency
    prefers the oscillator bulk ``render_modulated_waveform`` path (sample-
    accurate, phase-continuous, realtime-safe) and only falls back to the
    per-sample ``__next__`` loop when that API is unavailable.
    """
    target = float(target_frequency)
    previous = float(previous_frequency)

    if num_samples <= 0:
        return np.empty(0, dtype=np.float32), target

    if _frequencies_effectively_equal(previous, target):
        oscillator.frequency = target
        return oscillator.get_samples(num_samples), target

    sample_rate = float(getattr(oscillator, "sample_rate", 44100.0))
    frequencies = _frequency_slew_values(
        previous,
        target,
        num_samples,
        sample_rate,
        slew_time_ms,
    )

    samples = _render_bulk_frequency_buffer(oscillator, frequencies)
    if samples is None:
        samples = _render_frequency_buffer(oscillator, frequencies)

    final_frequency = float(frequencies[-1]) if len(frequencies) else target
    if _frequencies_effectively_equal(final_frequency, target):
        final_frequency = target
    oscillator.frequency = final_frequency
    return samples, final_frequency
