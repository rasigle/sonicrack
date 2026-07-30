"""Shared runtime helpers for GUI oscillator source modules.

Implementation lives in ``soniclab.generators.oscillators.runtime``; this module
re-exports the public API for existing SonicRack import paths.
"""

from __future__ import annotations

from soniclab.generators.oscillators.runtime import (
    DEFAULT_FREQUENCY_SLEW_TIME_MS,
    RuntimeOscillator,
    _frequency_slew_values,
    _render_frequency_buffer,
    render_with_clock_resets,
    render_with_frequency_ramp,
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
