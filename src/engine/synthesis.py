"""Convenience synthesis helpers built on top of engine oscillators.

This module contains small, user-facing utilities that compose the lower-level
oscillator classes into simple one-shot operations.
"""

from __future__ import annotations

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.oscillator_ramp import SawtoothOscillator, TriangleOscillator
from src.engine.oscillator_sine import SineOscillator
from src.engine.oscillator_square import SquareOscillator


def synth(
    frequency: float = 440,
    dur: float = 1.0,
    amplitude: float = 1.0,
    sr: float | int = DEFAULT_SAMPLE_RATE,
    stype: str = "sine",
    mode: str = "auto",
) -> np.ndarray:
    """Synthesize a waveform of the requested type.

    Args:
        frequency: Oscillator frequency in Hz.
        dur: Duration in seconds.
        amplitude: Linear output amplitude.
        sr: Sample rate in Hz.
        stype: Waveform type. Supports sine/sin, square, saw/sawtooth, tri/triangle.
        mode: Sample-generation mode passed through to the oscillator.

    Returns:
        A NumPy array containing the synthesized waveform.
    """
    if dur <= 0:
        raise ValueError("Duration must be positive.")
    if frequency < 0:
        raise ValueError("Frequency must be non-negative.")

    n_samples = int(dur * sr)
    normalized_type = stype.lower()
    if normalized_type == "sin":
        normalized_type = "sine"
    if normalized_type == "sawtooth":
        normalized_type = "saw"
    if normalized_type == "triangle":
        normalized_type = "tri"

    synth_map = {
        "sine": SineOscillator,
        "square": SquareOscillator,
        "saw": SawtoothOscillator,
        "tri": TriangleOscillator,
    }

    try:
        osc = synth_map[normalized_type](
            frequency=frequency,
            amplitude=amplitude,
            sample_rate=sr,
        )
    except KeyError as exc:
        raise ValueError(f"Unsupported waveform type: {stype}") from exc

    return osc.get_samples(n_samples, mode=mode)


__all__ = ["synth"]
