"""Backward-compatible oscillator exports and synth helper.
The original monolithic oscillator implementation has been split into focused
modules:
- oscillator_base.py: shared oscillator infrastructure and amplitude helpers
- oscillator_ramp.py: sawtooth and triangle oscillators
- oscillator_sine.py: sine oscillator modes
- oscillator_square.py: square oscillator plus strategy implementations


Waveform oscillators for audio synthesis.

This module provides a comprehensive set of oscillator classes for generating
basic waveforms (sine, square, sawtooth, triangle). All oscillators support
both iterator-based and vectorized sample generation, with runtime parameter
modification capabilities.

Classes:
    Oscillator: Abstract base class for all oscillators.
    SineOscillator: Generates pure sine waves.
    SquareOscillator: Generates square waves.
    SawtoothOscillator: Generates sawtooth waves.
    TriangleOscillator: Generates triangle waves.

Amplitude Control:
    Oscillators support both linear amplitude and decibel (dB) gain control:

    - **amplitude** (linear): Direct multiplier (0.0 to 1.0+)
      Example: amplitude=0.5 means output is halved

    - **gain_db** (decibels): Professional audio standard
      Example: gain_db=-6 means -6 dB reduction (≈half amplitude)

    - **wave_range**: Advanced feature for non-standard output ranges
      Default is (-1, 1) for audio. Rarely needed except for:
        * Control signals (e.g., 0 to 1 for LFO)
        * Legacy algorithm compatibility
        * Scientific applications

    How they work together:
    1. Waveform is generated in wave_range (default: -1 to 1)
    2. Converted to standard range if wave_range != (-1, 1)
    3. Multiplied by amplitude (determined from gain_db or amplitude parameter)

    Priority: gain_db > amplitude if both specified
    Default: gain_db=-20.0 (safe for mixing multiple sources)

Performance:
    - Iterator mode: Flexible but slower, suitable for small buffers
    - Vectorized mode: 50-85x faster, suitable for production use
    - Auto mode: Automatically select the best method based on buffer size

Note:
    All oscillators maintain phase continuity when switching between
    iterator and vectorized modes, enabling seamless parameter changes
    during audio generation.
"""

from src.engine.oscillator_base import (
    DEFAULT_TIME_AMPLITUDE_SMOOTHING_MS,
    Oscillator,
    _derive_amplitude_from_init,
)
from src.engine.oscillator_ramp import SawtoothOscillator, TriangleOscillator
from src.engine.oscillator_sine import SineOscillator
from src.engine.oscillator_square import (
    IdealSquareStrategy,
    IdealSquareStrategySmoothing,
    SoftSquareStrategy,
    SquareOscillator,
    SquareWaveFactory,
    SquareWaveMode,
    SquareWaveStrategy,
)

__all__ = [
    "DEFAULT_TIME_AMPLITUDE_SMOOTHING_MS",
    "Oscillator",
    "SineOscillator",
    "SawtoothOscillator",
    "TriangleOscillator",
    "SquareOscillator",
    "SquareWaveMode",
    "SquareWaveStrategy",
    "IdealSquareStrategy",
    "IdealSquareStrategySmoothing",
    "SoftSquareStrategy",
    "SquareWaveFactory",
    "_derive_amplitude_from_init",
]

