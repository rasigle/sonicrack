"""Audio synthesis engine for AudioPlayground.

This package provides a flexible, extensible audio synthesis engine with:
- Oscillators: Sine, Square, Sawtooth, Triangle waveforms
- Modulators: ADSR envelopes and other time-varying control signals
- Composers: Chain and WaveAdder for combining signals
- Modifiers: Panning, Volume, Clipping effects
- Filters: Butterworth filter design and application
- Noise generators: White, Pink, Brown, Blue, Perlin noise

Example:
    >>> from src.engine import SineOscillator, ADSREnvelope, ModulatedOscillator
    >>> osc = SineOscillator(freq=440, amp=1.0)
    >>> env = ADSREnvelope(attack_duration=0.1, decay_duration=0.2,
    ...                    sustain_level=0.7, release_duration=0.3)
    >>> mod_osc = ModulatedOscillator(osc, env, amp_mod=lambda a, e: a * e)
    >>> samples = mod_osc.get_samples(44100)
"""

# Oscillators
from src.engine.oscillator import (
    Oscillator,
    SineOscillator,
    SquareOscillator,
    SawtoothOscillator,
    TriangleOscillator,
    synth,
)

# Modulators
from src.engine.modulator import (
    Modulator,
    ADSREnvelope,
    getadsr,
)

# Modulated Oscillator
from src.engine.modulated_oscillator import ModulatedOscillator

# Composers
from src.engine.composer import (
    Composer,
    Chain,
    WaveAdder,
)

# Modifiers
from src.engine.modifier import (
    Modifier,
    Panner,
    ModulatedPanner,
    Volume,
    ModulatedVolume,
    Frequency,
    ModulatedFrequency,
    Clipper,
)

# Filters
from src.engine.filter import (
    butter,
    apply_filter,
)

# Noise generators
from src.engine.noise import (
    white_noise,
    pink_noise,
    brownian_noise,
    blue_noise,
    perlin_noise,
)

__all__ = [
    # Oscillators
    "Oscillator",
    "SineOscillator",
    "SquareOscillator",
    "SawtoothOscillator",
    "TriangleOscillator",
    "synth",

    # Modulators
    "Modulator",
    "ADSREnvelope",
    "getadsr",
    # Modulated Oscillator
    "ModulatedOscillator",
    # Composers
    "Composer",
    "Chain",
    "WaveAdder",
    # Modifiers
    "Modifier",
    "Panner",
    "ModulatedPanner",
    "Volume",
    "ModulatedVolume",
    "Frequency",
    "ModulatedFrequency",
    "Clipper",
    # Filters
    "butter",
    "apply_filter",
    # Noise
    "white_noise",
    "pink_noise",
    "brownian_noise",
    "blue_noise",
    "perlin_noise",
]

__version__ = "0.1.0"
