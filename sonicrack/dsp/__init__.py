"""Lightweight DSP processors used by SonicRack GUI modules.

These fill gaps not yet exposed by the ``soniclab`` engine package (modulation
FX, dynamics finishing, wavetable sources). Implementations are NumPy-based
and realtime-oriented for typical buffer sizes.
"""

from sonicrack.dsp.modulation_fx import Chorus, Phaser
from sonicrack.dsp.dynamics_eq import Limiter, ParametricEQ
from sonicrack.dsp.wavetable import WavetableOscillator
from sonicrack.dsp.utilities import Attenuverter, SignalMult

__all__ = [
    "Attenuverter",
    "Chorus",
    "Limiter",
    "ParametricEQ",
    "Phaser",
    "SignalMult",
    "WavetableOscillator",
]
