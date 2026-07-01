"""Control signal generators and modulators.

This package provides envelope generators and other modulation sources
for controlling audio synthesis parameters over time.
"""

from src.engine.dsp.modulators.base import Modulator
from src.engine.dsp.modulators.envelopes import (
    ADSREnvelope,
    DecayEnvelope,
    GateTriggeredADSR,
    getadsr,
)

__all__ = [
    "Modulator",
    "ADSREnvelope",
    "DecayEnvelope",
    "GateTriggeredADSR",
    "getadsr",
]
