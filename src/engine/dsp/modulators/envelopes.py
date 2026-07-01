"""Envelope generators for audio modulation.

This module provides a unified interface for all envelope generators.
The individual envelopes have been split into separate modules for better
organization and maintainability.

For direct imports, use:
- from src.engine.dsp.modulators.adsr_envelope import ADSREnvelope, ADSRPhase, getadsr
- from src.engine.dsp.modulators.decay_envelope import DecayEnvelope
- from src.engine.dsp.modulators.gate_triggered import GateTriggeredADSR
"""

from __future__ import annotations

from src.engine.dsp.modulators.adsr_envelope import ADSREnvelope, ADSRPhase, getadsr
from src.engine.dsp.modulators.decay_envelope import DecayEnvelope
from src.engine.dsp.modulators.gate_triggered import GateTriggeredADSR

# Re-export for backward compatibility
__all__ = [
    "ADSREnvelope",
    "ADSRPhase",
    "DecayEnvelope",
    "GateTriggeredADSR",
    "getadsr",
]
