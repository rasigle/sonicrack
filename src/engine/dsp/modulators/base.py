"""Base modulator class."""

from __future__ import annotations

import logging

from src.engine.generators.oscillators.oscillator_base import Generator

logger = logging.getLogger(__name__)


class Modulator(Generator):
    """Base class for all modulators.
    Modulators generate time-varying control signals used to modulate
    parameters of other audio components (e.g., amplitude, frequency).
    """

    # This is a base class - actual modulators will implement their own descriptors
    pass
