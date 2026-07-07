from __future__ import annotations

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.core.component import AudioComponent
from src.engine.utils.validation import validate_sample_rate


class Generator(AudioComponent):
    """Base for components that generate signals (oscillators, modulators, noise)."""

    def __init__(self, sample_rate: float = DEFAULT_SAMPLE_RATE):
        super().__init__()

        self.sample_rate = validate_sample_rate(sample_rate)
