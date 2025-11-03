from abc import abstractmethod, ABC

import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.engine_component_registry import ComponentCategory


class AudioComponent(ABC):
    """Base class for all audio components."""

    # Common metadata
    component_name: str
    component_category: ComponentCategory


class SignalGenerator(AudioComponent):
    """Base for components that generate signals (oscillators, modulators, noise)."""

    def __init__(self, sample_rate: float = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate


class SignalModifier(AudioComponent):
    """Base for components that modify signals (effects, filters)."""

    @abstractmethod
    def __call__(self, val: float | tuple) -> float | tuple:
        """Apply modification to a value."""
        pass


class SignalComposer(AudioComponent):
    """Base for components that combine signals (chain, mixer)."""

    def __init__(self, *components: AudioComponent):
        self.components = components

