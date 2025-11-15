from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any, TypeVar, Type

from src.constants import DEFAULT_SAMPLE_RATE

T = TypeVar("T", bound="AudioComponent")


class ComponentCategory(Enum):
    """Categories of audio synthesis components."""

    OSCILLATOR = "oscillator"
    MODULATOR = "modulator"
    MODIFIER = "modifier"
    COMPOSER = "composer"
    FILTER = "filter"
    EFFECT = "effect"


@dataclass(frozen=True)
class ComponentDescriptor:
    """Metadata describing a component type.

    Attributes:
        name: Unique identifier for the component (e.g., "sine_oscillator")
        category: Component category (oscillator, modulator, modifier, composer)
        config_params: List of parameter names to store in preset configs
        description: Human-readable description
        fluent_api_name: Optional name of the component for the fluent API
        serializer: Optional custom serializer function
        deserializer: Optional custom deserializer function
    """

    name: str
    category: ComponentCategory
    description: str = ""
    config_params: list[str] | None = None
    tags: list[str] | None = None
    fluent_api_name: str | None = None
    serializer: Callable | None = None
    deserializer: Callable | None = None

    def to_config(self, *args, **kwargs) -> dict[str, Any]:
        """Convert the given component configuration parameters to dictionary.

        Args:
            *args: Positional arguments (matched to config_params in order)
            **kwargs: Keyword arguments

        Example:
            >>> descriptor = ComponentDescriptor(
            ...     name="sine_oscillator",
            ...     category=ComponentCategory.OSCILLATOR,
            ...     config_params=["frequency", "amplitude"]
            ... )
            >>>     >>> conf = descriptor.to_config(frequency=440, amplitude=0.5)
            >>> print(conf)
            {'name': 'sine_oscillator', 'frequency': 440, 'amplitude': 0.5}

        Returns:
            Configuration dictionary
        """
        config = {"name": self.name, "category": self.category.value}
        if self.description:
            config["description"] = self.description

        if self.config_params:
            if args:
                # Add positional args mapped to config_params
                for i, (param, value) in enumerate(zip(self.config_params, args)):
                    config[param] = value

            if kwargs:
                # Add keyword args if they're in config_params
                for param in self.config_params[len(args) :]:
                    if param in kwargs:
                        config[param] = kwargs[param]

        return config


class AudioComponent(ABC):
    """Base class for all audio components."""

    # Common metadata
    component_name: str
    descriptor: ComponentDescriptor

    @classmethod
    def from_config(cls: Type[T], config: dict[str, Any]) -> T:
        """Creates an audio component instance from given configuration dictionary.

        Example:
            >>> from engine import SineOscillator
            >>> conf = {"frequency": 440, "amplitude": 0.5}
            >>> sine = SineOscillator.from_config(conf)

        Returns:
            Component instance
        """
        descr = cls.descriptor

        # Extract parameters from config
        kwargs = {
            param: config[param] for param in config if param in descr.config_params
        }
        return cls(**kwargs)


class Generator(AudioComponent):
    """Base for components that generate signals (oscillators, modulators, noise)."""

    def __init__(self, sample_rate: float = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate
