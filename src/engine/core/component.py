from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import numpy as np

from src.constants import DEFAULT_GAIN_DB, DEFAULT_SAMPLE_RATE
from src.engine.utils.validation import validate_sample_count, validate_sample_rate

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
class ParameterDescriptor:
    """Metadata describing a component parameter."""

    name: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None
    clamp: bool = False
    choices: tuple[Any, ...] | None = None
    description: str = ""


COMMON_PARAMETER_DESCRIPTORS: dict[str, ParameterDescriptor] = {
    "frequency": ParameterDescriptor(
        name="frequency",
        default=440.0,
        minimum=0.0,
        unit="Hz",
        description="Oscillator frequency.",
    ),
    "amplitude": ParameterDescriptor(
        name="amplitude",
        default=1.0,
        minimum=0.0,
        description="Linear gain multiplier.",
    ),
    "gain_db": ParameterDescriptor(
        name="gain_db",
        default=DEFAULT_GAIN_DB,
        unit="dB",
        description="Gain in decibels.",
    ),
    "phase": ParameterDescriptor(
        name="phase",
        default=0.0,
        unit="deg",
        description="Initial phase offset.",
    ),
    "sample_rate": ParameterDescriptor(
        name="sample_rate",
        default=DEFAULT_SAMPLE_RATE,
        minimum=1.0,
        unit="Hz",
        description="Processing sample rate.",
    ),
    "wave_range": ParameterDescriptor(
        name="wave_range",
        default=(-1, 1),
        description="Output range as minimum and maximum values.",
    ),
}


def make_parameter_descriptors(
    *names: str, **overrides: ParameterDescriptor
) -> dict[str, ParameterDescriptor]:
    """Build ordered parameter metadata from common descriptors and overrides."""
    parameters: dict[str, ParameterDescriptor] = {}
    for name in names:
        if name in overrides:
            parameters[name] = overrides[name]
        elif name in COMMON_PARAMETER_DESCRIPTORS:
            parameters[name] = COMMON_PARAMETER_DESCRIPTORS[name]
        else:
            raise KeyError(f"No ParameterDescriptor registered for {name!r}")
    return parameters


@dataclass(frozen=True)
class ComponentDescriptor:
    """Metadata describing a component type.

    Attributes:
        name: Unique identifier for the component (e.g., "sine_oscillator")
        category: Component category (oscillator, modulator, modifier, composer)
        parameters: Ordered runtime parameter metadata for config/UI policy.
        description: More detailed and human-readable description of the component.
        fluent_api_name: Optional name of the component for the fluent API
        serializer: Optional custom serializer function
        deserializer: Optional custom deserializer function
    """

    name: str
    category: ComponentCategory
    description: str = ""
    parameters: dict[str, ParameterDescriptor] | None = None
    tags: list[str] | None = None
    fluent_api_name: str | None = None
    serializer: Callable | None = None
    deserializer: Callable | None = None

    @property
    def parameter_names(self) -> list[str]:
        """Return serializable parameter names in constructor/config order."""
        return list(self.parameters or {})

    def to_config(self, *args, **kwargs) -> dict[str, Any]:
        """Convert the given component configuration parameters to dictionary.

        Args:
            *args: Positional arguments (matched to parameters in order)
            **kwargs: Keyword arguments

        Example:
            >>> descriptor = ComponentDescriptor(
            ...     name="sine_oscillator",
            ...     category=ComponentCategory.OSCILLATOR,
            ...     parameters={
            ...         "frequency": ParameterDescriptor("frequency", 440.0),
            ...         "amplitude": ParameterDescriptor("amplitude", 1.0),
            ...     }
            ... )
            >>> conf = descriptor.to_config(frequency=440, amplitude=0.5)
            >>> print(conf)
            {'name': 'sine_oscillator', 'frequency': 440, 'amplitude': 0.5}

        Returns:
            Configuration dictionary
        """
        config = {"name": self.name, "category": self.category.value}
        if self.description:
            config["description"] = self.description

        parameter_names = self.parameter_names
        if parameter_names:
            if args:
                # Add positional args mapped to parameters.
                for param, value in zip(parameter_names, args, strict=False):
                    config[param] = value

            if kwargs:
                # Add keyword args if they're declared parameters.
                for param in parameter_names[len(args) :]:
                    if param in kwargs:
                        config[param] = kwargs[param]

        return config


class AudioComponent(ABC):
    """Base class for all audio components."""

    # Common metadata
    component_name: str
    descriptor: ComponentDescriptor
    _provided_args: set[str]
    ended: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Permit dynamic construction through registry-returned base types."""
        _ = args, kwargs

    def __iter__(self) -> AudioComponent:
        return self

    def __next__(self) -> float:
        raise StopIteration

    def get_samples_vectorized(self, n: int) -> np.ndarray:
        validate_sample_count(n)
        raise NotImplementedError

    def get_samples(self, n: int, *args: Any, **kwargs: Any) -> np.ndarray:
        _ = args, kwargs
        n = validate_sample_count(n)
        return self.get_samples_vectorized(n)

    @classmethod
    def create_instance(cls: type[T], *args: Any, **kwargs: Any) -> T:
        return cls(*args, **kwargs)

    @classmethod
    def from_config(cls: type[T], config: dict[str, Any]) -> T:
        """Creates an audio component instance from given configuration dictionary.

        Example:
            >>> from src.engine import SineOscillator
            >>> conf = {"frequency": 440, "amplitude": 0.5}
            >>> sine = SineOscillator.from_config(conf)

        Returns:
            Component instance
        """
        descr = cls.descriptor
        parameter_names = descr.parameter_names

        # Extract parameters from config
        kwargs = {param: config[param] for param in config if param in parameter_names}
        return cls(**kwargs)

    def get_component_name(self) -> str:
        """Resolve a human-readable component name for logs and diagnostics."""
        instance_name = getattr(self, "component_name", None)
        if isinstance(instance_name, str) and instance_name:
            return instance_name

        descriptor = getattr(type(self), "descriptor", None)
        descriptor_name = getattr(descriptor, "name", None)
        if isinstance(descriptor_name, str) and descriptor_name:
            return descriptor_name

        return self.__class__.__name__

    def __str__(self) -> str:
        return f"AudioComponent {self.get_component_name()}"


class Generator(AudioComponent):
    """Base for components that generate signals (oscillators, modulators, noise)."""

    def __init__(self, sample_rate: float = DEFAULT_SAMPLE_RATE):
        super().__init__()

        self.sample_rate = validate_sample_rate(sample_rate)
