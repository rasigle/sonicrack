"""Component registry system for PatchBuilder.

This module provides a registry-based architecture for managing audio synthesis
components, enabling the PatchBuilder to be automatically extended without
code modification when new components are added.

The registry system separates component metadata from the builder logic,
making the system highly maintainable and extensible.

Classes:
    ComponentDescriptor: Metadata describing a component type
    ComponentCategory: Enum for component categories
    ComponentRegistry: Central registry for all components

Example:
    >>> from src.builder import registry, ComponentDescriptor, PatchBuilder
    >>>
    >>> # Register a new oscillator
    >>> registry.register(ComponentDescriptor(
    ...     name="custom_oscillator",
    ...     category=ComponentCategory.OSCILLATOR,
    ...     factory=CustomOscillator,
    ...     config_params=["frequency", "amplitude"],
    ...     description="A custom oscillator"
    ... ))
    >>>
    >>> # PatchBuilder can now use it automatically
    >>> patch = PatchBuilder().custom_oscillator(440).build()
"""

from __future__ import annotations
from typing import Any, Callable, Type
from dataclasses import dataclass
from enum import Enum

from src.utils.logging_config import get_logger

logger = get_logger("builder.component_registry")


class ComponentCategory(Enum):
    """Categories of audio synthesis components."""

    OSCILLATOR = "oscillator"
    MODULATOR = "modulator"
    MODIFIER = "modifier"
    COMPOSER = "composer"


@dataclass
class ComponentDescriptor:
    """Metadata describing a component type.

    Attributes:
        name: Unique identifier for the component (e.g., "sine_oscillator")
        category: Component category (oscillator, modulator, modifier, composer)
        factory: Class or factory function to create instances
        config_params: List of parameter names to store in preset configs
        description: Human-readable description
        method_name: Optional method name for PatchBuilder (defaults to name without
            category suffix)
        builder_handler: Optional custom handler for adding to PatchBuilder
        serializer: Optional custom serializer function
        deserializer: Optional custom deserializer function
    """

    name: str
    category: ComponentCategory
    factory: Type | Callable
    config_params: list[str]
    description: str = ""
    method_name: str | None = None
    builder_handler: Callable | None = None
    serializer: Callable | None = None
    deserializer: Callable | None = None

    def __post_init__(self):
        """Set defaults for optional fields."""
        if self.method_name is None:
            # Remove category suffix if present (e.g., "sine_oscillator" -> "sine")
            base_name = self.name
            for cat in ComponentCategory:
                suffix = f"_{cat.value}"
                if base_name.endswith(suffix):
                    base_name = base_name[: -len(suffix)]
                    break
            self.method_name = base_name

    def create_instance(self, *args, **kwargs) -> Any:
        """Create an instance of this component.

        Args:
            *args: Positional arguments for factory
            **kwargs: Keyword arguments for factory

        Returns:
            Component instance
        """
        return self.factory(*args, **kwargs)

    def to_config(self, *args, **kwargs) -> dict[str, Any]:
        """Convert component parameters to config dictionary.

        Args:
            *args: Positional arguments (matched to config_params in order)
            **kwargs: Keyword arguments

        Returns:
            Configuration dictionary
        """
        config = {"type": self.name}

        # Add positional args mapped to config_params
        for i, (param, value) in enumerate(zip(self.config_params, args)):
            config[param] = value

        # Add keyword args if they're in config_params
        for param in self.config_params[len(args) :]:
            if param in kwargs:
                config[param] = kwargs[param]

        return config

    def from_config(self, config: dict[str, Any]) -> Any:
        """Create instance from configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Component instance
        """
        if self.deserializer:
            return self.deserializer(config)

        # Extract parameters from config
        kwargs = {
            param: config[param] for param in self.config_params if param in config
        }
        return self.create_instance(**kwargs)


class ComponentRegistry:
    """Central registry for all audio synthesis components.

    This registry maintains metadata about all available components and
    provides utilities for registration, lookup, serialization, and
    deserialization.

    Example:
        >>> from src.builder.component_registry import registry
        >>>
        >>> # Register a component
        >>> registry.register(descriptor)
        >>>
        >>> # Look up a component
        >>> desc = registry.get("sine_oscillator")
        >>>
        >>> # Create instance from config
        >>> config = {"type": "sine_oscillator", "frequency": 440}
        >>> instance = registry.create_from_config(config)
    """

    def __init__(self):
        """Initialize empty registry."""
        self._components: dict[str, ComponentDescriptor] = {}
        self._categories: dict[ComponentCategory, list[str]] = {
            cat: [] for cat in ComponentCategory
        }

    def register(self, descriptor: ComponentDescriptor) -> None:
        """Register a component descriptor.

        Args:
            descriptor: Component descriptor to register

        Raises:
            ValueError: If component name already registered
        """
        if descriptor.name in self._components:
            logger.warning(
                f"Component '{descriptor.name}' already registered, overwriting"
            )

        self._components[descriptor.name] = descriptor

        # Add to category index
        if descriptor.name not in self._categories[descriptor.category]:
            self._categories[descriptor.category].append(descriptor.name)

        logger.debug(
            f"Registered component: {descriptor.name} ({descriptor.category.value})"
        )

    def get(self, name: str) -> ComponentDescriptor | None:
        """Get component descriptor by name.

        Args:
            name: Component name

        Returns:
            ComponentDescriptor or None if not found
        """
        return self._components.get(name)

    def get_by_category(self, category: ComponentCategory) -> list[ComponentDescriptor]:
        """Get all components in a category.

        Args:
            category: Component category

        Returns:
            List of component descriptors
        """
        return [self._components[name] for name in self._categories[category]]

    def list_components(self) -> list[str]:
        """List all registered component names.

        Returns:
            List of component names
        """
        return list(self._components.keys())

    def create_from_config(self, config: dict[str, Any], **extra_kwargs) -> Any:
        """Create component instance from configuration.

        Args:
            config: Configuration dictionary with 'type' key
            **extra_kwargs: Additional keyword arguments (e.g., sample_rate)

        Returns:
            Component instance

        Raises:
            ValueError: If component type not registered
        """
        comp_type = config.get("type")
        if not comp_type:
            raise ValueError("Config must contain 'type' field")

        descriptor = self.get(comp_type)
        if not descriptor:
            raise ValueError(f"Unknown component type: {comp_type}")

        # Merge extra kwargs
        merged_config = {**config, **extra_kwargs}

        return descriptor.from_config(merged_config)

    def to_config(self, name: str, *args, **kwargs) -> dict[str, Any]:
        """Convert component parameters to config.

        Args:
            name: Component name
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Configuration dictionary

        Raises:
            ValueError: If component not registered
        """
        descriptor = self.get(name)
        if not descriptor:
            raise ValueError(f"Unknown component: {name}")

        return descriptor.to_config(*args, **kwargs)

    def clear(self) -> None:
        """Clear all registered components."""
        self._components.clear()
        for cat in self._categories:
            self._categories[cat].clear()
        logger.info("Cleared component registry")


# Global singleton registry
registry = ComponentRegistry()


def register_component(
    name: str,
    category: str | ComponentCategory,
    factory: Type | Callable,
    config_params: list[str],
    description: str = "",
    **kwargs,
) -> ComponentDescriptor:
    """Convenience function to register a component.

    Args:
        name: Component name
        category: Component category (string or enum)
        factory: Factory class or function
        config_params: List of config parameter names
        description: Description
        **kwargs: Additional descriptor options

    Returns:
        Created ComponentDescriptor
    """
    # Convert string category to enum
    if isinstance(category, str):
        category = ComponentCategory(category)

    descriptor = ComponentDescriptor(
        name=name,
        category=category,
        factory=factory,
        config_params=config_params,
        description=description,
        **kwargs,
    )

    registry.register(descriptor)
    return descriptor
