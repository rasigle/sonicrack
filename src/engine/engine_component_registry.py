"""Standardized interface for audio engine components.

This module defines the abstract interfaces that all audio components must implement,
enabling automatic registration, introspection, and integration with the builder system.

The interface system provides:
- Component metadata (name, category, parameters)
- Auto-registration capability
- Parameter validation and introspection
- Serialization/deserialization support
- Type safety and documentation
"""

from __future__ import annotations
from typing import Any, Type, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import inspect

from src.utils.logging_config import get_logger

logger = get_logger("engine.component_registry")


class ComponentCategory(Enum):
    """Categories of audio synthesis components."""

    OSCILLATOR = "oscillator"
    MODULATOR = "modulator"
    MODIFIER = "modifier"
    COMPOSER = "composer"
    FILTER = "filter"
    EFFECT = "effect"


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
    """Central registry for all audio components with auto-discovery.

    This registry maintains all registered components and provides
    introspection capabilities. It automatically loads all engine
    components on first access.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._components: dict[str, ComponentDescriptor] = {}
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization - load all engine components on first access."""
        if self._initialized:
            return

        self._initialized = True

        # Import all engine modules to trigger decorator registration
        try:
            from src.engine import oscillator
            from src.engine import modulator
            from src.engine import modifier
            from src.engine import composer
            from src.engine import modulated_oscillator
            logger.debug("Auto-loaded engine components into registry")
        except Exception as e:
            logger.error(f"Failed to auto-load engine components: {e}")

    def register(self, descriptor: ComponentDescriptor, override: bool = False) -> None:
        """Register a component descriptor.

        Args:
            descriptor: ComponentDescriptor to register
            override: Allow overriding existing registration

        Raises:
            ValueError: If component already registered and override=False
        """
        if descriptor.name in self._components and not override:
            logger.warning(f"Component {descriptor.name} already registered, skipping")
            return

        self._components[descriptor.name] = descriptor
        logger.debug(f"Registered component: {descriptor.name} ({descriptor.category.value})")

    def get(self, name: str) -> Optional[ComponentDescriptor]:
        """Get component descriptor by name.

        Args:
            name: Component name

        Returns:
            ComponentDescriptor or None if not found
        """
        self._ensure_initialized()
        return self._components.get(name)

    def list_by_category(self, category: ComponentCategory) -> list[str]:
        """List all components in a category.

        Args:
            category: Component category

        Returns:
            List of component names
        """
        self._ensure_initialized()
        return [
            name for name, desc in self._components.items()
            if desc.category == category
        ]

    def list_all(self) -> list[str]:
        """List all registered component names.

        Returns:
            List of all component names
        """
        self._ensure_initialized()
        return list(self._components.keys())

    def create_instance(self, name: str, *args, **kwargs) -> Any:
        """Create component instance by name.

        Args:
            name: Component name
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Component instance

        Raises:
            ValueError: If component not registered
        """
        self._ensure_initialized()
        descriptor = self.get(name)
        if not descriptor:
            raise ValueError(f"Component not registered: {name}")

        return descriptor.create_instance(*args, **kwargs)

    def create_from_config(self, config: dict[str, Any], **extra_kwargs) -> Any:
        """Create component from configuration.

        Args:
            config: Configuration dictionary with 'type' key
            **extra_kwargs: Additional keyword arguments to merge (e.g., sample_rate)

        Returns:
            Component instance

        Raises:
            ValueError: If type not specified or not registered
        """
        self._ensure_initialized()
        component_type = config.get("type")
        if not component_type:
            raise ValueError("Config must contain 'type' field")

        descriptor = self.get(component_type)
        if not descriptor:
            raise ValueError(f"Component not registered: {component_type}")

        # Merge extra kwargs into config
        merged_config = {**config, **extra_kwargs}
        return descriptor.from_config(merged_config)


def register_component(
    category: ComponentCategory,
    description: str = "",
    tags: list[str] = None,
    **metadata_kwargs
) -> Callable:
    """Decorator for auto-registering component classes.

    This decorator automatically extracts metadata from the class and registers
    it in the global registry when the class is defined.

    Args:
        category: Component category
        description: Component description
        tags: Optional tags (not used currently, for future expansion)
        **metadata_kwargs: Additional metadata fields

    Returns:
        Decorator function

    Example:
        >>> @register_component(
        ...     category=ComponentCategory.OSCILLATOR,
        ...     description="Sine wave oscillator"
        ... )
        ... class SineOscillator:
        ...     def __init__(self, frequency: float = 440.0):
        ...         pass
    """
    def decorator(cls: Type) -> Type:
        # Generate component name from class name
        name = _class_name_to_component_name(cls.__name__)

        # Extract parameter names from __init__ signature
        config_params = _extract_parameter_names(cls)

        # Create and register descriptor
        descriptor = ComponentDescriptor(
            name=name,
            category=category,
            factory=cls,
            config_params=config_params,
            description=description or cls.__doc__ or "",
        )

        registry.register(descriptor)

        return cls

    return decorator


def _class_name_to_component_name(class_name: str) -> str:
    """Convert class name to component name.

    Args:
        class_name: Class name (e.g., "SineOscillator")

    Returns:
        Component name (e.g., "sine_oscillator")
    """
    import re
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', class_name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


def _extract_parameter_names(cls: Type) -> list[str]:
    """Extract parameter names from __init__ signature.

    Args:
        cls: Component class

    Returns:
        List of parameter names (excluding 'self', 'args', 'kwargs')
    """
    try:
        sig = inspect.signature(cls.__init__)
        return [
            param_name
            for param_name in sig.parameters.keys()
            if param_name not in ('self', 'args', 'kwargs')
        ]
    except Exception as e:
        logger.warning(f"Could not extract parameters from {cls.__name__}: {e}")
        return []


# Global singleton registry
registry = ComponentRegistry()
