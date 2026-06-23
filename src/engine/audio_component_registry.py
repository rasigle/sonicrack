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

import inspect
from collections.abc import Callable

from src.engine.audio_component import (
    AudioComponent,
    ComponentCategory,
    ComponentDescriptor,
)
from src.utils.logging_config import get_engine_logger

logger = get_engine_logger("audio_component_registry")


class AudioComponentRegistry:
    """Central registry for all audio components with auto-discovery.

    This registry maintains all registered components and provides
    introspection capabilities. It automatically loads all engine
    components on first access.
    """

    def __init__(self):
        """Initialize empty registry."""

        self._components: dict[str, type[AudioComponent]] = {}
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization - load all engine components on first access."""
        if self._initialized:
            return

        self._initialized = True

        # Import all engine modules to trigger decorator registration
        try:

            logger.debug("Auto-loaded engine components into registry")
        except Exception as e:
            logger.error(f"Failed to auto-load engine components: {e}")

    def register(
        self, component_class: type[AudioComponent], override: bool = False
    ) -> None:
        """Register a component descriptor.

        Args:
            component_class: The class of the audio component to register
            override: Allow overriding existing registration

        Raises:
            TypeError: If component_class is not a valid class or doesn't inherit from
                AudioComponent
            AttributeError: If component_class doesn't have a descriptor attribute
        """
        # Validate module class
        if not inspect.isclass(component_class):
            raise TypeError(
                f"component_class must be a class, got {type(component_class)}"
            )

        if not issubclass(component_class, AudioComponent):
            raise TypeError(
                f"component_class must inherit from AudioComponent, "
                f"got {component_class.__bases__}"
            )

        # Check for descriptor attribute
        if not hasattr(component_class, "descriptor"):
            raise AttributeError(
                f"Component class {component_class.__name__} must have a 'descriptor' "
                f"class attribute with ComponentDescriptor metadata"
            )

        descriptor = component_class.descriptor

        # Validate descriptor
        if not isinstance(descriptor, ComponentDescriptor):
            raise TypeError(
                f"descriptor attribute must be ComponentDescriptor, "
                f"got {type(descriptor)}"
            )

        # Check if already registered
        if descriptor.name in self._components and not override:
            existing_component = self._components[descriptor.name]
            logger.warning(
                f"Skipping component '{descriptor.name}' provided from "
                f"'{component_class.__name__}', because it was already registered from "
                f"class '{existing_component.__name__}'. Use override=True to replace."
            )
            return

        # Register the component
        self._components[descriptor.name] = component_class
        logger.debug(
            f"✓ Registered component: {descriptor.name} "
            f"({descriptor.category.value}) - {component_class.__name__}"
        )

    @property
    def components(self) -> dict[str, type[AudioComponent]]:
        """Get all registered components.

        Returns:
            Dictionary of component name to component class
        """
        self._ensure_initialized()
        return self._components

    def get(self, name: str, strict: bool = False) -> type[AudioComponent] | None:
        """Get component class by name.

        Args:
            name: Component name
            strict: If True, raise ValueError when component not found.
                   If False (default), return None when not found.

        Returns:
            Component class or None if not found (when strict=False)

        Raises:
            ValueError: If component not registered and strict=True

        Example:
            >>> from engine import audio_registry
            >>>
            >>> # Safe lookup (returns None if not found)
            >>> sine_component = audio_registry.get("Sine")
            >>> if component:
            ...     instance = component(frequency=440)
            >>>
            >>> # Strict lookup (raises exception if not found)
            >>> sine_component = audio_registry.get("Sine", strict=True)  # Must exist
            >>> instance = component(frequency=440)
        """
        self._ensure_initialized()
        component = self._components.get(name)

        if component is None and strict:
            available = ", ".join(sorted(self._components.keys())[:10])
            if len(self._components) > 10:
                available += ", ..."
            raise ValueError(
                f"Component '{name}' not registered. "
                f"Available components: {available}"
            )

        return component

    def list_components(self) -> list[str]:
        """List all registered component names.

        Returns:
            List of all component names
        """
        self._ensure_initialized()
        return list(self._components.keys())

    def list_by_category(self, category: ComponentCategory) -> list[str]:
        """List all components of a specific category.

        Args:
            category: Component category

        Returns:
            List of component names
        """
        self._ensure_initialized()
        return [
            name
            for name, component in self._components.items()
            if component.descriptor.category == category
        ]

    def count(self) -> int:
        """Get count of registered components.

        Returns:
            Number of registered components
        """
        self._ensure_initialized()
        return len(self._components)

    def debug_info(self) -> str:
        """Get debug information about registry state.

        Returns:
            Formatted string with registry statistics
        """
        self._ensure_initialized()
        info = [
            "=== Audio Component Registry ===",
            f"Total components: {len(self._components)}",
            f"Initialized: {self._initialized}",
            "",
        ]

        # Group by category
        by_category: dict[str, list[str]] = {}
        for name, component in self._components.items():
            category = component.descriptor.category.value
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(name)

        for category, names in sorted(by_category.items()):
            info.append(f"{category.upper()}: {len(names)}")
            for name in sorted(names):
                info.append(f"  - {name}")

        return "\n".join(info)

    def create_instance(self, name: str, *args, **kwargs) -> AudioComponent:
        """Create component instance by name.

        Args:
            name: Component name
            *args: Positional arguments for component __init__
            **kwargs: Keyword arguments for component __init__

        Returns:
            Component instance

        Raises:
            ValueError: If component not registered

        Example:
            >>> # Create a sine oscillator
            >>> sine = audio_registry.create_instance("Sine", frequency=440)
        """
        self._ensure_initialized()
        component = self.get(name, strict=True)  # Will raise if not found
        assert component is not None
        return component(*args, **kwargs)


def _class_name_to_component_name(class_name: str) -> str:
    """Convert class name to component name.

    Args:
        class_name: Class name (e.g., "SineOscillator")

    Returns:
        Component name (e.g., "sine_oscillator")
    """
    import re

    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
    name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def _extract_parameter_names(cls: type) -> list[str]:
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
            for param_name in sig.parameters
            if param_name not in ("self", "args", "kwargs")
        ]
    except Exception as e:
        logger.warning(f"Could not extract parameters from {cls.__name__}: {e}")
        return []


def register_component(override: bool = False) -> Callable:
    """Decorator for auto-registering component classes.

    This decorator automatically registers the component class in the global
    registry when the class is defined. The component must have a `descriptor`
    class attribute with ComponentDescriptor metadata.

    Args:
        override: Allow overriding existing registration

    Returns:
        Decorator function

    Example:
        >>> @register_component()
        ... class SineOscillator(AudioComponent):
        ...     descriptor = ComponentDescriptor(
        ...         name="Sine",
        ...         category=ComponentCategory.OSCILLATOR,
        ...         description="Sine wave oscillator"
        ...     )
        ...     def __init__(self, frequency: float = 440.0):
        ...         pass
    """

    def decorator(cls: type[AudioComponent]) -> type[AudioComponent]:
        # Validate that the class has a descriptor
        if not hasattr(cls, "descriptor"):
            logger.error(
                f"Cannot register {cls.__name__}: missing 'descriptor' class attribute"
            )
            return cls

        # Register the component
        audio_registry.register(component_class=cls, override=override)
        return cls

    return decorator


# Global singleton registry for audio components
audio_registry = AudioComponentRegistry()
