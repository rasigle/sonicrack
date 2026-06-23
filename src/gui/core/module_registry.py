"""Dynamic module registration system for the modular synth.

This module provides decorators and utilities for automatic module registration,
enabling a plugin-like architecture where modules can self-register without
modifying the central registry.

New modules should use the @register_module decorator in their own files.
"""

import importlib
import importlib.util
import inspect
import logging
from collections.abc import Callable
from pathlib import Path

from src.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """Central registry for all available modules.

    Modules can be registered using the @register_module decorator,
    or programmatically via register() method.

    Example:
        ```python
        @register_module()
        class MyModule(ModuleWidget):
            ...
        ```
    """

    def __init__(self):
        """Initialize the module registry."""
        self._modules: dict[str, type[ModuleWidget]] = {}
        self._categories: dict[str, list[str]] = {}
        self._metadata: dict[str, dict] = {}

    def register(
        self, module_class: type[ModuleWidget], **override_metadata
    ) -> type[ModuleWidget]:
        """Register a module class.

        Args:
            module_class: The module class to register
            **override_metadata: Optional metadata to override extracted values

        Returns:
            The registered module class (for decorator usage)

        Raises:
            TypeError: If module_class doesn't inherit from ModuleWidget
        """
        # Validate module class
        if not inspect.isclass(module_class):
            raise TypeError(f"module_class must be a class, got {type(module_class)}")

        if not issubclass(module_class, ModuleWidget):
            raise TypeError(
                f"module_class must inherit from ModuleWidget, "
                f"got {module_class.__bases__}"
            )

        # Extract metadata from class properties using property descriptors
        # This avoids instantiation
        metadata = module_class.metadata
        name = metadata.title
        category = metadata.category
        description = metadata.description
        version = metadata.version
        author = metadata.author

        # Check for duplicate names
        if name in self._modules:
            logger.warning(
                f"Module '{name}' already registered, overwriting with "
                f"{module_class.__name__}"
            )

        # Register the module
        self._modules[name] = module_class

        # Add to category
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

        # Store metadata
        self._metadata[name] = {
            "class": module_class.__name__,
            "category": category,
            "description": description,
            "author": author,
            "version": version,
            **{
                k: v
                for k, v in override_metadata.items()
                if k not in ["name", "category", "description", "author", "version"]
            },
        }

        logger.debug(f"Registered module: {name} ({module_class.__name__})")

        return module_class

    def unregister(self, name: str) -> bool:
        """Unregister a module.

        Args:
            name: Name of the module to unregister

        Returns:
            True if module was unregistered, False if not found
        """
        if name not in self._modules:
            return False

        # Remove from modules
        del self._modules[name]

        # Remove from categories
        for category_modules in self._categories.values():
            if name in category_modules:
                category_modules.remove(name)

        # Remove metadata
        if name in self._metadata:
            del self._metadata[name]

        logger.debug(f"Unregistered module: {name}")
        return True

    def get(self, name: str, strict: bool = False) -> type[ModuleWidget] | None:
        """Get a module class by name.

        Args:
            name: Name of the module
            strict: If True, raise ValueError when module not found.
                   If False (default), return None when not found.

        Returns:
            The module class, or None if not found

        Raises:
            ValueError: If component not registered and strict=True
        """
        module = self._modules.get(name)
        if module is None and strict:
            available = ", ".join(sorted(self._modules.keys())[:10])
            raise ValueError(
                f"Component '{name}' not registered.\nAvailable components: {available}"
            )

        return module

    def get_all(self) -> dict[str, type[ModuleWidget]]:
        """Get all registered modules.

        Returns:
            Dictionary mapping module names to classes
        """
        return self._modules.copy()

    def get_by_category(self, category: str) -> dict[str, type[ModuleWidget]]:
        """Get all modules in a category.

        Args:
            category: Category name

        Returns:
            Dictionary mapping module names to classes in the category
        """
        module_names = self._categories.get(category, [])
        return {name: self._modules[name] for name in module_names}

    def get_categories(self) -> list[str]:
        """Get list of all categories.

        Returns:
            List of category names
        """
        return list(self._categories.keys())

    def get_metadata(self, name: str) -> dict | None:
        """Get metadata for a module.

        Args:
            name: Name of the module

        Returns:
            Metadata dictionary, or None if not found
        """
        return self._metadata.get(name)

    def list_modules(self) -> list[str]:
        """Get list of all registered module names.

        Returns:
            List of module names
        """
        return list(self._modules.keys())

    def count(self) -> int:
        """Get number of registered modules.

        Returns:
            Number of modules
        """
        return len(self._modules)

    def clear(self):
        """Clear all registered modules."""
        self._modules.clear()
        self._categories.clear()
        self._metadata.clear()
        logger.info("Cleared all registered modules")


def get_registry() -> ModuleRegistry:
    """Get the global module registry.

    Returns:
        The global ModuleRegistry instance
    """
    return _global_registry


def register_module(**override_metadata) -> Callable:
    """Decorator to register a module class.

    This decorator automatically registers the module when the class is defined.

    Args:
        **override_metadata: Optional metadata to override extracted values
                           (name, category, description, author, version, etc.)

    Returns:
        Decorator function

    Example:
        ```python
        # Simple usage - metadata from class properties
        @register_module()
        class MyOscillator(ModuleWidget):

        # With overrides
        @register_module(author="Your Name")
        class MyOscillator(ModuleWidget):
            ...
        ```
    """

    def decorator(cls: type[ModuleWidget]) -> type[ModuleWidget]:
        _global_registry.register(module_class=cls, **override_metadata)
        return cls

    return decorator


def discover_modules(
    package_path: str = "src.gui.modules", recursive: bool = False
) -> int:
    """Discover and import all module files in a package.

    This automatically imports all Python files in the modules package,
    which triggers the @register_module decorators.

    Args:
        package_path: Dot-separated package path to scan
        recursive: If True, search subdirectories recursively

    Returns:
        Number of modules discovered

    Example:
        ```python
        # At application startup:
        discover_modules("src.gui.modules")
        ```
    """
    try:
        # Import the package
        package = importlib.import_module(package_path)
        package_file = package.__file__
        if package_file is None:
            raise ValueError(f"Package '{package_path}' has no filesystem path")
        package_dir = Path(package_file).parent

        # Find all Python files
        module_files = (
            package_dir.rglob("*.py") if recursive else package_dir.glob("*.py")
        )

        count = 0
        for module_file in module_files:

            # Skip files and directories that start with '_' (private / __init__.py)
            rel_path = module_file.relative_to(package_dir)
            if any(part.startswith("_") for part in rel_path.parts):
                continue

            # Build dotted module name relative to the package
            rel = Path(*rel_path.parts).with_suffix("").as_posix().replace("/", ".")
            if rel.endswith(".__init__"):
                rel = rel[: -len(".__init__")]
            module_name = package_path if rel == "" else f"{package_path}.{rel}"

            # Import the module
            try:
                importlib.import_module(module_name)
                count += 1
                logger.debug(f"Discovered module file: {module_name}")
            except Exception as e:
                logger.error(f"Failed to import {module_name}: {e}")

        logger.debug(f"Discovered {count} module files")
        return count

    except Exception as e:
        logger.error(f"Failed to discover modules in {package_path}: {e}")
        return 0


def load_module(module_path: str) -> bool:
    """Loads a module from an external plugin Python file.

    This allows loading modules from external Python files,
    enabling a true plugin system.

    Args:
        module_path: Path to the plugin Python file

    Returns:
        True if loaded successfully, False otherwise

    Example:
        ```python
        # Load external plugin
        load_plugin("plugins/my_custom_module.py")
        ```
    """
    try:
        # Import the plugin file as a module
        spec = importlib.util.spec_from_file_location("plugin", module_path)
        if spec is None or spec.loader is None:
            logger.error(f"Could not load plugin: {module_path}")
            return False

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        logger.info(f"Loaded plugin: {module_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to load plugin {module_path}: {e}")
        return False


def initialize_module_registry() -> ModuleRegistry:
    """Initialize the module registry and discovers all available modules.

    This function:
    1. Auto-discovers and registers all modules in the modules package
    2. Loads external plugins if configured

    Call this at application startup.
    """
    registry = get_registry()

    # Auto-discover and register all modules in the modules package
    # This will import all .py files and trigger their @register_module decorators
    discover_modules("src.gui.modules", recursive=True)
    logger.info(f"Auto-discovered and initialized {registry.count()} modules")

    return registry


# Export public API
__all__ = [
    "ModuleRegistry",
    "get_registry",
    "register_module",
    "discover_modules",
    "load_module",
    "initialize_module_registry",
]


# Global registry instance
_global_registry = ModuleRegistry()
