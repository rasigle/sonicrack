"""Dynamic module registration system for the modular synth.

This module provides decorators and utilities for automatic module registration,
enabling a plugin-like architecture where modules can self-register without
modifying the central registry.

New modules should use the @register_module decorator in their own files.
See dynamic_registry.py for the plugin system documentation.
"""
import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Type, Callable

from src.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)

class ModuleRegistry:
    """Central registry for all available modules.

    Modules can be registered using the @register_module decorator,
    or programmatically via register() method.

    Example:
        ```python
        @register_module("My Module", category="oscillator")
        class MyModule(ModuleWidget):
            ...
        ```
    """

    def __init__(self):
        """Initialize the module registry."""
        self._modules: dict[str, Type[ModuleWidget]] = {}
        self._categories: dict[str, list[str]] = {}
        self._metadata: dict[str, dict] = {}

    def register(
        self,
        module_class: Type[ModuleWidget],
        **override_metadata
    ) -> Type[ModuleWidget]:
        """Register a module class.

        Metadata is automatically extracted from the module class properties:
        - name: from module_title property
        - category: from module_category property
        - description: from module_description property
        - version: from module_version property
        - author: from module_author property

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

        # Create a temporary instance to extract metadata
        # We need to be careful here - some modules might need parameters
        # So we'll use a try/except and fall back to class attributes
        try:
            temp_instance = module_class()
            name = temp_instance.module_title
            category_enum = temp_instance.module_category
            category = category_enum.value if hasattr(category_enum, 'value') else str(category_enum)
            description = temp_instance.module_description
            version = temp_instance.module_version
            author = temp_instance.module_author
        except Exception as e:
            logger.warning(f"Could not instantiate {module_class.__name__} to extract metadata: {e}")
            # Fall back to defaults
            name = module_class.__name__
            category = "other"
            description = ""
            version = "1.0.0"
            author = ""

        # Allow overriding extracted metadata
        name = override_metadata.get("name", name)
        category = override_metadata.get("category", category)
        description = override_metadata.get("description", description)
        version = override_metadata.get("version", version)
        author = override_metadata.get("author", author)

        # Check for duplicate names
        if name in self._modules:
            logger.warning(
                f"Module '{name}' already registered, overwriting with {module_class.__name__}"
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
            **{k: v for k, v in override_metadata.items()
               if k not in ["name", "category", "description", "author", "version"]}
        }

        logger.info(f"Registered module: {name} ({module_class.__name__})")

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

        logger.info(f"Unregistered module: {name}")
        return True

    def get(self, name: str) -> Type[ModuleWidget] | None:
        """Get a module class by name.

        Args:
            name: Name of the module

        Returns:
            The module class, or None if not found
        """
        return self._modules.get(name)

    def get_all(self) -> dict[str, Type[ModuleWidget]]:
        """Get all registered modules.

        Returns:
            Dictionary mapping module names to classes
        """
        return self._modules.copy()

    def get_by_category(self, category: str) -> dict[str, Type[ModuleWidget]]:
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


# Global registry instance
_global_registry = ModuleRegistry()


def get_registry() -> ModuleRegistry:
    """Get the global module registry.

    Returns:
        The global ModuleRegistry instance
    """
    return _global_registry


def register_module(**override_metadata) -> Callable:
    """Decorator to register a module class.

    This decorator automatically registers the module when the class is defined.
    Metadata is extracted from the module's properties (module_title, module_category, etc.)

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
            @property
            def module_title(self):
                return "My Oscillator"

            @property
            def module_category(self):
                return ModuleCategory.SOURCE

        # With overrides
        @register_module(author="Your Name")
        class MyOscillator(ModuleWidget):
            ...
        ```
    """
    def decorator(cls: Type[ModuleWidget]) -> Type[ModuleWidget]:
        _global_registry.register(
            module_class=cls,
            **override_metadata
        )
        return cls

    return decorator


def discover_modules(package_path: str = "src.gui.modules") -> int:
    """Discover and import all module files in a package.

    This automatically imports all Python files in the modules package,
    which triggers the @register_module decorators.

    Args:
        package_path: Dot-separated package path to scan

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
        package_dir = Path(package.__file__).parent

        # Find all Python files
        module_files = package_dir.glob("*.py")

        count = 0
        for module_file in module_files:
            # Skip __init__.py and private modules
            if module_file.name.startswith("_"):
                continue

            # Import the module
            module_name = f"{package_path}.{module_file.stem}"
            try:
                importlib.import_module(module_name)
                count += 1
                logger.debug(f"Discovered module file: {module_name}")
            except Exception as e:
                logger.error(f"Failed to import {module_name}: {e}")

        logger.info(f"Discovered {count} module files")
        return count

    except Exception as e:
        logger.error(f"Failed to discover modules in {package_path}: {e}")
        return 0


def load_plugin(plugin_path: str) -> bool:
    """Load a module from an external plugin file.

    This allows loading modules from external Python files,
    enabling a true plugin system.

    Args:
        plugin_path: Path to the plugin Python file

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
        spec = importlib.util.spec_from_file_location("plugin", plugin_path)
        if spec is None or spec.loader is None:
            logger.error(f"Could not load plugin: {plugin_path}")
            return False

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        logger.info(f"Loaded plugin: {plugin_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to load plugin {plugin_path}: {e}")
        return False


# Backward compatibility: provide MODULE_REGISTRY dict-like interface
class _RegistryCompat:
    """Compatibility wrapper to make registry behave like a dict."""

    def __getitem__(self, key):
        result = _global_registry.get(key)
        if result is None:
            raise KeyError(f"Module '{key}' not found in registry")
        return result

    def __contains__(self, key):
        return _global_registry.get(key) is not None

    def get(self, key, default=None):
        return _global_registry.get(key) or default

    def keys(self):
        return _global_registry.list_modules()

    def values(self):
        return _global_registry.get_all().values()

    def items(self):
        return _global_registry.get_all().items()

    def __iter__(self):
        return iter(_global_registry.list_modules())

    def __len__(self):
        return _global_registry.count()


# Create the compatibility object
MODULE_REGISTRY = _RegistryCompat()

__all__ = [
    "ModuleRegistry",
    "get_registry",
    "register_module",
    "discover_modules",
    "load_plugin",
    "MODULE_REGISTRY",
    "initialize_modules",
]


def initialize_modules():
    """Initialize the module registry.

    This function:
    1. Auto-discovers and registers all modules in the modules package
    2. Can load external plugins if configured

    Call this at application startup.
    """
    registry = get_registry()

    # Auto-discover and register all modules in the modules package
    # This will import all .py files and trigger their @register_module decorators
    count = discover_modules("src.gui.modules")

    logger.info(f"Auto-discovered and initialized {registry.count()} modules")

    return registry
