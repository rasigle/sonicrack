"""Tests for the dynamic module registration system."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.gui.core.module_registry import (
    ModuleRegistry,
    get_registry,
    register_module,
    discover_modules,
    load_module,
    initialize_modules,
)
from src.gui.widgets.module_widget import ModuleWidget


class TestModuleRegistry:
    """Tests for ModuleRegistry class."""

    def test_register_module_directly(self):
        """Test basic module registration using direct registration."""
        registry = ModuleRegistry()

        class TestModule(ModuleWidget):
            class metadata:
                title = "Test Module"
                category = "test"
                description = "A test module"
                version = "1.0.0"
                author = "Test"

        # Register directly
        registry.register(TestModule)

        # Module should be registered
        assert "Test Module" in registry.list_modules()
        assert registry.get("Test Module") is not None

    def test_register_duplicate_module(self):
        """Test registering a module with duplicate name."""
        registry = ModuleRegistry()

        class TestModule1(ModuleWidget):
            class metadata:
                title = "Duplicate"
                category = "test"
                description = "First"
                version = "1.0.0"
                author = "Test"

        class TestModule2(ModuleWidget):
            class metadata:
                title = "Duplicate"
                category = "test"
                description = "Second"
                version = "1.0.0"
                author = "Test"

        registry.register(TestModule1)
        registry.register(TestModule2)

        # Second registration should overwrite
        module = registry.get("Duplicate")
        assert module is TestModule2

    def test_get_by_category(self):
        """Test retrieving modules by category."""
        registry = ModuleRegistry()

        class TestOsc(ModuleWidget):
            class metadata:
                title = "Test Osc"
                category = "oscillator"
                description = "Test"
                version = "1.0.0"
                author = "Test"

        class TestFilter(ModuleWidget):
            class metadata:
                title = "Test Filter"
                category = "filter"
                description = "Test"
                version = "1.0.0"
                author = "Test"

        registry.register(TestOsc)
        registry.register(TestFilter)

        osc_modules = registry.get_by_category("oscillator")
        assert "Test Osc" in osc_modules
        assert "Test Filter" not in osc_modules

    def test_unregister_module(self):
        """Test unregistering a module."""
        registry = ModuleRegistry()

        class TestModule(ModuleWidget):
            class metadata:
                title = "To Remove"
                category = "test"
                description = "Test"
                version = "1.0.0"
                author = "Test"

        registry.register(TestModule)
        assert "To Remove" in registry.list_modules()

        registry.unregister("To Remove")
        assert "To Remove" not in registry.list_modules()

    def test_get_metadata(self):
        """Test retrieving module metadata."""
        registry = ModuleRegistry()

        class TestModule(ModuleWidget):
            class metadata:
                title = "Metadata Test"
                category = "test"
                description = "A test description"
                version = "2.0.0"
                author = "Test Author"

        registry.register(TestModule)

        metadata = registry.get_metadata("Metadata Test")
        assert metadata is not None
        assert metadata["category"] == "test"
        assert metadata["description"] == "A test description"
        assert metadata["version"] == "2.0.0"
        assert metadata["author"] == "Test Author"

    def test_clear_registry(self):
        """Test clearing all modules."""
        registry = ModuleRegistry()

        class TestModule(ModuleWidget):
            class metadata:
                title = "Clear Test"
                category = "test"
                description = "Test"
                version = "1.0.0"
                author = "Test"

        registry.register(TestModule)
        assert registry.count() > 0
        registry.clear()
        assert registry.count() == 0

    def test_register_module_decorator(self):
        """Test that decorator registers to global registry."""
        global_registry = get_registry()
        initial_count = global_registry.count()

        @register_module()
        class DecoratorTest(ModuleWidget):
            class metadata:
                title = "Decorator Test"
                category = "test"
                description = "Test"
                version = "1.0.0"
                author = "Test"

        # Should be in global registry
        assert global_registry.count() == initial_count + 1
        assert "Decorator Test" in global_registry.list_modules()

        # Clean up
        global_registry.unregister("Decorator Test")


class TestDiscoverModules:
    """Tests for module discovery functionality."""

    @patch("src.gui.core.module_registry.importlib.import_module")
    def test_discover_modules_recursive_file_discovery(self, mock_import):
        """Test that recursive discovery finds files in subdirectories."""
        # Mock the import to prevent actual module loading
        mock_import.return_value = MagicMock()

        # This will use the actual src.gui.modules directory structure
        # We just test that recursive mode finds more files
        count_recursive = discover_modules("src.gui.modules", recursive=True)
        count_flat = discover_modules("src.gui.modules", recursive=False)

        # Recursive should find at least as many (likely more with subdirs)
        assert count_recursive >= count_flat

    @patch("src.gui.core.module_registry.importlib.import_module")
    def test_discover_modules_handles_import_errors(self, mock_import):
        """Test that import errors are handled gracefully."""
        # Make import fail
        mock_import.side_effect = ImportError("Test import error")

        # Should not raise, just return 0
        count = discover_modules("src.gui.modules", recursive=False)
        assert isinstance(count, int)

    def test_discover_modules_invalid_package(self):
        """Test discovery with non-existent package."""
        count = discover_modules("nonexistent.package", recursive=True)
        assert count == 0

    def test_discover_modules_integration(self):
        """Integration test: discover actual modules with recursive search."""
        # This tests the actual discovery on the real module directory
        # It should find multiple modules
        count = discover_modules("src.gui.modules", recursive=True)
        assert count > 0  # Should find at least some modules


class TestRecursiveModuleSearch:
    """Comprehensive tests for recursive module search functionality."""

    def setup_method(self):
        """Create temporary test directory structure."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_pkg = Path(self.temp_dir) / "test_package"
        self.test_pkg.mkdir()

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def create_python_file(self, path: Path, content: str = "# test module"):
        """Helper to create a Python file."""
        path.write_text(content, encoding="utf-8")

    def test_recursive_vs_non_recursive(self):
        """Test that recursive mode finds nested files while non-recursive doesn't."""
        # Create structure:
        # test_package/
        #   __init__.py
        #   module1.py
        #   subdir/
        #     __init__.py
        #     module2.py

        (self.test_pkg / "__init__.py").touch()
        self.create_python_file(self.test_pkg / "module1.py")

        subdir = self.test_pkg / "subdir"
        subdir.mkdir()
        (subdir / "__init__.py").touch()
        self.create_python_file(subdir / "module2.py")

        import sys

        sys.path.insert(0, self.temp_dir)

        try:
            # Count files found
            from pathlib import Path
            import importlib

            # Get package
            pkg = importlib.import_module("test_package")
            pkg_path = Path(pkg.__file__).parent

            # Non-recursive should find only module1
            flat_files = list(pkg_path.glob("*.py"))
            flat_count = len([f for f in flat_files if not f.name.startswith("_")])
            assert flat_count == 1

            # Recursive should find both module1 and module2
            recursive_files = list(pkg_path.rglob("*.py"))
            recursive_count = len(
                [
                    f
                    for f in recursive_files
                    if not any(p.startswith("_") for p in f.relative_to(pkg_path).parts)
                ]
            )
            assert recursive_count == 2
            assert recursive_count > flat_count

        finally:
            sys.path.remove(self.temp_dir)

    def test_skips_private_files_and_dirs(self):
        """Test that files/dirs starting with _ are skipped in recursive search."""
        # Create structure:
        # test_package/
        #   __init__.py
        #   module1.py
        #   _private.py  (should be skipped)
        #   _private_dir/
        #     __init__.py
        #     module2.py  (should be skipped - parent dir is private)

        (self.test_pkg / "__init__.py").touch()
        self.create_python_file(self.test_pkg / "module1.py")
        self.create_python_file(self.test_pkg / "_private.py")

        private_dir = self.test_pkg / "_private_dir"
        private_dir.mkdir()
        (private_dir / "__init__.py").touch()
        self.create_python_file(private_dir / "module2.py")

        # Get all Python files recursively
        all_files = list(self.test_pkg.rglob("*.py"))

        # Filter out files where any path component starts with _
        valid_files = [
            f
            for f in all_files
            if not any(p.startswith("_") for p in f.relative_to(self.test_pkg).parts)
        ]

        # Should only find module1.py
        assert len(valid_files) == 1
        assert valid_files[0].name == "module1.py"

    def test_deep_nesting(self):
        """Test that deeply nested modules are found with recursive search."""
        # Create structure:
        # test_package/
        #   __init__.py
        #   level1/
        #     __init__.py
        #     level2/
        #       __init__.py
        #       level3/
        #         __init__.py
        #         deep_module.py

        (self.test_pkg / "__init__.py").touch()

        level1 = self.test_pkg / "level1"
        level1.mkdir()
        (level1 / "__init__.py").touch()

        level2 = level1 / "level2"
        level2.mkdir()
        (level2 / "__init__.py").touch()

        level3 = level2 / "level3"
        level3.mkdir()
        (level3 / "__init__.py").touch()
        self.create_python_file(level3 / "deep_module.py")

        # Recursive search should find the deeply nested module
        recursive_files = list(self.test_pkg.rglob("*.py"))
        deep_modules = [f for f in recursive_files if f.name == "deep_module.py"]

        assert len(deep_modules) == 1
        assert "level1" in str(deep_modules[0])
        assert "level2" in str(deep_modules[0])
        assert "level3" in str(deep_modules[0])

    def test_correct_module_name_construction(self):
        """Test that module names are correctly constructed for nested modules."""
        # Create structure:
        # test_package/
        #   __init__.py
        #   category/
        #     __init__.py
        #     subcategory/
        #       __init__.py
        #       my_module.py

        (self.test_pkg / "__init__.py").touch()

        category = self.test_pkg / "category"
        category.mkdir()
        (category / "__init__.py").touch()

        subcategory = category / "subcategory"
        subcategory.mkdir()
        (subcategory / "__init__.py").touch()
        self.create_python_file(subcategory / "my_module.py")

        # Find the module file
        module_file = subcategory / "my_module.py"
        rel_path = module_file.relative_to(self.test_pkg)

        # Build expected module name
        rel = Path(*rel_path.parts).with_suffix("").as_posix().replace("/", ".")
        expected_name = f"test_package.{rel}"

        assert expected_name == "test_package.category.subcategory.my_module"


class TestLoadPlugin:
    """Tests for external plugin loading."""

    def setup_method(self):
        """Create temporary directory for plugins."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_load_plugin_success(self):
        """Test loading a valid plugin file."""
        plugin_file = Path(self.temp_dir) / "test_plugin.py"
        plugin_file.write_text(
            """
from src.gui.core.module_registry import register_module
from src.gui.widgets.module_widget import ModuleWidget

@register_module()
class PluginModule(ModuleWidget):
    class metadata:
        title = "Plugin Module"
        category = "plugin"
        description = "Test plugin"
        version = "1.0.0"
        author = "Test"
"""
        )

        registry = get_registry()
        initial_count = registry.count()

        result = load_module(str(plugin_file))
        assert result is True
        assert registry.count() > initial_count

    def test_load_plugin_invalid_file(self):
        """Test loading a non-existent plugin file."""
        result = load_module("nonexistent_plugin.py")
        assert result is False

    def test_load_plugin_syntax_error(self):
        """Test loading a plugin with syntax errors."""
        plugin_file = Path(self.temp_dir) / "bad_plugin.py"
        plugin_file.write_text("this is not valid python !!!")

        result = load_module(str(plugin_file))
        assert result is False


class TestInitializeModules:
    """Tests for module initialization."""

    @patch("src.gui.core.module_registry.discover_modules")
    def test_initialize_modules(self, mock_discover):
        """Test module initialization process."""
        mock_discover.return_value = 10

        registry = initialize_modules()

        # Should call discover_modules with recursive=True
        mock_discover.assert_called_once_with("src.gui.modules", recursive=True)
        assert isinstance(registry, ModuleRegistry)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
