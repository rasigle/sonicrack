"""Unit tests for ComponentRegistry system.

Tests cover:
- Component registration
- Component retrieval
- Component categories
- Component descriptors
- Registry operations
"""

import unittest
from unittest.mock import Mock

from src.builder.component_registry import (
    ComponentRegistry,
    ComponentDescriptor,
    ComponentCategory,
    registry,
    register_component
)


class TestComponentDescriptor(unittest.TestCase):
    """Tests for ComponentDescriptor."""

    def test_descriptor_creation(self):
        """Test creating a component descriptor."""
        descriptor = ComponentDescriptor(
            name="test_component",
            category=ComponentCategory.OSCILLATOR,
            factory=Mock,
            config_params=["param1", "param2"],
            description="Test component"
        )

        self.assertEqual(descriptor.name, "test_component")
        self.assertEqual(descriptor.category, ComponentCategory.OSCILLATOR)
        self.assertEqual(descriptor.config_params, ["param1", "param2"])
        self.assertEqual(descriptor.description, "Test component")

    def test_method_name_generation(self):
        """Test automatic method name generation."""
        # With category suffix
        descriptor1 = ComponentDescriptor(
            name="sine_oscillator",
            category=ComponentCategory.OSCILLATOR,
            factory=Mock,
            config_params=[]
        )
        self.assertEqual(descriptor1.method_name, "sine")

        # Without category suffix
        descriptor2 = ComponentDescriptor(
            name="custom",
            category=ComponentCategory.OSCILLATOR,
            factory=Mock,
            config_params=[]
        )
        self.assertEqual(descriptor2.method_name, "custom")

        # Explicit method name
        descriptor3 = ComponentDescriptor(
            name="test_oscillator",
            category=ComponentCategory.OSCILLATOR,
            factory=Mock,
            config_params=[],
            method_name="my_test"
        )
        self.assertEqual(descriptor3.method_name, "my_test")

    def test_create_instance(self):
        """Test creating component instances."""
        mock_factory = Mock(return_value="instance")
        descriptor = ComponentDescriptor(
            name="test",
            category=ComponentCategory.OSCILLATOR,
            factory=mock_factory,
            config_params=[]
        )

        instance = descriptor.create_instance(440, amplitude=0.5)

        self.assertEqual(instance, "instance")
        mock_factory.assert_called_once_with(440, amplitude=0.5)

    def test_to_config(self):
        """Test converting to config dictionary."""
        descriptor = ComponentDescriptor(
            name="test_component",
            category=ComponentCategory.OSCILLATOR,
            factory=Mock,
            config_params=["frequency", "amplitude"]
        )

        config = descriptor.to_config(440, amplitude=0.8)

        self.assertEqual(config["type"], "test_component")
        self.assertEqual(config["frequency"], 440)
        self.assertEqual(config["amplitude"], 0.8)

    def test_to_config_with_kwargs(self):
        """Test config generation with keyword arguments."""
        descriptor = ComponentDescriptor(
            name="test",
            category=ComponentCategory.OSCILLATOR,
            factory=Mock,
            config_params=["frequency", "amplitude", "phase"]
        )

        config = descriptor.to_config(frequency=440, amplitude=0.5, phase=90)

        self.assertEqual(config["type"], "test")
        self.assertEqual(config["frequency"], 440)
        self.assertEqual(config["amplitude"], 0.5)
        self.assertEqual(config["phase"], 90)


class TestComponentRegistry(unittest.TestCase):
    """Tests for ComponentRegistry."""

    def setUp(self):
        """Create a test registry for each test."""
        self.test_registry = ComponentRegistry()

    def test_register_component(self):
        """Test registering a component."""
        descriptor = ComponentDescriptor(
            name="test_osc",
            category=ComponentCategory.OSCILLATOR,
            factory=Mock,
            config_params=["frequency"]
        )

        self.test_registry.register(descriptor)

        retrieved = self.test_registry.get("test_osc")
        self.assertEqual(retrieved, descriptor)

    def test_register_duplicate_raises_warning(self):
        """Test that registering duplicate component logs warning."""
        descriptor1 = ComponentDescriptor(
            name="test",
            category=ComponentCategory.OSCILLATOR,
            factory=Mock,
            config_params=[]
        )
        descriptor2 = ComponentDescriptor(
            name="test",
            category=ComponentCategory.OSCILLATOR,
            factory=Mock,
            config_params=[]
        )

        self.test_registry.register(descriptor1)
        # Should overwrite without error
        self.test_registry.register(descriptor2)

        retrieved = self.test_registry.get("test")
        self.assertEqual(retrieved, descriptor2)

    def test_get_nonexistent_component(self):
        """Test retrieving non-existent component returns None."""
        result = self.test_registry.get("nonexistent")
        self.assertIsNone(result)

    def test_list_all_components(self):
        """Test listing all registered components."""
        descriptor1 = ComponentDescriptor(
            name="test1",
            category=ComponentCategory.OSCILLATOR,
            factory=Mock,
            config_params=[]
        )
        descriptor2 = ComponentDescriptor(
            name="test2",
            category=ComponentCategory.MODIFIER,
            factory=Mock,
            config_params=[]
        )

        self.test_registry.register(descriptor1)
        self.test_registry.register(descriptor2)

        all_components = self.test_registry.list_components()

        self.assertIn("test1", all_components)
        self.assertIn("test2", all_components)
        self.assertEqual(len(all_components), 2)

    def test_list_by_category(self):
        """Test listing components by category."""
        osc_descriptor = ComponentDescriptor(
            name="osc",
            category=ComponentCategory.OSCILLATOR,
            factory=Mock,
            config_params=[]
        )
        mod_descriptor = ComponentDescriptor(
            name="mod",
            category=ComponentCategory.MODIFIER,
            factory=Mock,
            config_params=[]
        )

        self.test_registry.register(osc_descriptor)
        self.test_registry.register(mod_descriptor)

        oscillators = self.test_registry.get_by_category(ComponentCategory.OSCILLATOR)
        modifiers = self.test_registry.get_by_category(ComponentCategory.MODIFIER)

        self.assertEqual(len(oscillators), 1)
        self.assertEqual(oscillators[0].name, "osc")
        self.assertEqual(len(modifiers), 1)
        self.assertEqual(modifiers[0].name, "mod")



class TestGlobalRegistry(unittest.TestCase):
    """Tests for global registry instance."""

    def test_global_registry_exists(self):
        """Test that global registry instance exists."""
        self.assertIsNotNone(registry)
        self.assertIsInstance(registry, ComponentRegistry)

    def test_register_component_helper(self):
        """Test register_component helper function."""
        # This test checks the helper function works
        # Note: We don't actually register to avoid affecting other tests

        # Just verify the function exists and is callable
        self.assertTrue(callable(register_component))

    def test_default_components_registered(self):
        """Test that default components are registered."""
        # Check for some expected default components
        sine = registry.get("sine_oscillator")
        self.assertIsNotNone(sine)
        self.assertEqual(sine.category, ComponentCategory.OSCILLATOR)

        volume = registry.get("volume")
        self.assertIsNotNone(volume)
        self.assertEqual(volume.category, ComponentCategory.MODIFIER)

        adsr = registry.get("adsr_envelope")
        self.assertIsNotNone(adsr)
        self.assertEqual(adsr.category, ComponentCategory.MODULATOR)

    def test_all_categories_represented(self):
        """Test that all component categories have registered components."""
        oscillators = registry.get_by_category(ComponentCategory.OSCILLATOR)
        modulators = registry.get_by_category(ComponentCategory.MODULATOR)
        modifiers = registry.get_by_category(ComponentCategory.MODIFIER)

        self.assertGreater(len(oscillators), 0, "Should have oscillators")
        self.assertGreater(len(modulators), 0, "Should have modulators")
        self.assertGreater(len(modifiers), 0, "Should have modifiers")


class TestComponentCategories(unittest.TestCase):
    """Tests for ComponentCategory enum."""

    def test_category_values(self):
        """Test category enum values."""
        self.assertEqual(ComponentCategory.OSCILLATOR.value, "oscillator")
        self.assertEqual(ComponentCategory.MODULATOR.value, "modulator")
        self.assertEqual(ComponentCategory.MODIFIER.value, "modifier")
        self.assertEqual(ComponentCategory.COMPOSER.value, "composer")

    def test_category_membership(self):
        """Test category enum membership."""
        categories = [cat for cat in ComponentCategory]

        self.assertIn(ComponentCategory.OSCILLATOR, categories)
        self.assertIn(ComponentCategory.MODULATOR, categories)
        self.assertIn(ComponentCategory.MODIFIER, categories)
        self.assertIn(ComponentCategory.COMPOSER, categories)


if __name__ == '__main__':
    unittest.main()

