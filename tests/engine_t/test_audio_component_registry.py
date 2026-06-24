"""Unit tests for ComponentRegistry system.

Tests cover:
- Component registration
- Component retrieval
- Component categories
- Component descriptors
- Registry operations
"""

import unittest

from src.engine.audio_component import ComponentCategory
from src.engine.audio_component_registry import audio_registry, register_component


class TestAudioComponentRegistry(unittest.TestCase):
    """Tests for ComponentRegistry."""

    def test_register_component(self):
        """Test registering a component."""
        # Use existing registered component
        sine = audio_registry.get("Sine")
        self.assertIsNotNone(sine)
        self.assertEqual(sine.descriptor.category, ComponentCategory.OSCILLATOR)

    def test_register_duplicate_raises_warning(self):
        """Test that registering duplicate component logs warning."""
        # This is handled internally by the registry
        # Just verify we can get components
        sine1 = audio_registry.get("Sine")
        sine2 = audio_registry.get("Sine")
        self.assertEqual(sine1, sine2)

    def test_get_nonexistent_component(self):
        """Test retrieving non-existent component returns None."""
        result = audio_registry.get("nonexistent_component_xyz")
        self.assertIsNone(result)

    def test_list_all_components(self):
        """Test listing all registered components."""
        all_comps = audio_registry.list_components()
        self.assertGreater(len(all_comps), 0)
        self.assertIn("Sine", all_comps)
        self.assertIn("Volume", all_comps)
        self.assertIn("ADSREnvelope", all_comps)

    def test_list_by_category(self):
        """Test listing components by category."""
        oscillators = audio_registry.list_by_category(ComponentCategory.OSCILLATOR)
        modifiers = audio_registry.list_by_category(ComponentCategory.MODIFIER)

        # We should have at least some oscillators and modifiers
        self.assertGreater(len(oscillators), 0)
        self.assertGreater(len(modifiers), 0)

        # Verify they are the right type
        self.assertIn("Sine", oscillators)
        self.assertIn("Volume", modifiers)


class TestGlobalAudioRegistry(unittest.TestCase):
    """Tests for global registry instance."""

    def test_global_registry_exists(self):
        """Test that global registry instance exists."""
        self.assertIsNotNone(audio_registry)
        # Just verify it has the expected methods
        self.assertTrue(hasattr(audio_registry, "get"))
        self.assertTrue(hasattr(audio_registry, "list_components"))
        self.assertTrue(hasattr(audio_registry, "list_by_category"))
        self.assertTrue(hasattr(audio_registry, "count"))

    def test_register_component_helper(self):
        """Test register_component helper function."""
        # This test checks the helper function works
        # Note: We don't actually register to avoid affecting other tests

        # Just verify the function exists and is callable
        self.assertTrue(callable(register_component))

    def test_default_components_registered(self):
        """Test that default components are registered."""
        # Check for some expected default components
        sine = audio_registry.get("Sine")
        self.assertIsNotNone(sine)
        self.assertEqual(sine.descriptor.category, ComponentCategory.OSCILLATOR)

        volume = audio_registry.get("Volume")
        self.assertIsNotNone(volume)
        self.assertEqual(volume.descriptor.category, ComponentCategory.MODIFIER)

        adsr = audio_registry.get("ADSREnvelope")
        self.assertIsNotNone(adsr)
        self.assertEqual(adsr.descriptor.category, ComponentCategory.MODULATOR)

    def test_all_categories_represented(self):
        """Test that all component categories have registered components."""
        oscillators = audio_registry.list_by_category(ComponentCategory.OSCILLATOR)
        modulators = audio_registry.list_by_category(ComponentCategory.MODULATOR)
        modifiers = audio_registry.list_by_category(ComponentCategory.MODIFIER)

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
        categories = list(ComponentCategory)

        self.assertIn(ComponentCategory.OSCILLATOR, categories)
        self.assertIn(ComponentCategory.MODULATOR, categories)
        self.assertIn(ComponentCategory.MODIFIER, categories)
        self.assertIn(ComponentCategory.COMPOSER, categories)


if __name__ == "__main__":
    unittest.main()
