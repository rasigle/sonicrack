"""Unit tests for default component registrations.

Tests cover:
- Oscillator component registrations
- Modulator component registrations
- Modifier component registrations
- Component factories work correctly
- Config parameter mappings
"""

import unittest

from src.builder.component_registry import registry, ComponentCategory
from src.engine.oscillator import SineOscillator, SquareOscillator, SawtoothOscillator, TriangleOscillator
from src.engine.modulator import ADSREnvelope
from src.engine.modifier import Volume, Panner, Clipper


class TestOscillatorRegistrations(unittest.TestCase):
    """Tests for oscillator component registrations."""

    def test_sine_oscillator_registered(self):
        """Test sine oscillator is registered correctly."""
        descriptor = registry.get("sine_oscillator")

        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.name, "sine_oscillator")
        self.assertEqual(descriptor.category, ComponentCategory.OSCILLATOR)
        self.assertEqual(descriptor.method_name, "sine")
        self.assertEqual(descriptor.factory, SineOscillator)

    def test_sine_oscillator_creation(self):
        """Test creating sine oscillator via registry."""
        descriptor = registry.get("sine_oscillator")
        instance = descriptor.create_instance(440, amplitude=0.8)

        self.assertIsInstance(instance, SineOscillator)
        self.assertEqual(instance.frequency, 440)
        self.assertEqual(instance.amplitude, 0.8)

    def test_square_oscillator_registered(self):
        """Test square oscillator is registered correctly."""
        descriptor = registry.get("square_oscillator")

        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.name, "square_oscillator")
        self.assertEqual(descriptor.category, ComponentCategory.OSCILLATOR)
        self.assertEqual(descriptor.method_name, "square")
        self.assertEqual(descriptor.factory, SquareOscillator)

    def test_square_oscillator_creation(self):
        """Test creating square oscillator via registry."""
        descriptor = registry.get("square_oscillator")
        instance = descriptor.create_instance(440)

        self.assertIsInstance(instance, SquareOscillator)

    def test_triangle_oscillator_registered(self):
        """Test triangle oscillator is registered correctly."""
        descriptor = registry.get("triangle_oscillator")

        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.name, "triangle_oscillator")
        self.assertEqual(descriptor.category, ComponentCategory.OSCILLATOR)
        self.assertEqual(descriptor.method_name, "triangle")
        self.assertEqual(descriptor.factory, TriangleOscillator)

    def test_triangle_oscillator_creation(self):
        """Test creating triangle oscillator via registry."""
        descriptor = registry.get("triangle_oscillator")
        instance = descriptor.create_instance(440)

        self.assertIsInstance(instance, TriangleOscillator)

    def test_sawtooth_oscillator_registered(self):
        """Test sawtooth oscillator is registered correctly."""
        descriptor = registry.get("sawtooth_oscillator")

        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.name, "sawtooth_oscillator")
        self.assertEqual(descriptor.category, ComponentCategory.OSCILLATOR)
        self.assertEqual(descriptor.method_name, "sawtooth")
        self.assertEqual(descriptor.factory, SawtoothOscillator)

    def test_sawtooth_oscillator_creation(self):
        """Test creating sawtooth oscillator via registry."""
        descriptor = registry.get("sawtooth_oscillator")
        instance = descriptor.create_instance(440)

        self.assertIsInstance(instance, SawtoothOscillator)

    def test_all_oscillators_have_common_params(self):
        """Test all oscillators have frequency, amplitude, phase, sample_rate in config."""
        oscillator_names = ["sine_oscillator", "square_oscillator", "triangle_oscillator", "sawtooth_oscillator"]

        for osc_name in oscillator_names:
            descriptor = registry.get(osc_name)
            config_params = descriptor.config_params

            self.assertIn("frequency", config_params, f"{osc_name} missing frequency")
            self.assertIn("amplitude", config_params, f"{osc_name} missing amplitude")
            self.assertIn("phase", config_params, f"{osc_name} missing phase")
            self.assertIn("sample_rate", config_params, f"{osc_name} missing sample_rate")


class TestModulatorRegistrations(unittest.TestCase):
    """Tests for modulator component registrations."""

    def test_adsr_envelope_registered(self):
        """Test ADSR envelope is registered correctly."""
        descriptor = registry.get("adsr_envelope")

        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.name, "adsr_envelope")
        self.assertEqual(descriptor.category, ComponentCategory.MODULATOR)
        self.assertEqual(descriptor.factory, ADSREnvelope)

    def test_adsr_envelope_creation(self):
        """Test creating ADSR envelope via registry."""
        descriptor = registry.get("adsr_envelope")
        instance = descriptor.create_instance(0.1, 0.2, 0.7, 0.3)

        self.assertIsInstance(instance, ADSREnvelope)

    def test_adsr_config_params(self):
        """Test ADSR has correct config parameters."""
        descriptor = registry.get("adsr_envelope")
        config_params = descriptor.config_params

        self.assertIn("attack_duration", config_params)
        self.assertIn("decay_duration", config_params)
        self.assertIn("sustain_level", config_params)
        self.assertIn("release_duration", config_params)
        self.assertIn("sample_rate", config_params)

    def test_adsr_to_config(self):
        """Test ADSR envelope config generation."""
        descriptor = registry.get("adsr_envelope")
        config = descriptor.to_config(0.1, 0.2, 0.7, 0.3, sample_rate=44100)

        self.assertEqual(config["type"], "adsr_envelope")
        self.assertEqual(config["attack_duration"], 0.1)
        self.assertEqual(config["decay_duration"], 0.2)
        self.assertEqual(config["sustain_level"], 0.7)
        self.assertEqual(config["release_duration"], 0.3)
        self.assertEqual(config["sample_rate"], 44100)


class TestModifierRegistrations(unittest.TestCase):
    """Tests for modifier component registrations."""

    def test_volume_registered(self):
        """Test volume modifier is registered correctly."""
        descriptor = registry.get("volume")

        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.name, "volume")
        self.assertEqual(descriptor.category, ComponentCategory.MODIFIER)
        self.assertEqual(descriptor.method_name, "volume")
        self.assertEqual(descriptor.factory, Volume)

    def test_volume_creation(self):
        """Test creating volume modifier via registry."""
        descriptor = registry.get("volume")
        instance = descriptor.create_instance(0.5)

        self.assertIsInstance(instance, Volume)
        self.assertEqual(instance.amplitude, 0.5)

    def test_volume_config_params(self):
        """Test volume has correct config parameters."""
        descriptor = registry.get("volume")
        config_params = descriptor.config_params

        self.assertIn("amplitude", config_params)

    def test_panner_registered(self):
        """Test panner modifier is registered correctly."""
        descriptor = registry.get("panner")

        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.name, "panner")
        self.assertEqual(descriptor.category, ComponentCategory.MODIFIER)
        self.assertEqual(descriptor.method_name, "panner")
        self.assertEqual(descriptor.factory, Panner)

    def test_panner_creation(self):
        """Test creating panner modifier via registry."""
        descriptor = registry.get("panner")
        instance = descriptor.create_instance(0.5)

        self.assertIsInstance(instance, Panner)
        self.assertEqual(instance.position, 0.5)

    def test_panner_config_params(self):
        """Test panner has correct config parameters."""
        descriptor = registry.get("panner")
        config_params = descriptor.config_params

        self.assertIn("position", config_params)

    def test_clipper_registered(self):
        """Test clipper modifier is registered correctly."""
        descriptor = registry.get("clipper")

        self.assertIsNotNone(descriptor)
        self.assertEqual(descriptor.name, "clipper")
        self.assertEqual(descriptor.category, ComponentCategory.MODIFIER)
        self.assertEqual(descriptor.method_name, "clipper")
        self.assertEqual(descriptor.factory, Clipper)

    def test_clipper_creation(self):
        """Test creating clipper modifier via registry."""
        descriptor = registry.get("clipper")
        instance = descriptor.create_instance((-0.8, 0.8))

        self.assertIsInstance(instance, Clipper)

    def test_clipper_config_params(self):
        """Test clipper has correct config parameters."""
        descriptor = registry.get("clipper")
        config_params = descriptor.config_params

        # Clipper uses clip_range parameter
        self.assertIn("clip_range", config_params)


class TestComponentCoverage(unittest.TestCase):
    """Tests for overall component coverage."""

    def test_minimum_components_registered(self):
        """Test that minimum expected components are registered."""
        all_components = registry.list_components()

        # Should have at least these components
        expected_components = [
            "sine_oscillator",
            "square_oscillator",
            "triangle_oscillator",
            "sawtooth_oscillator",
            "adsr_envelope",
            "volume",
            "panner",
            "clipper"
        ]

        for component in expected_components:
            self.assertIn(component, all_components, f"Missing {component}")

    def test_category_distribution(self):
        """Test that components are distributed across categories."""
        oscillators = registry.get_by_category(ComponentCategory.OSCILLATOR)
        modulators = registry.get_by_category(ComponentCategory.MODULATOR)
        modifiers = registry.get_by_category(ComponentCategory.MODIFIER)

        # Should have multiple of each
        self.assertGreaterEqual(len(oscillators), 4, "Should have at least 4 oscillators")
        self.assertGreaterEqual(len(modulators), 1, "Should have at least 1 modulator")
        self.assertGreaterEqual(len(modifiers), 3, "Should have at least 3 modifiers")

    def test_all_components_have_descriptions(self):
        """Test that all default components have descriptions."""
        all_components = registry.list_components()

        for comp_name in all_components:
            descriptor = registry.get(comp_name)
            # Description can be empty string, but should exist
            self.assertIsNotNone(descriptor.description)

    def test_all_components_creatable(self):
        """Test that all registered components can be instantiated."""
        # Test each oscillator
        sine = registry.get("sine_oscillator").create_instance(440)
        self.assertIsNotNone(sine)

        square = registry.get("square_oscillator").create_instance(440)
        self.assertIsNotNone(square)

        triangle = registry.get("triangle_oscillator").create_instance(440)
        self.assertIsNotNone(triangle)

        sawtooth = registry.get("sawtooth_oscillator").create_instance(440)
        self.assertIsNotNone(sawtooth)

        # Test modulator
        adsr = registry.get("adsr_envelope").create_instance(0.1, 0.2, 0.7, 0.3)
        self.assertIsNotNone(adsr)

        # Test modifiers
        volume = registry.get("volume").create_instance(0.5)
        self.assertIsNotNone(volume)

        panner = registry.get("panner").create_instance(0.0)
        self.assertIsNotNone(panner)

        clipper = registry.get("clipper").create_instance((-0.8, 0.8))
        self.assertIsNotNone(clipper)


if __name__ == '__main__':
    unittest.main()

