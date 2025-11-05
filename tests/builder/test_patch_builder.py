"""Unit tests for PatchBuilder

Tests cover:
- Fluent API patch building
- Integration with existing components
"""

import unittest

import numpy as np

from constants import DEFAULT_GAIN_DB
from src.builder import PresetBuilder
from src.engine.oscillator import SineOscillator


class TestPatchBuilder(unittest.TestCase):
    """Tests for PatchBuilder fluent API."""

    def test_simple_sine_wave(self):
        """Test building a simple sine wave patch."""
        patch = PresetBuilder().sine(440).build()

        # Should be a SineOscillator
        self.assertIsInstance(patch, SineOscillator)

        # Should generate samples
        samples = patch.get_samples(1000)
        self.assertEqual(len(samples), 1000)
        self.assertTrue(np.all(np.abs(samples) <= 1.0))

    def test_oscillator_gain(self):
        patch = PresetBuilder().sine(440).build()

        # Should be a SineOscillator, by default gain_db=-20 maps to amplitude ~0.1
        self.assertIsInstance(patch, SineOscillator)
        self.assertEqual(patch.gain_db, DEFAULT_GAIN_DB)
        self.assertEqual(patch.amplitude, 0.1)

        patch = PresetBuilder().sine(440, gain_db=-6.0).build()
        self.assertAlmostEqual(float(patch.gain_db), -6.0, 8)
        self.assertAlmostEqual(patch.amplitude, 0.5, 1)

        # Update gain_db after creation
        patch.gain_db = 0
        self.assertAlmostEqual(float(patch.gain_db), 0.0, 8)
        self.assertAlmostEqual(patch.amplitude, 1.0)

    def test_oscillator_amplitude(self):
        patch = PresetBuilder().sine(440, amplitude=0.1).build()
        self.assertEqual(patch.amplitude, 0.1)
        self.assertAlmostEqual(patch.gain_db, DEFAULT_GAIN_DB)

        patch = PresetBuilder().sine(440, amplitude=1.0).build()
        self.assertEqual(patch.amplitude, 1.0)
        self.assertAlmostEqual(patch.gain_db, 0.0)

    def test_oscillator_types(self):
        """Test building patches with different oscillator types."""
        # Sine
        sine_patch = PresetBuilder().sine(440).build()
        sine_samples = sine_patch.get_samples(100)
        self.assertEqual(len(sine_samples), 100)

        # Square
        square_patch = PresetBuilder().square(440).build()
        square_samples = square_patch.get_samples(100)
        self.assertEqual(len(square_samples), 100)

        # Triangle
        triangle_patch = PresetBuilder().triangle(440).build()
        triangle_samples = triangle_patch.get_samples(100)
        self.assertEqual(len(triangle_samples), 100)

        # Sawtooth
        sawtooth_patch = PresetBuilder().sawtooth(440).build()
        sawtooth_samples = sawtooth_patch.get_samples(100)
        self.assertEqual(len(sawtooth_samples), 100)

    def test_with_volume(self):
        """Test adding volume control."""
        patch = PresetBuilder().sine(440).volume(0.5).build()

        # Generate samples
        samples = patch.get_samples(1000)
        self.assertEqual(len(samples), 1000)

        # Volume should reduce amplitude
        max_amp = np.max(np.abs(samples))
        self.assertLess(max_amp, 0.6)  # Should be around 0.5

    def test_with_pan(self):
        """Test adding stereo panning."""
        patch = PresetBuilder().sine(440).panner(1.0).build()

        # Generate samples
        samples = patch.get_samples(100)

        # Should be stereo (tuple of left, right for each sample)
        # Chain with Panner returns stereo tuples
        self.assertEqual(len(samples), 100)

    def test_with_clip(self):
        """Test adding clipping."""
        patch = PresetBuilder().sine(440, amplitude=2.0).clipper((-0.5, 0.5)).build()

        # Generate samples
        samples = patch.get_samples(1000)

        # All samples should be within clip range
        self.assertTrue(np.all(samples >= -0.5))
        self.assertTrue(np.all(samples <= 0.5))

    def test_with_adsr(self):
        """Test adding ADSR envelope."""
        patch = PresetBuilder().sine(440).adsr(0.1, 0.2, 0.7, 0.3).build()

        # Trigger the envelope to start attack phase
        # The patch is a ModulatedOscillator, modulators is a tuple
        if hasattr(patch, "modulators") and len(patch.modulators) > 0:
            # Get the first modulator (the ADSR envelope)
            envelope = patch.modulators[0]
            if hasattr(envelope, "trigger_note_on"):
                envelope.trigger_note_on()

        # Generate samples
        samples = patch.get_samples(44100)  # 1 second

        # Should have envelope shape
        # Attack phase should ramp up
        attack_samples = samples[: int(0.1 * 44100)]
        self.assertTrue(
            np.mean(np.abs(attack_samples[:100]))
            < np.mean(np.abs(attack_samples[-100:]))
        )

    def test_method_chaining(self):
        """Test fluent API method chaining."""
        patch = (
            PresetBuilder()
            .sine(440, amplitude=0.8)
            .adsr(0.1, 0.2, 0.7, 0.3)
            .volume(0.5)
            .panner(0.0)
            .clipper((-0.9, 0.9))
            .build()
        )

        # Should successfully build
        samples = patch.get_samples(1000)
        self.assertEqual(len(samples), 1000)

    def test_multiple_oscillators(self):
        """Test mixing multiple oscillators."""
        patch = (
            PresetBuilder()
            .add_oscillator(PresetBuilder().sine(220).build())
            .add_oscillator(PresetBuilder().sine(440).build())
            .add_oscillator(PresetBuilder().sine(880).build())
            .build()
        )

        # Should create a WaveAdder
        samples = patch.get_samples(1000)
        self.assertEqual(len(samples), 1000)

    def test_build_without_source_raises_error(self):
        """Test that building without oscillator raises error."""
        with self.assertRaises(ValueError):
            PresetBuilder().volume(0.5).build()

    def test_get_config(self):
        """Test getting configuration dictionary."""
        builder = (
            PresetBuilder()
            .sine(440, amplitude=0.8)
            .adsr(0.1, 0.2, 0.7, 0.3)
            .volume(0.5)
        )

        config = builder.get_config()

        # Should have version and components
        self.assertIn("version", config)
        self.assertIn("components", config)
        self.assertEqual(len(config["components"]), 3)  # sine, adsr, volume


class TestPatchBuilderIntegration(unittest.TestCase):
    """Integration tests with existing components."""

    def test_output_consistency_with_manual_creation(self):
        """Test that builder output matches manually created patch."""
        # Create patch manually
        from src.engine import SineOscillator, Chain, Volume

        manual_osc = SineOscillator(440, amplitude=0.8, gain_db=None)
        manual_patch = Chain(manual_osc, Volume(0.5))

        # Create same patch with builder
        builder_patch = (
            PresetBuilder().sine(440, amplitude=0.8, gain_db=None).volume(0.5).build()
        )

        # Both should generate similar samples
        manual_samples = manual_patch.get_samples(1000, reset=True)
        builder_samples = builder_patch.get_samples(1000, reset=True)

        # Should be very similar (allow for small differences)
        correlation = np.corrcoef(manual_samples, builder_samples)[0, 1]
        self.assertGreater(correlation, 0.99)

    def test_sample_rate_configuration(self):
        """Test setting custom sample rate."""
        custom_sr = 48000
        patch = PresetBuilder().set_sample_rate(custom_sr).sine(440).build()

        # Should use custom sample rate
        self.assertEqual(patch._sample_rate, custom_sr)


class TestPatchBuilderConvenience(unittest.TestCase):
    """Tests for convenience methods (describe, summary, modify, etc.)."""

    def test_name_and_description(self):
        """Test setting and getting patch name and description."""
        # Test with constructor
        patch1 = PresetBuilder("My Synth", "A cool synthesizer")
        self.assertEqual(patch1.get_name(), "My Synth")
        self.assertEqual(patch1.get_description(), "A cool synthesizer")

        # Test with setter methods
        patch2 = (
            PresetBuilder()
            .set_name("Lead Synth")
            .set_description("Bright lead sound")
            .sine(440)
        )
        self.assertEqual(patch2.get_name(), "Lead Synth")
        self.assertEqual(patch2.get_description(), "Bright lead sound")

        # Test default values
        patch3 = PresetBuilder()
        self.assertEqual(patch3.get_name(), "Untitled Preset")
        self.assertEqual(patch3.get_description(), "")

    def test_get_source(self):
        """Test accessing source oscillator."""
        from src.engine.oscillator import SineOscillator

        patch = PresetBuilder().sine(440)
        source = patch.get_source()

        self.assertIsInstance(source, SineOscillator)
        self.assertEqual(source.frequency, 440)

    def test_get_modifiers(self):
        """Test accessing modifiers list."""
        from src.engine.modifier import Volume, Panner

        patch = PresetBuilder().sine(440).volume(0.5).panner(0.3)
        modifiers = patch.get_modifiers()

        self.assertEqual(len(modifiers), 2)
        self.assertIsInstance(modifiers[0], Volume)
        self.assertIsInstance(modifiers[1], Panner)

    def test_get_modulators(self):
        """Test accessing modulators dictionary."""
        patch = PresetBuilder().sine(440).adsr(0.1, 0.2, 0.7, 0.3)
        modulators = patch.get_modulators()

        self.assertIn("amplitude_mod", modulators)

    def test_get_components(self):
        """Test accessing all components."""
        patch = (
            PresetBuilder("Test Patch").sine(440).adsr(0.1, 0.2, 0.7, 0.3).volume(0.5)
        )

        components = patch.get_components()

        self.assertIn("source", components)
        self.assertIn("modifiers", components)
        self.assertIn("modulators", components)
        self.assertIn("name", components)
        self.assertIn("description", components)
        self.assertIn("sample_rate", components)

        self.assertEqual(components["name"], "Test Patch")
        self.assertEqual(len(components["modifiers"]), 1)

    def test_describe(self):
        """Test patch description."""
        builder = (
            PresetBuilder("My Lead")
            .set_description("Bright lead sound")
            .sine(440, amplitude=0.8, gain_db=None)
            .adsr(0.1, 0.2, 0.7, 0.3)
            .volume(0.5)
            .panner(0.3)
        )

        description = builder.describe().lower()

        # Should contain name and description
        self.assertIn("my lead", description)
        self.assertIn("bright lead sound", description)
        # Should contain component descriptions
        self.assertIn("sine wave oscillator", description)
        self.assertIn("440", description)
        self.assertIn("adsr envelope", description)
        self.assertIn("volume", description)
        self.assertIn("pan", description)

    def test_summary(self):
        """Test patch summary statistics."""
        builder = (
            PresetBuilder("Test")
            .sine(440)
            .adsr(0.1, 0.2, 0.7, 0.3)
            .volume(0.5)
            .panner(0.3)
        )

        summary = builder.summary()

        # Check summary structure
        self.assertEqual(summary["name"], "Test")
        self.assertEqual(summary["oscillators"], 1)
        self.assertEqual(summary["modulators"], 1)
        self.assertEqual(summary["effects"], 2)  # volume + pan
        self.assertEqual(summary["components"], 4)

    def test_clone(self):
        """Test patch cloning."""
        original = PresetBuilder().sine(440).volume(0.5)
        clone = original.clone()

        # Clone should have same config
        self.assertEqual(clone.get_config(), original.get_config())

        # But modifying clone shouldn't affect original
        clone.volume(0.7)

        # Get volume values from both configs
        orig_config = original.get_config()
        clone_config = clone.get_config()

        orig_volume = None
        clone_volume = None

        for comp in orig_config["components"]:
            if comp["name"] == "Volume":
                orig_volume = comp["amplitude"]

        for comp in clone_config["components"]:
            if comp["name"] == "Volume":
                clone_volume = comp["amplitude"]

        # Original should still have 0.5, clone should have 0.7
        self.assertEqual(orig_volume, 0.5)
        self.assertEqual(clone_volume, 0.7)

    def test_modify_frequency(self):
        """Test modifying patch frequency."""
        patch = PresetBuilder().sine(440)

        # Modify frequency
        patch.modify_frequency(880)

        # Check config was updated
        config = patch.get_config()
        osc_config = config["components"][0]
        self.assertEqual(osc_config["frequency"], 880)

    def test_modify_amplitude(self):
        """Test modifying patch amplitude."""
        patch = PresetBuilder().sine(440, amplitude=1.0)

        # Modify amplitude
        patch.modify_amplitude(0.5)

        # Check config was updated
        config = patch.get_config()
        osc_config = config["components"][0]
        self.assertEqual(osc_config["amplitude"], 0.5)

    def test_clear_effects(self):
        """Test clearing effects from patch."""
        builder = PresetBuilder().sine(440).volume(0.5).panner(0.3).clipper((-0.9, 0.9))

        # Clear effects
        builder.clear_effects()

        # Should only have oscillator left
        config = builder.get_config()
        component_categorys = [c["name"] for c in config["components"]]
        self.assertIn("Sine", component_categorys)
        self.assertNotIn("Volume", component_categorys)
        self.assertNotIn("Panner", component_categorys)
        self.assertNotIn("Clipper", component_categorys)


if __name__ == "__main__":
    unittest.main()
