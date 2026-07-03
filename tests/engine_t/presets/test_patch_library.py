"""Unit tests for PresetLibrary

Tests cover:
- Preset saving and loading
- Preset library management
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.engine.presets import PresetBuilder, PresetLibrary


class TestPresetSaveLoad(unittest.TestCase):
    """Tests for preset saving and loading."""

    def test_save_and_load_simple_preset(self):
        """Test saving and loading a simple preset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_preset.json"

            # Create and save patch
            builder = PresetBuilder().sine(440, amplitude=0.8).volume(0.5)
            builder.save_preset(filepath)

            # File should exist
            self.assertTrue(filepath.exists())

            # Load preset
            loaded_builder = PresetBuilder.from_preset(filepath)
            loaded_patch = loaded_builder.build()

            # Should generate samples
            samples = loaded_patch.get_samples(1000)
            self.assertEqual(len(samples), 1000)

    def test_save_and_load_complex_preset(self):
        """Test saving and loading a complex preset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "complex_preset.json"

            # Create complex patch
            builder = (
                PresetBuilder()
                .sine(440, amplitude=0.8, phase=45)
                .adsr(0.1, 0.2, 0.7, 0.3)
                .volume(0.6)
                .panner(-0.5)
                .clipper((-0.8, 0.8))
            )

            builder.save_preset(filepath)

            # Load and verify
            loaded_builder = PresetBuilder.from_preset(filepath)
            config = loaded_builder.get_config()

            # Check components were loaded
            self.assertEqual(len(config["components"]), 5)

            # Verify oscillator config
            osc_config = config["components"][0]
            self.assertEqual(osc_config["name"], "Sine")
            self.assertEqual(osc_config["frequency"], 440)
            self.assertEqual(osc_config["amplitude"], 0.8)
            self.assertEqual(osc_config["phase"], 45)

            # Verify ADSR config
            adsr_config = config["components"][1]
            self.assertEqual(adsr_config["name"], "ADSREnvelope")
            self.assertEqual(adsr_config["attack_duration"], 0.1)
            self.assertEqual(adsr_config["sustain_level"], 0.7)

    def test_preset_json_format(self):
        """Test that preset JSON has expected structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"

            builder = PresetBuilder().sine(440).volume(0.5)
            builder.save_preset(filepath)

            # Read JSON directly
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            # Check structure
            self.assertIn("version", data)
            self.assertIn("components", data)
            self.assertIn("graph", data)
            self.assertIsInstance(data["components"], list)
            self.assertEqual(data["version"], "1.0")
            self.assertEqual(data["graph"]["schema_version"], 1)
            self.assertEqual(len(data["graph"]["nodes"]), len(data["components"]))

    def test_load_nonexistent_preset_raises_error(self):
        """Test loading nonexistent preset raises error."""
        with self.assertRaises(FileNotFoundError):
            PresetBuilder.from_preset("nonexistent.json")


class TestPresetLibrary(unittest.TestCase):
    """Tests for PresetLibrary management."""

    def test_create_library(self):
        """Test creating a preset library."""
        with tempfile.TemporaryDirectory() as tmpdir:
            PresetLibrary(tmpdir)

            # Directory should exist
            self.assertTrue(Path(tmpdir).exists())

    def test_save_to_library(self):
        """Test saving preset to library."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PresetLibrary(tmpdir)

            builder = PresetBuilder().sine(440).volume(0.5)
            library.save(builder, "my_patch")

            # File should exist
            preset_file = Path(tmpdir) / "my_patch.json"
            self.assertTrue(preset_file.exists())

    def test_load_from_library(self):
        """Test loading preset from library."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PresetLibrary(tmpdir)

            # Save preset
            builder = PresetBuilder().sine(440).volume(0.5)
            library.save(builder, "test_patch")

            # Load preset
            loaded_builder = library.load("test_patch")
            patch = loaded_builder.build()

            # Should generate samples
            samples = patch.get_samples(1000)
            self.assertEqual(len(samples), 1000)

    def test_list_presets(self):
        """Test listing builder in library."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PresetLibrary(tmpdir)

            # Save multiple builder
            library.save(PresetBuilder().sine(440), "preset1")
            library.save(PresetBuilder().square(220), "preset2")
            library.save(PresetBuilder().triangle(880), "preset3")

            # List builder
            presets = library.list_presets()

            self.assertEqual(len(presets), 3)
            self.assertIn("preset1", presets)
            self.assertIn("preset2", presets)
            self.assertIn("preset3", presets)

    def test_save_with_category(self):
        """Test saving preset with category."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PresetLibrary(tmpdir)

            builder = PresetBuilder().sine(440)
            library.save(builder, "lead1", category="leads")

            # File should be in category subdirectory
            preset_file = Path(tmpdir) / "leads" / "lead1.json"
            self.assertTrue(preset_file.exists())

    def test_list_presets_by_category(self):
        """Test listing builder filtered by category."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PresetLibrary(tmpdir)

            # Save builder in different categories
            library.save(PresetBuilder().sine(440), "lead1", category="leads")
            library.save(PresetBuilder().sine(220), "bass1", category="bass")
            library.save(PresetBuilder().sine(110), "bass2", category="bass")

            # List bass builder
            bass_presets = library.list_presets(category="bass")

            self.assertEqual(len(bass_presets), 2)
            self.assertIn("bass/bass1", bass_presets)
            self.assertIn("bass/bass2", bass_presets)

    def test_get_categories(self):
        """Test getting list of categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PresetLibrary(tmpdir)

            # Save builder in categories
            library.save(PresetBuilder().sine(440), "p1", category="leads")
            library.save(PresetBuilder().sine(220), "p2", category="bass")
            library.save(PresetBuilder().sine(880), "p3", category="fx")

            # Get categories
            categories = library.get_categories()

            self.assertEqual(len(categories), 3)
            self.assertIn("leads", categories)
            self.assertIn("bass", categories)
            self.assertIn("fx", categories)

    def test_delete_preset(self):
        """Test deleting a preset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PresetLibrary(tmpdir)

            # Save preset
            library.save(PresetBuilder().sine(440), "to_delete")

            # Verify it exists
            self.assertIn("to_delete", library.list_presets())

            # Delete it
            library.delete("to_delete")

            # Should be gone
            self.assertNotIn("to_delete", library.list_presets())

    def test_save_with_metadata(self):
        """Test saving preset with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PresetLibrary(tmpdir)

            builder = PresetBuilder().sine(440)
            metadata = {
                "author": "Test Author",
                "description": "Test preset",
                "tags": ["synth", "lead"],
            }

            library.save(builder, "preset_with_meta", metadata=metadata)

            # Load and check metadata
            preset_file = Path(tmpdir) / "preset_with_meta.json"
            with open(preset_file, encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("metadata", data)
            self.assertEqual(data["metadata"]["author"], "Test Author")
            self.assertEqual(data["metadata"]["description"], "Test preset")
