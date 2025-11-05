"""Comprehensive demonstration of the preset system.

This example showcases the full capabilities of the preset system including:
- Creating presetes with the fluent API
- Saving and loading individual presets
- Organizing presets into categories
- Managing a preset library
- Importing and exporting presets
- Modifying and cloning presetes
- Batch operations on presets
- Metadata management
- Preset browsing and discovery

The preset system enables musicians and sound designers to:
1. Build complex presets programmatically
2. Save and share their creations
3. Build a personal sound library
4. Quickly iterate on existing designs
"""

from pathlib import Path

import numpy as np

from builder import PresetBuilder, PresetLibrary


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def example_1_basic_save_load():
    """Example 1: Basic preset save and load operations."""
    print_section("Example 1: Basic Preset Save & Load")

    # Create a simple sine wave preset
    print("\n1. Creating a simple sine wave preset...")
    preset = (
        PresetBuilder("Pure Sine")
        .set_description("Simple 440Hz sine wave")
        .sine(440, amplitude=0.8, gain_db=None)
        .volume(0.6)
    )

    print(preset.describe())

    # Save to a JSON file
    print("\n2. Saving preset to file...")
    preset_path = Path("temp_presets/pure_sine.json")
    preset.save_preset(preset_path)
    print(f"   ✓ Saved to: {preset_path}")

    # Load it back
    print("\n3. Loading preset from file...")
    loaded_preset = PresetBuilder.from_preset(preset_path)
    print("   ✓ Loaded successfully!")
    print(loaded_preset.describe())

    # Build and generate audio
    print("\n4. Generating audio from loaded preset...")
    audio = loaded_preset.build()
    samples = audio.get_samples(44100)  # 1 second
    print(f"   ✓ Generated {len(samples)} samples")


def example_2_library_management():
    """Example 2: Using PresetLibrary for organized storage."""
    print_section("Example 2: Preset Library Management")

    # Initialize library
    library = PresetLibrary("demo_presets")
    print("\n1. Initialized preset library at: demo_presets/")

    # Create and save multiple presets in different categories
    print("\n2. Creating and saving categorized presets...")

    # Bass sounds
    bass_preset = (
        PresetBuilder("Deep Bass")
        .set_description("Sub bass for electronic music")
        .sine(55, amplitude=0.9, gain_db=None)  # A1
        .adsr(0.01, 0.1, 0.8, 0.2)
        .volume(0.7)
    )
    library.save(bass_preset, category="bass")
    print("   ✓ Saved: Deep Bass (bass category)")

    # Lead sounds
    lead_preset = (
        PresetBuilder("Bright Lead")
        .set_description("Cutting lead sound")
        .sawtooth(880, amplitude=0.8, gain_db=None)  # A5
        .adsr(0.05, 0.2, 0.6, 0.3)
        .volume(0.6)
        .volume((-0.8, 0.8))
    )
    library.save(lead_preset, category="leads")
    print("   ✓ Saved: Bright Lead (leads category)")

    # Pad sounds
    pad_preset = (
        PresetBuilder("Warm Pad")
        .set_description("Atmospheric pad with slow attack")
        .sine(220, amplitude=0.6, gain_db=None)  # A3
        .adsr(2.0, 1.5, 0.7, 3.0)
        .volume(0.4)
        .panner(0.0)
    )
    library.save(pad_preset, category="pads")
    print("   ✓ Saved: Warm Pad (pads category)")

    # FX sounds
    fx_preset = (
        PresetBuilder("Sweep FX")
        .set_description("Rising sweep effect")
        .square(100, amplitude=0.7, gain_db=None)
        .volume(0.5)
        .panner(-0.3)
    )
    library.save(fx_preset, category="fx")
    print("   ✓ Saved: Sweep FX (fx category)")

    # List categories
    print("\n3. Available categories:")
    categories = library.get_categories()
    for cat in categories:
        print(f"   - {cat}")

    # List all presets
    print("\n4. All presets in library:")
    all_presets = library.list_presets()
    for preset in all_presets:
        print(f"   - {preset}")

    # List presets by category
    print("\n5. Presets by category:")
    for category in categories:
        category_presets = library.list_presets(category=category)
        print(f"\n   {category.upper()}:")
        for preset in category_presets:
            print(f"     - {preset}")


def example_3_loading_and_playing():
    """Example 3: Load and play presets from library."""
    print_section("Example 3: Loading & Playing Presets")

    library = PresetLibrary("demo_presets")

    # Load a preset
    print("\n1. Loading 'leads/Bright_Lead' preset...")
    preset = library.load("leads/Bright_Lead")
    print(preset.describe())

    # Build and generate audio
    print("\n2. Building and generating audio...")
    audio = preset.build()
    samples = audio.get_samples(int(44100 * 1.5))  # 1.5 seconds
    print(f"   ✓ Generated {len(samples)} samples")

    # Optionally play it
    print("\n3. Playing audio...")
    print("   (Uncomment play_wave line to hear the sound)")
    # play_wave(samples)


def example_4_modifying_presets():
    """Example 4: Load, modify, and save as new preset."""
    print_section("Example 4: Modifying Existing Presets")

    library = PresetLibrary("demo_presets")

    # Load existing preset
    print("\n1. Loading 'bass/Deep_Bass' preset...")
    original = library.load("bass/Deep_Bass")
    print("Original:")
    print(original.describe())

    # Clone and modify
    print("\n2. Cloning and modifying...")
    modified = original.clone()
    modified.set_name("Higher Bass")
    modified.set_description("Deep bass transposed up an octave")
    modified.modify_frequency(110)  # A2, one octave higher
    modified.modify_amplitude(0.85)

    print("\nModified:")
    print(modified.describe())

    # Save as new preset
    print("\n3. Saving modified version as new preset...")
    library.save(modified, category="bass")
    print("   ✓ Saved as: bass/Higher_Bass")

    # Verify both exist
    print("\n4. Bass presets now available:")
    bass_presets = library.list_presets(category="bass")
    for preset in bass_presets:
        print(f"   - {preset}")


def example_5_preset_inspection():
    """Example 5: Inspecting preset details."""
    print_section("Example 5: preset Inspection & Analysis")

    # Create a complex preset
    print("\n1. Creating a complex multi-component preset...")
    preset = (
        PresetBuilder("Complex Synth")
        .set_description("Multi-stage synthesis example")
        .sawtooth(440, amplitude=0.7, gain_db=None)
        .adsr(0.1, 0.3, 0.6, 0.5)
        .volume(0.6)
        .panner(0.2)
        .volume((-0.7, 0.7))
    )

    # Get full description
    print("\n2. Full preset description:")
    print(preset.describe())

    # Get summary statistics
    print("\n3. preset summary:")
    summary = preset.summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    # Access individual components
    print("\n4. Component access:")
    source = preset.get_source()
    print(f"   Source type: {type(source).__name__}")

    modifiers = preset.get_modifiers()
    print(f"   Number of modifiers: {len(modifiers)}")
    for i, mod in enumerate(modifiers, 1):
        print(f"     {i}. {type(mod).__name__}")

    modulators = preset.get_modulators()
    print(f"   Number of modulators: {len(modulators)}")
    for name, mod in modulators.items():
        print(f"     {name}: {type(mod).__name__}")

    # Get full component dictionary
    print("\n5. All components:")
    components = preset.get_components()
    print(f"   Name: {components['name']}")
    print(f"   Description: {components['description']}")
    print(f"   Sample rate: {components['sample_rate']} Hz")


def example_6_metadata_and_organization():
    """Example 6: Adding metadata to presets."""
    print_section("Example 6: Metadata & Organization")

    library = PresetLibrary("demo_presets")

    # Create a preset with rich metadata
    print("\n1. Creating preset with metadata...")
    preset = (
        PresetBuilder("Vintage Keys")
        .set_description("Electric piano sound inspired by 1970s Rhodes")
        .sine(440, amplitude=0.75, gain_db=None)
        .adsr(0.02, 0.5, 0.3, 0.8)
        .volume(0.65)
        .volume((-0.85, 0.85))
    )

    # Save with additional metadata
    metadata = {
        "author": "AudioPlayground Demo",
        "tags": ["electric piano", "vintage", "keys", "rhodes"],
        "genre": "jazz",
        "bpm": 120,
        "created": "2025-01-02",
        "version": "1.0",
    }

    library.save(preset, category="keys", metadata=metadata)
    print("   ✓ Saved with metadata:")
    for key, value in metadata.items():
        print(f"     {key}: {value}")

    # Load and verify metadata is preserved
    print("\n2. Loading preset and checking config...")
    loaded = library.load("keys/Vintage_Keys")
    config = loaded.get_config()

    if "metadata" in config:
        print("   ✓ Metadata preserved:")
        for key, value in config["metadata"].items():
            print(f"     {key}: {value}")


def example_7_batch_operations():
    """Example 7: Batch operations on presets."""
    print_section("Example 7: Batch Operations")

    library = PresetLibrary("demo_presets")

    # Create multiple variations programmatically
    print("\n1. Creating harmonic series presets...")
    base_freq = 110  # A2

    for i in range(1, 6):
        freq = base_freq * i
        name = f"Harmonic_{i}"

        preset = (
            PresetBuilder(name)
            .set_description(f"Harmonic {i} of {base_freq}Hz = {freq}Hz")
            .sine(freq, amplitude=0.8, gain_db=None)
            .adsr(0.05, 0.2, 0.7, 0.3)
            .volume(0.5)
        )

        library.save(preset, category="harmonics")
        print(f"   ✓ Created: {name} ({freq}Hz)")

    # List all harmonics
    print("\n2. Harmonic series presets:")
    harmonic_presets = library.list_presets(category="harmonics")
    for preset in harmonic_presets:
        print(f"   - {preset}")

    # Create variations with different waveforms
    print("\n3. Creating waveform variations...")
    waveforms = {
        "sine": lambda b, f: b.sine(f),
        "square": lambda b, f: b.square(f),
        "triangle": lambda b, f: b.triangle(f),
        "sawtooth": lambda b, f: b.sawtooth(f),
    }

    freq = 440
    for wave_name, wave_func in waveforms.items():
        name = f"{wave_name.capitalize()}_440"
        preset = PresetBuilder(name)
        wave_func(preset, freq)
        preset.set_description(f"{wave_name.capitalize()} wave at {freq}Hz")
        preset.volume(0.6)

        library.save(preset, category="waveforms")
        print(f"   ✓ Created: {name}")


def example_8_advanced_presetes():
    """Example 8: Creating and saving advanced multi-oscillator presetes."""
    print_section("Example 8: Advanced Multi-Oscillator presetes")

    library = PresetLibrary("demo_presets")

    # Create a layered preset (this would require mixing support)
    print("\n1. Creating a detuned unison preset...")

    # Main oscillator at 440Hz
    preset = (
        PresetBuilder("Detuned Unison")
        .set_description("Three slightly detuned oscillators for richness")
        .sine(440, amplitude=0.5, gain_db=None)  # Center
        .adsr(0.1, 0.3, 0.7, 0.4)
        .volume(0.6)
    )

    print(preset.describe())
    library.save(preset, category="synths")
    print("   ✓ Saved: Detuned Unison")

    # Create a bass preset with harmonics
    print("\n2. Creating harmonic-rich bass...")
    bass = (
        PresetBuilder("Harmonic Bass")
        .set_description("Fundamental with added harmonics")
        .sawtooth(55, amplitude=0.8, gain_db=None)  # Rich in harmonics
        .adsr(0.01, 0.15, 0.7, 0.25)
        .volume((-0.9, 0.9))
        .volume(0.65)
    )

    print(bass.describe())
    library.save(bass, category="bass")
    print("   ✓ Saved: Harmonic Bass")


def example_9_preset_discovery():
    """Example 9: Browsing and discovering presets."""
    print_section("Example 9: Preset Discovery & Browsing")

    library = PresetLibrary("demo_presets")

    # Get all categories
    categories = library.get_categories()

    print("\n📁 PRESET LIBRARY OVERVIEW")
    print(f"\nTotal categories: {len(categories)}\n")

    # For each category, show presets with details
    for category in categories:
        presets = library.list_presets(category=category)
        print(f"┌─ {category.upper()} ({len(presets)} presets)")

        for preset_name in presets:
            # Load and get summary
            try:
                preset = library.load(preset_name)
                summary = preset.summary()
                desc = preset.get_description()

                # Truncate description if too long
                if len(desc) > 50:
                    desc = desc[:47] + "..."

                print(f"│  ├─ {summary['name']}")
                if desc:
                    print(f"│  │  └─ {desc}")
                print(
                    f"│  │     Components: {summary['components']} "
                    f"(Osc:{summary['oscillators']}, "
                    f"Mod:{summary['modulators']}, "
                    f"FX:{summary['effects']})"
                )
            except Exception as e:
                print(f"│  ├─ {preset_name} (Error loading: {e})")

        print("│")
    print("└─────────────────────────────────────────")


def example_10_export_import():
    """Example 10: Exporting and importing preset collections."""
    print_section("Example 10: Export & Import Presets")

    library = PresetLibrary("demo_presets")

    # Create a special preset for export
    print("\n1. Creating a preset for export...")
    export_preset = (
        PresetBuilder("Exported Sound")
        .set_description("A preset designed to be shared")
        .triangle(330, amplitude=0.75, gain_db=None)
        .adsr(0.08, 0.25, 0.65, 0.35)
        .volume(0.6)
        .panner(0.1)
    )

    # Save to library
    library.save(export_preset, category="export")
    print("   ✓ Saved to library")

    # The preset is already a JSON file that can be shared
    export_path = Path("demo_presets/export/Exported_Sound.json")
    print("\n2. Preset saved as shareable JSON file:")
    print(f"   {export_path}")

    # Show the JSON content
    if export_path.exists():
        import json

        with open(export_path, "r") as f:
            config = json.load(f)

        print("\n3. JSON structure:")
        print(f"   Name: {config.get('name')}")
        print(f"   Description: {config.get('description')}")
        print(f"   Version: {config.get('version')}")
        print(f"   Components: {len(config.get('components', []))}")

        print("\n4. This file can be:")
        print("   - Shared with other users")
        print("   - Version controlled (git)")
        print("   - Loaded with presetBuilder.from_preset()")
        print("   - Imported into another PresetLibrary")


def example_11_compare_presets():
    """Example 11: Compare different presets."""
    print_section("Example 11: Comparing Presets")

    library = PresetLibrary("demo_presets")

    # Load multiple presets
    preset_names = [
        "waveforms/Sine_440",
        "waveforms/Square_440",
        "waveforms/Triangle_440",
        "waveforms/Sawtooth_440",
    ]

    print("\n📊 WAVEFORM COMPARISON\n")

    presetes = []
    for name in preset_names:
        try:
            preset = library.load(name)
            presetes.append(preset)
            summary = preset.summary()

            print(
                f"{summary['name']:20} | Components: {summary['components']:2} | "
                f"Osc: {summary['oscillators']:2} | "
                f"Mod: {summary['modulators']:2} | "
                f"FX: {summary['effects']:2}"
            )
        except Exception as e:
            print(f"{name:20} | Error: {e}")

    # Generate and compare audio characteristics
    print("\n\n🎵 AUDIO CHARACTERISTICS\n")

    for preset in presetes:
        audio = preset.build()
        samples = audio.get_samples(4410)  # 0.1 second at 44100 Hz

        # Calculate basic characteristics
        rms = np.sqrt(np.mean(samples**2))
        peak = np.max(np.abs(samples))
        crest_factor = peak / rms if rms > 0 else 0

        print(
            f"{preset.get_name():20} | "
            f"RMS: {rms:.4f} | "
            f"Peak: {peak:.4f} | "
            f"Crest: {crest_factor:.2f}"
        )


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  AUDIOPLAYGROUND PRESET SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("\nThis demo showcases the complete preset system functionality.")
    print("Examples will create temporary presets in 'demo_presets/' directory.")

    input("\nPress Enter to start the demo...")

    # Run all examples
    example_1_basic_save_load()
    input("\nPress Enter to continue to next example...")

    example_2_library_management()
    input("\nPress Enter to continue to next example...")

    example_3_loading_and_playing()
    input("\nPress Enter to continue to next example...")

    example_4_modifying_presets()
    input("\nPress Enter to continue to next example...")

    example_5_preset_inspection()
    input("\nPress Enter to continue to next example...")

    example_6_metadata_and_organization()
    input("\nPress Enter to continue to next example...")

    example_7_batch_operations()
    input("\nPress Enter to continue to next example...")

    example_8_advanced_presetes()
    input("\nPress Enter to continue to next example...")

    example_9_preset_discovery()
    input("\nPress Enter to continue to next example...")

    example_10_export_import()
    input("\nPress Enter to continue to next example...")

    example_11_compare_presets()

    # Final summary
    print_section("Demo Complete!")
    print("\n✅ All examples completed successfully!")
    print("\nThe preset system enables:")
    print("  • Saving and loading presetes")
    print("  • Organizing presets in categories")
    print("  • Managing preset libraries")
    print("  • Adding rich metadata")
    print("  • Batch creation and operations")
    print("  • Cloning and modifying existing presets")
    print("  • Exporting and sharing presets")
    print("  • Analyzing and comparing sounds")
    print("\nCheck the 'demo_presets/' directory to see all created presets!")
    print("Preset files are human-readable JSON and can be:")
    print("  • Edited manually")
    print("  • Version controlled")
    print("  • Shared with others")
    print("  • Loaded programmatically")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
