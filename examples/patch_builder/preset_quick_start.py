"""Quick start guide for the preset system.

This is a simplified introduction to the preset system showing the most
common use cases.
"""

from builder import PatchBuilder, PatchLibrary


def main():
    print("AudioPlayground Preset System - Quick Start")
    print("=" * 60)

    # ============================================================
    # Part 1: Creating and Saving a Preset
    # ============================================================
    print("\n1. CREATE A PATCH")
    print("-" * 60)

    # Build a patch using the fluent API
    my_patch = (
        PatchBuilder("My First Synth")
        .set_description("A simple lead sound")
        .sine(440, amplitude=0.8)
        .adsr(0.1, 0.2, 0.7, 0.3)
        .volume(0.6)
        .pan(0.0)
    )

    print(my_patch.describe())

    # ============================================================
    # Part 2: Save to File
    # ============================================================
    print("\n2. SAVE TO FILE")
    print("-" * 60)

    # Method 1: Save directly to a file
    my_patch.save_preset("my_first_synth.json")
    print("✓ Saved to: my_first_synth.json")

    # ============================================================
    # Part 3: Load from File
    # ============================================================
    print("\n3. LOAD FROM FILE")
    print("-" * 60)

    # Load the preset
    loaded = PatchBuilder.from_preset("my_first_synth.json")
    print("✓ Loaded successfully!")

    # Build and use it
    audio = loaded.build()
    samples = audio.get_samples(44100)  # 1 second
    print(f"✓ Generated {len(samples)} samples")

    # ============================================================
    # Part 4: Using a Preset Library
    # ============================================================
    print("\n4. USING A PRESET LIBRARY")
    print("-" * 60)

    # Create a library
    library = PatchLibrary("my_presets")
    print("✓ Created library in 'my_presets/' directory")

    # Save patches to the library with categories
    bass = (
        PatchBuilder("Deep Bass")
        .set_description("Sub bass sound")
        .sine(55, amplitude=0.9)
        .adsr(0.01, 0.1, 0.8, 0.2)
        .volume(0.7)
    )

    library.save(bass, category="bass")
    print("✓ Saved bass preset")

    lead = (
        PatchBuilder("Bright Lead")
        .set_description("Lead synth")
        .sawtooth(880, amplitude=0.8)
        .adsr(0.05, 0.2, 0.6, 0.3)
        .volume(0.6)
    )

    library.save(lead, category="leads")
    print("✓ Saved lead preset")

    # ============================================================
    # Part 5: Browse Library
    # ============================================================
    print("\n5. BROWSE LIBRARY")
    print("-" * 60)

    # List categories
    categories = library.get_categories()
    print(f"Categories: {categories}")

    # List all presets
    all_presets = library.list_presets()
    print("\nAll presets:")
    for preset in all_presets:
        print(f"  - {preset}")

    # ============================================================
    # Part 6: Load from Library
    # ============================================================
    print("\n6. LOAD FROM LIBRARY")
    print("-" * 60)

    # Load a preset from the library
    loaded_bass = library.load("bass/Deep_Bass")
    print(loaded_bass.describe())

    # ============================================================
    # Part 7: Modify and Save
    # ============================================================
    print("\n7. MODIFY AND SAVE AS NEW")
    print("-" * 60)

    # Clone an existing patch
    modified = loaded_bass.clone()

    # Modify it
    modified.set_name("Mid Bass")
    modified.set_description("Bass transposed up")
    modified.modify_frequency(110)  # One octave higher

    # Save as new preset
    library.save(modified, category="bass")
    print("✓ Saved modified version")

    # ============================================================
    # Part 8: Inspect Patches
    # ============================================================
    print("\n8. INSPECT PATCH DETAILS")
    print("-" * 60)

    # Get summary
    summary = modified.summary()
    print(f"Patch: {summary['name']}")
    print(f"  Oscillators: {summary['oscillators']}")
    print(f"  Modulators: {summary['modulators']}")
    print(f"  Effects: {summary['effects']}")
    print(f"  Total components: {summary['components']}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("QUICK START COMPLETE!")
    print("=" * 60)
    print("\nYou now know how to:")
    print("  ✓ Create patches with PatchBuilder")
    print("  ✓ Save patches to files")
    print("  ✓ Load patches from files")
    print("  ✓ Organize presets in a library")
    print("  ✓ Browse and load from library")
    print("  ✓ Clone and modify existing patches")
    print("  ✓ Inspect patch details")
    print("\nCheck these directories for your presets:")
    print("  - my_first_synth.json")
    print("  - my_presets/")
    print("\nFor more advanced examples, see: preset_system_demo.py")


if __name__ == "__main__":
    main()
