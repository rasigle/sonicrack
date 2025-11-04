"""Demonstration of the Engine Component Registry System.

This example shows how the registry system makes it easy to:
1. Use the RegistryPatchBuilder (drop-in replacement for PatchBuilder)
2. Register custom components without modifying core code
3. Create plugins that extend the system
4. Maintain backward compatibility with presets
"""

import numpy as np

from builder import PresetBuilder, ComponentCategory
from constants import DEFAULT_SAMPLE_RATE
from engine import (
    Oscillator,
    Modifier,
    audio_registry,
    ComponentDescriptor,
)


def example_1_basic_usage():
    """Example 1: Basic usage - identical to PatchBuilder."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)

    # Use exactly like the original PatchBuilder
    patch = (
        PresetBuilder("My Synth")
        .set_description("A simple lead sound")
        .sine(440, amplitude=0.8)
        .adsr(0.1, 0.2, 0.7, 0.3)
        .volume(0.6)
        .panner(0.2)
    )

    print("\nPatch description:")
    print(patch.describe())

    print("\nPatch summary:")
    summary = patch.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Build and generate audio
    audio = patch.build()
    samples = audio.get_samples(44100)
    print(f"\n✓ Generated {len(samples)} samples")

    # Save preset
    patch.save_preset("temp_registry_test.json")
    print("✓ Saved preset")

    # Load preset
    loaded = PresetBuilder.from_preset("temp_registry_test.json")
    print("✓ Loaded preset")
    print(f"  Loaded patch name: {loaded.get_name()}")


def example_2_custom_oscillator():
    """Example 2: Register a custom oscillator."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Oscillator Registration")
    print("=" * 70)

    # Define a custom noise oscillator
    class NoiseOscillator(Oscillator):
        """White noise generator."""

        descriptor = ComponentDescriptor(
            name="MyCustomNoiseOscillator",
            category=ComponentCategory.OSCILLATOR,
            config_params=["amplitude", "sample_rate"],
            description="White noise generator",
            fluent_api_name="my_custom_noise",
        )

        def __init__(
            self, amplitude: float = 1.0, sample_rate: int = DEFAULT_SAMPLE_RATE
        ):
            super().__init__(
                frequency=0, amplitude=amplitude, phase=0, sample_rate=sample_rate
            )

        def __next__(self):
            """Return next white noise sample.

            Returns:
                float: Random value scaled by amplitude.
            """
            return self._a * (2 * np.random.random() - 1)

        def get_samples_vectorized(self, n: int) -> np.ndarray:
            """Generate n samples of white noise."""
            return self._a * (2 * np.random.random(n) - 1)

    # Register it with the system
    audio_registry.register(NoiseOscillator)
    print("   ✓ Registered")

    # Now use it immediately!
    print("\n2. Using the new oscillator...")
    patch = (
        PresetBuilder("Noise Patch")
        .my_custom_noise(amplitude=0.5)  # Method auto-generated!
        .volume(0.3)
    )

    print(patch.describe())

    audio = patch.build()
    samples = audio.get_samples(1000)
    print(f"   ✓ Generated {len(samples)} noise samples")
    print(f"   ✓ Sample RMS: {np.sqrt(np.mean(samples**2)):.3f}")


def example_3_custom_effect():
    """Example 3: Register a custom effect/modifier."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Custom Effect Registration")
    print("=" * 70)

    # Define a custom distortion effect
    class DistortionEffect(Modifier):
        """Simple distortion/overdrive effect."""

        descriptor = ComponentDescriptor(
            name="Distortion",
            category=ComponentCategory.MODIFIER,
            config_params=["drive", "mix"],
            description="Soft clipping distortion/overdrive",
            fluent_api_name="distortion",
        )

        def __init__(self, drive: float = 2.0, mix: float = 1.0):
            self.drive = drive
            self.mix = mix

        def __call__(self, val):
            """Apply distortion to value."""
            if isinstance(val, tuple):
                # Stereo
                return tuple(self._distort(v) for v in val)
            else:
                # Mono
                return self._distort(val)

        def _distort(self, x):
            """Soft clipping distortion."""
            driven = x * self.drive
            # Soft clip using tanh
            distorted = np.tanh(driven)
            # Mix dry/wet
            return x * (1 - self.mix) + distorted * self.mix

    # Register it
    print("\n1. Registering custom 'distortion' effect...")
    audio_registry.register(DistortionEffect)
    print("   ✓ Registered")

    # Use it in a patch
    print("\n2. Creating patch with distortion...")
    patch = (
        PresetBuilder("Distorted Lead")
        .sawtooth(440, amplitude=0.9)
        .distortion(drive=3.0, mix=0.7)  # Auto-generated method!
        .volume(0.6)
    )

    print(patch.describe())

    audio = patch.build()
    samples = audio.get_samples(1000)
    print(f"   ✓ Generated {len(samples)} samples with distortion")
    print(f"   ✓ Peak value: {np.max(np.abs(samples)):.3f}")


def example_4_list_components():
    """Example 4: Inspect registered components."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Component Registry Inspection")
    print("=" * 70)

    # List all registered components
    print("\n1. All registered components:")
    for comp in sorted(audio_registry.list_components()):
        descriptor = audio_registry.get(comp, strict=True).descriptor
        print(
            f"   - {comp:30} ({descriptor.category.value:12}) -> .{descriptor.name}()"
        )

    # List by category
    print("\n2. Components by category:")
    for category in ComponentCategory:
        components = audio_registry.list_by_category(category)
        print(f"\n   {category.value.upper()}:")
        for component in components:
            desc = audio_registry.get(component, strict=True).descriptor
            print(f"     - {desc.name} ({desc.description})")
            params = ", ".join(desc.config_params) if desc.config_params else []
            if params:
                print(f"       Parameter: {params}")


def example_5_preset_compatibility():
    """Example 5: Presets work with custom components."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Preset System Integration")
    print("=" * 70)

    # Create patch with custom component (noise from example 2)
    print("\n1. Creating patch with custom components...")
    patch = (
        PresetBuilder("Custom Patch")
        .set_description("Uses custom registered components")
        .my_custom_noise(amplitude=0.6)
        .volume(0.5)
        .panner(-0.3)
    )

    # Save it
    preset_file = "custom_component_preset.json"
    patch.save_preset(preset_file)
    print(f"   ✓ Saved to {preset_file}")

    # Load it back
    print("\n2. Loading preset...")
    loaded = PresetBuilder.from_preset(preset_file)
    print(f"   ✓ Loaded: {loaded.get_name()}")
    print(f"   ✓ Description: {loaded.get_description()}")

    # Build and use
    audio = loaded.build()
    samples = audio.get_samples(500)
    print(f"   ✓ Generated {len(samples)} samples from loaded preset")

    # Show the JSON
    print("\n3. Preset JSON content:")
    import json

    with open(preset_file, "r") as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))


def example_6_plugin_system():
    """Example 6: Simulating a plugin system."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Plugin System Simulation")
    print("=" * 70)

    print("\nImagine these components are in a separate plugin module...")

    # Plugin 1: Supersaw oscillator
    class SuperSawOscillator(Oscillator):
        """Supersaw with multiple detuned voices."""

        descriptor = ComponentDescriptor(
            name="SuperSaw",
            category=ComponentCategory.OSCILLATOR,
            config_params=["frequency", "voices", "detune", "sample_rate"],
            description="Supersaw oscillator with multiple detuned voices",
            fluent_api_name="supersaw",
        )

        def __init__(
            self,
            frequency: float,
            voices: int = 7,
            detune: float = 0.1,
            sample_rate: int = DEFAULT_SAMPLE_RATE,
        ):
            super().__init__(
                frequency=frequency, amplitude=1.0, phase=0, sample_rate=sample_rate
            )
            self.voices = voices
            self.detune = detune

        def __iter__(self):
            # Simplified - just return base frequency
            phase = 0
            while True:
                yield np.sin(2 * np.pi * phase)
                phase = (phase + self.frequency / self.sample_rate) % 1.0

        def get_samples_vectorized(self, n: int) -> np.ndarray:
            # Mix multiple detuned voices
            result = np.zeros(n)
            for i in range(self.voices):
                detune_factor = 1.0 + self.detune * (i - self.voices // 2) / self.voices
                freq = self.frequency * detune_factor
                t = np.arange(n) / self.sample_rate
                result += np.sin(2 * np.pi * freq * t)
            return result / self.voices

    # Plugin 2: Resonant filter (simplified)
    class ResonantFilter(Modifier):
        """Simple resonant low-pass filter."""

        descriptor = ComponentDescriptor(
            name="ResonantFilter",
            category=ComponentCategory.MODIFIER,
            config_params=["cutoff", "resonance"],
            description="Resonant low-pass filter",
            fluent_api_name="resonant_filter",
        )

        def __init__(self, cutoff: float = 1000, resonance: float = 0.5):
            self.cutoff = cutoff
            self.resonance = resonance

        def __call__(self, val):
            # Simplified - just attenuate high frequencies
            return val * 0.8

    # Register plugin components
    print("\n1. Registering plugin components...")

    audio_registry.register(SuperSawOscillator)
    print("   ✓ Registered supersaw oscillator")

    audio_registry.register(ResonantFilter)
    print("   ✓ Registered resonant filter")

    # Now use the plugin components!
    print("\n2. Using plugin components...")
    patch = (
        PresetBuilder("Plugin Demo")
        .supersaw(440, voices=9, detune=0.15)
        .resonant_filter(cutoff=2000, resonance=0.7)
        .volume(0.6)
    )

    print(patch.describe())

    audio = patch.build()
    samples = audio.get_samples(1000)
    print(f"   ✓ Generated {len(samples)} samples using plugin components")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  COMPONENT REGISTRY SYSTEM - DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows how the registry system enables extensibility")
    print("without modifying the core PatchBuilder code.\n")

    # Run examples
    example_1_basic_usage()

    input("\nPress Enter to continue to Example 2...")
    example_2_custom_oscillator()

    input("\nPress Enter to continue to Example 3...")
    example_3_custom_effect()

    input("\nPress Enter to continue to Example 4...")
    example_4_list_components()

    input("\nPress Enter to continue to Example 5...")
    example_5_preset_compatibility()

    input("\nPress Enter to continue to Example 6...")
    example_6_plugin_system()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("\n✅ RegistryPatchBuilder is a drop-in replacement")
    print("✅ New components can be registered without code changes")
    print("✅ Custom components work with presets")
    print("✅ Plugin system is straightforward")
    print("✅ Registry is inspectable and self-documenting")
    print("\nThe registry system provides maximum maintainability:")
    print("  • Add components in ONE place (registration)")
    print("  • Methods are auto-generated")
    print("  • Serialization is automatic")
    print("  • No need to modify PatchBuilder ever again!")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
