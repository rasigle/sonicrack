"""Example demonstrating presetBuilder convenience methods.

This script shows how to use the new convenience methods for preset
inspection and modification.
"""

from engine.presets import PresetBuilder
from utils import play_wave

# Create a preset using fluent API
print("Creating a preset...")
preset = (
    PresetBuilder()
    .sine(1000, amplitude=0.8)
    .adsr(0.1, 0.2, 0.7, 0.3)
    .volume(0.5)
    .panner(0.3)
    .clipper((-0.9, 0.9))
)

# Display preset description
print("\n" + "=" * 60)
print(preset.describe())
print("=" * 60)

# Get preset summary
summary = preset.summary()
print("\npreset Summary:")
print(f"  Oscillators: {summary['oscillators']}")
print(f"  Modulators: {summary['modulators']}")
print(f"  Effects: {summary['effects']}")
print(f"  Total Components: {summary['components']}")
print(f"  Sample Rate: {summary['sample_rate']} Hz")

# Modify preset parameters
print("\nModifying preset...")
preset.modify_frequency(880)  # Change to A5
preset.modify_amplitude(0.6)

print("\nAfter modifications:")
print(preset.describe())

# Clone the preset
print("\nCloning preset...")
clone = preset.clone()
clone.modify_frequency(1760)  # Change clone to A6

print("\nOriginal preset:")
print(preset.describe())
print("\nCloned preset (modified):")
print(clone.describe())

# Clear effects
print("\nClearing effects from clone...")
clone.clear_effects()
print(clone.describe())

# Build and generate audio
print("\nBuilding presetes and generating audio...")
original_audio = preset.build()
clone_audio = clone.build()

original_samples = original_audio.get_samples(44100)
clone_samples = clone_audio.get_samples(44100 * 5)

play_wave(clone_samples)

print(f"\nOriginal preset generated {len(original_samples)} samples")
print(f"Clone preset generated {len(clone_samples)} samples")
