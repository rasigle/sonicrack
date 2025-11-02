"""Example demonstrating PatchBuilder convenience methods.

This script shows how to use the new convenience methods for patch
inspection and modification.
"""

from builder import PatchBuilder
from utils import play_wave

# Create a patch using fluent API
print("Creating a patch...")
patch = (PatchBuilder()
        .sine(1000, amplitude=0.8)
        .adsr(0.1, 0.2, 0.7, 0.3)
        .volume(0.5)
        .pan(0.3)
        .clip(-0.9, 0.9))

# Display patch description
print("\n" + "="*60)
print(patch.describe())
print("="*60)

# Get patch summary
summary = patch.summary()
print("\nPatch Summary:")
print(f"  Oscillators: {summary['oscillators']}")
print(f"  Modulators: {summary['modulators']}")
print(f"  Effects: {summary['effects']}")
print(f"  Total Components: {summary['components']}")
print(f"  Sample Rate: {summary['sample_rate']} Hz")

# Modify patch parameters
print("\nModifying patch...")
patch.modify_frequency(880)  # Change to A5
patch.modify_amplitude(0.6)

print("\nAfter modifications:")
print(patch.describe())

# Clone the patch
print("\nCloning patch...")
clone = patch.clone()
clone.modify_frequency(1760)  # Change clone to A6

print("\nOriginal patch:")
print(patch.describe())
print("\nCloned patch (modified):")
print(clone.describe())

# Clear effects
print("\nClearing effects from clone...")
clone.clear_effects()
print(clone.describe())

# Build and generate audio
print("\nBuilding patches and generating audio...")
original_audio = patch.build()
clone_audio = clone.build()

original_samples = original_audio.get_samples(44100)
clone_samples = clone_audio.get_samples(44100*5)

play_wave(clone_samples)

print(f"\nOriginal patch generated {len(original_samples)} samples")
print(f"Clone patch generated {len(clone_samples)} samples")
