"""
Example: Using Audio Effects (Distortion, Delay, Reverb)

This example demonstrates how to use the audio effects with existing audio chains.
Effects can be easily chained together with oscillators and other components.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.engine import (
    SineOscillator,
    SquareOscillator,
    Chain,
    Distortion,
    Delay,
    Reverb,
    ADSREnvelope,
    ModulatedOscillator,
)

# Set up parameters
sample_rate = 44100
duration = 2.0  # seconds
n_samples = int(duration * sample_rate)

print("Audio Effects Examples\n")
print("=" * 70)

# Example 1: Simple Distortion
print("\nExample 1: Distortion Effect")
print("-" * 70)

osc = SineOscillator(frequency=220, gain_db=-6)
distorted = Distortion(
    source=osc,
    drive=3.0,  # Heavy distortion
    mix=0.8,  # 80% wet
    output_gain=0.5,  # Compensate for increased level
    distortion_type="soft",  # Soft clipping
)

samples = distorted.get_samples_vectorized(4410)  # 0.1 seconds
print(f"  Generated {len(samples)} distorted samples")
print(f"  Drive: {distorted.drive}")
print(f"  Mix: {distorted.mix}")
print(f"  Type: {distorted.distortion_type}")

# Example 2: Delay Effect
print("\nExample 2: Delay Effect")
print("-" * 70)

osc2 = SquareOscillator(frequency=440, gain_db=-12)
delayed = Delay(
    source=osc2,
    delay_time=0.3,  # 300ms delay
    feedback=0.5,  # Moderate feedback
    mix=0.5,  # Equal mix
    sample_rate=sample_rate,
)

samples = delayed.get_samples_vectorized(8820)  # 0.2 seconds
print(f"  Generated {len(samples)} delayed samples")
print(f"  Delay time: {delayed.delay_time}s")
print(f"  Feedback: {delayed.feedback}")
print(f"  Mix: {delayed.mix}")

# Example 3: Reverb Effect
print("\nExample 3: Reverb Effect")
print("-" * 70)

osc3 = SineOscillator(frequency=880, gain_db=-12)
reverbed = Reverb(
    source=osc3,
    room_size=0.7,  # Large room
    damping=0.4,  # Bright reverb
    mix=0.4,  # 40% wet
    sample_rate=sample_rate,
)

samples = reverbed.get_samples_vectorized(4410)
print(f"  Generated {len(samples)} reverbed samples")
print(f"  Room size: {reverbed.room_size}")
print(f"  Damping: {reverbed.damping}")
print(f"  Mix: {reverbed.mix}")

# Example 4: Chaining Multiple Effects
print("\nExample 4: Effect Chain (Distortion → Delay → Reverb)")
print("-" * 70)

# Create source
osc4 = SineOscillator(frequency=330, gain_db=-12)

# Chain effects together

# Note: Chain needs to handle the implicit source passing
# For now, let's use manual chaining
dist = Distortion(osc4, drive=2.0, mix=0.6, output_gain=0.6, distortion_type="tube")
delay = Delay(dist, delay_time=0.25, feedback=0.3, mix=0.3, sample_rate=sample_rate)
reverb = Reverb(delay, room_size=0.5, damping=0.5, mix=0.2, sample_rate=sample_rate)
chain = Chain(osc4, dist, delay, reverb)

samples = reverb.get_samples_vectorized(8820)
print(f"  Generated {len(samples)} samples through effect chain")
print("  Effects: Tube Distortion → Delay → Reverb")

# Example 5: Different Distortion Types
print("\nExample 5: Distortion Types Comparison")
print("-" * 70)

distortion_types = ["soft", "hard", "fuzz", "tube"]
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
fig.suptitle("Distortion Types Comparison", fontsize=16)

for i, dist_type in enumerate(distortion_types):
    osc = SineOscillator(frequency=220, gain_db=-6)
    dist = Distortion(
        source=osc, drive=4.0, mix=1.0, output_gain=0.5, distortion_type=dist_type
    )

    samples = dist.get_samples_vectorized(2205)  # 0.05 seconds
    time = np.arange(len(samples)) / sample_rate

    axes[i].plot(time * 1000, samples, linewidth=1)
    axes[i].set_title(f"{dist_type.capitalize()} Distortion")
    axes[i].set_ylabel("Amplitude")
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim(-1, 1)

axes[-1].set_xlabel("Time (ms)")
plt.tight_layout()
plt.savefig("distortion_types_comparison.png", dpi=150, bbox_inches="tight")
print("  ✓ Saved distortion comparison plot")

# Example 6: ADSR Envelope with Effects
print("\nExample 6: ADSR Envelope with Delay and Reverb")
print("-" * 70)

# Create ADSR envelope
env = ADSREnvelope(
    attack_duration=0.05, decay_duration=0.1, sustain_level=0.6, release_duration=0.3
)

# Create modulated oscillator
env.trigger_note_on()
osc = SineOscillator(frequency=440, amplitude=1.0, gain_db=-12)
mod_osc = ModulatedOscillator(
    osc, env, amp_mod=lambda base_amp, env_val: base_amp * env_val
)

# Add effects
delay = Delay(mod_osc, delay_time=0.2, feedback=0.4, mix=0.3, sample_rate=sample_rate)
reverb = Reverb(delay, room_size=0.6, damping=0.5, mix=0.3, sample_rate=sample_rate)

# Generate note
samples = reverb.get_samples_vectorized(44100)  # 1 second
mod_osc.trigger_release()  # Trigger release after generating

print(f"  Generated {len(samples)} samples")
print("  Effect chain: ADSR Envelope → Delay → Reverb")

# Plot the result
plt.figure(figsize=(12, 4))
time = np.arange(len(samples)) / sample_rate
plt.plot(time, samples, linewidth=0.5)
plt.title("ADSR Envelope with Delay and Reverb")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("adsr_with_effects.png", dpi=150, bbox_inches="tight")
print("  ✓ Saved ADSR with effects plot")

# Example 7: Real-time Parameter Changes
print("\nExample 7: Real-time Parameter Changes")
print("-" * 70)

osc = SineOscillator(frequency=440, gain_db=-12)
dist = Distortion(osc, drive=1.0, mix=1.0, output_gain=0.5)

# Generate with different drive amounts
all_samples = []
drive_values = [0.5, 1.0, 2.0, 4.0, 8.0]

for drive in drive_values:
    dist.drive = drive  # Real-time parameter change
    samples = dist.get_samples_vectorized(2205)  # 50ms per setting
    all_samples.extend(samples)
    print(f"  Drive = {drive}: Generated {len(samples)} samples")

print(f"  Total samples: {len(all_samples)}")

# Example 8: Delay Feedback Loop
print("\nExample 8: Delay Feedback Exploration")
print("-" * 70)

osc = SineOscillator(frequency=330, gain_db=-12)
feedback_values = [0.0, 0.3, 0.6, 0.9]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Delay Feedback Comparison", fontsize=16)

for i, fb in enumerate(feedback_values):
    row, col = i // 2, i % 2

    osc_fresh = SineOscillator(frequency=330, gain_db=-12)
    delay = Delay(
        osc_fresh, delay_time=0.15, feedback=fb, mix=0.7, sample_rate=sample_rate
    )

    samples = delay.get_samples_vectorized(8820)  # 0.2 seconds
    time = np.arange(len(samples)) / sample_rate * 1000

    axes[row, col].plot(time, samples, linewidth=0.8)
    axes[row, col].set_title(f"Feedback = {fb}")
    axes[row, col].set_ylabel("Amplitude")
    axes[row, col].grid(True, alpha=0.3)

    if row == 1:
        axes[row, col].set_xlabel("Time (ms)")

plt.tight_layout()
plt.savefig("delay_feedback_comparison.png", dpi=150, bbox_inches="tight")
print("  ✓ Saved delay feedback comparison plot")

print("\n" + "=" * 70)
print("✓ All examples completed successfully!")
print("=" * 70)

print("\nUsage Tips:")
print("  • Distortion: Use 'soft' for warm overdrive, 'hard' for aggressive clipping")
print("  • Delay: Keep feedback < 0.95 to avoid infinite buildup")
print("  • Reverb: Lower damping = brighter reverb, higher = darker")
print("  • Chain effects in order: Distortion → Delay → Reverb")
print("  • Use mix parameter to blend dry/wet signals")
print("  • Adjust output_gain on distortion to compensate for level increase")

print("\nEffect Parameters:")
print("  Distortion:")
print("    - drive: 0.0-10.0 (distortion amount)")
print("    - mix: 0.0-1.0 (dry/wet)")
print("    - output_gain: 0.0-2.0 (output level)")
print("    - distortion_type: 'soft', 'hard', 'fuzz', 'tube'")
print("  Delay:")
print("    - delay_time: 0.001-2.0 seconds")
print("    - feedback: 0.0-0.95")
print("    - mix: 0.0-1.0")
print("  Reverb:")
print("    - room_size: 0.0-1.0")
print("    - damping: 0.0-1.0")
print("    - mix: 0.0-1.0")
