"""Demonstration of improved Panner and ModulatedPanner with constant-power law.

This example demonstrates:
1. Constant-power panning law
2. Vectorized panning operations
3. Modulated panning with ADSR envelope
4. Power preservation across pan positions
"""

import matplotlib.pyplot as plt
import numpy as np

from engine import Panner, ModulatedPanner, ADSREnvelope, SineOscillator, Chain


def demonstrate_constant_power_panning():
    """Demonstrate constant-power panning law."""
    print("=" * 70)
    print("Constant-Power Panning Demonstration")
    print("=" * 70)

    # Test different pan positions
    positions = np.linspace(-1.0, 1.0, 21)

    left_gains = []
    right_gains = []
    total_powers = []

    print("\nPan Position | Left Gain | Right Gain | Total Power")
    print("-" * 60)

    for pos in positions:
        panner = Panner(pos)
        left, right = panner(1.0)
        power = left**2 + right**2

        left_gains.append(left)
        right_gains.append(right)
        total_powers.append(power)

        print(f"{pos:12.2f} | {left:9.4f} | {right:10.4f} | {power:11.6f}")

    # Verify constant power
    print(f"\n✓ Power variation: {np.std(total_powers):.10f}")
    print(f"✓ All powers ≈ 1.0: {np.allclose(total_powers, 1.0)}")

    # Plot the panning law
    plt.figure(figsize=(10, 6))
    plt.plot(positions, left_gains, 'b-', label='Left Channel', linewidth=2)
    plt.plot(positions, right_gains, 'r-', label='Right Channel', linewidth=2)
    plt.plot(positions, total_powers, 'g--', label='Total Power', linewidth=2)
    plt.xlabel('Pan Position (-1=Left, 1=Right)')
    plt.ylabel('Gain / Power')
    plt.title('Constant-Power Panning Law')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color='k', linestyle=':', alpha=0.5)
    plt.axvline(x=0.0, color='k', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('panning_law.png', dpi=150)
    print("\n✓ Saved plot to 'panning_law.png'")


def demonstrate_vectorized_panning():
    """Demonstrate vectorized panning performance."""
    print("\n" + "=" * 70)
    print("Vectorized Panning Demonstration")
    print("=" * 70)

    # Create test signal
    sample_rate = 44100
    duration = 0.1  # 100ms
    num_samples = int(sample_rate * duration)

    # Sine wave at 440 Hz
    t = np.arange(num_samples) / sample_rate
    signal = np.sin(2 * np.pi * 440 * t)

    # Pan to the right
    panner = Panner(0.7)

    # Method 1: Element-by-element (slow)
    import time
    start = time.time()
    left_slow = []
    right_slow = []
    for sample in signal:
        l, r = panner(sample)
        left_slow.append(l)
        right_slow.append(r)
    time_slow = time.time() - start

    # Method 2: Vectorized (fast)
    start = time.time()
    left_fast, right_fast = panner.pan_vectorized(signal)
    time_fast = time.time() - start

    # Method 3: Direct call with array (also fast)
    start = time.time()
    left_direct, right_direct = panner(signal)
    time_direct = time.time() - start

    print(f"\nProcessing {num_samples} samples:")
    print(f"  Element-by-element: {time_slow*1000:.2f} ms")
    print(f"  Vectorized method:  {time_fast*1000:.2f} ms")
    print(f"  Direct call:        {time_direct*1000:.2f} ms")
    if time_fast > 0:
        print(f"\n✓ Speedup (vs element-by-element): {time_slow/time_fast:.1f}x")

    # Verify results are identical
    np.testing.assert_array_almost_equal(left_slow, left_fast)
    np.testing.assert_array_almost_equal(right_slow, right_fast)
    np.testing.assert_array_almost_equal(left_fast, left_direct)
    print("✓ All methods produce identical results")


def demonstrate_modulated_panning():
    """Demonstrate modulated panning with ADSR envelope."""
    print("\n" + "=" * 70)
    print("Modulated Panning Demonstration")
    print("=" * 70)

    # Create ADSR envelope for panning modulation
    sample_rate = 1000  # Lower for visualization
    env = ADSREnvelope(
        attack_duration=0.2,
        decay_duration=0.2,
        sustain_level=0.5,
        release_duration=0.3,
        sample_rate=sample_rate
    )

    # Create modulated panner
    panner = ModulatedPanner(env)

    # Generate samples
    num_samples = int(sample_rate * 1.0)  # 1 second

    positions = []
    left_values = []
    right_values = []
    powers = []

    for i in range(num_samples):
        # Get pan position
        pos = panner.position
        positions.append(pos)

        # Pan a unit signal
        left, right = panner(1.0)
        left_values.append(left)
        right_values.append(right)

        # Calculate power
        power = left**2 + right**2
        powers.append(power)

        # Advance modulator
        next(panner)

        # Trigger release at 60% through
        if i == int(num_samples * 0.6):
            env.trigger_release()

    # Convert to arrays
    positions = np.array(positions)
    left_values = np.array(left_values)
    right_values = np.array(right_values)
    powers = np.array(powers)
    time_axis = np.arange(num_samples) / sample_rate

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Pan position over time
    axes[0].plot(time_axis, positions, 'g-', linewidth=2)
    axes[0].set_ylabel('Pan Position')
    axes[0].set_title('Modulated Panning with ADSR Envelope')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle=':', alpha=0.5)
    axes[0].set_ylim(-1.1, 1.1)

    # Left and right channel gains
    axes[1].plot(time_axis, left_values, 'b-', label='Left Channel', linewidth=2)
    axes[1].plot(time_axis, right_values, 'r-', label='Right Channel', linewidth=2)
    axes[1].set_ylabel('Channel Gain')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Total power
    axes[2].plot(time_axis, powers, 'purple', linewidth=2)
    axes[2].axhline(y=1.0, color='g', linestyle='--', label='Expected Power = 1.0')
    axes[2].set_ylabel('Total Power')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0.95, 1.05)

    plt.tight_layout()
    plt.savefig('modulated_panning.png', dpi=150)
    print("\n✓ Saved plot to 'modulated_panning.png'")

    # Verify power preservation
    print(f"\n✓ Power variation: {np.std(powers):.10f}")
    print(f"✓ All powers ≈ 1.0: {np.allclose(powers, 1.0)}")


def demonstrate_stereo_audio():
    """Demonstrate panning with actual audio signal."""
    print("\n" + "=" * 70)
    print("Stereo Audio Generation")
    print("=" * 70)

    sample_rate = 44100
    duration = 2.0

    # Create oscillator
    osc = SineOscillator(440, amplitude=0.5, sample_rate=sample_rate)

    # Create ADSR for panning (sweeps from left to right)
    pan_env = ADSREnvelope(
        attack_duration=1.0,
        decay_duration=0.5,
        sustain_level=1.0,
        release_duration=0.5,
        sample_rate=sample_rate
    )

    # Create modulated panner
    panner = ModulatedPanner(pan_env)

    # Chain oscillator with panner
    chain = Chain(osc, panner)

    # Generate stereo samples
    num_samples = int(sample_rate * duration)

    # Initialize iterator
    iter(chain)

    left_channel = []
    right_channel = []

    print(f"\nGenerating {duration} seconds of stereo audio...")

    for i in range(num_samples):
        sample = next(chain)
        if isinstance(sample, tuple):
            left_channel.append(sample[0])
            right_channel.append(sample[1])
        else:
            # Mono sample, shouldn't happen with panner
            left_channel.append(sample)
            right_channel.append(sample)

        # Trigger release
        if i == int(num_samples * 0.75):
            pan_env.trigger_release()

    left_channel = np.array(left_channel)
    right_channel = np.array(right_channel)

    print(f"✓ Generated {len(left_channel)} stereo samples")
    print(f"  Left channel range: [{left_channel.min():.4f}, {left_channel.max():.4f}]")
    print(f"  Right channel range: [{right_channel.min():.4f}, {right_channel.max():.4f}]")

    # Plot waveforms
    time_axis = np.arange(len(left_channel)) / sample_rate

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, left_channel, 'b-', linewidth=0.5, alpha=0.7)
    plt.ylabel('Left Channel')
    plt.title('Stereo Audio with Modulated Panning')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, duration)

    plt.subplot(2, 1, 2)
    plt.plot(time_axis, right_channel, 'r-', linewidth=0.5, alpha=0.7)
    plt.ylabel('Right Channel')
    plt.xlabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, duration)

    plt.tight_layout()
    plt.savefig('stereo_audio.png', dpi=150)
    print("\n✓ Saved plot to 'stereo_audio.png'")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("PANNER IMPROVEMENTS DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo showcases the improved Panner implementation:")
    print("  • Constant-power panning law for perceptually uniform panning")
    print("  • Vectorized operations for better performance")
    print("  • Modulated panning with envelopes")
    print("  • Power preservation across all pan positions")

    demonstrate_constant_power_panning()

    demonstrate_vectorized_panning()

    demonstrate_modulated_panning()

    demonstrate_stereo_audio()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n✅ Constant-power panning law implemented")
    print("✅ Vectorized operations provide significant speedup")
    print("✅ Modulated panning works correctly with envelopes")
    print("✅ Power is preserved across all pan positions")
    print("✅ New pan range: -1.0 (left) to 1.0 (right), 0.0 (center)")
    print("\nGenerated plots:")
    print("  • panning_law.png - Constant-power law visualization")
    print("  • modulated_panning.png - ADSR-modulated panning")
    print("  • stereo_audio.png - Stereo audio waveforms")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

