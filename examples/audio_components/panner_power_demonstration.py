"""Demonstrate constant-power Panner and ModulatedPanner behavior.

This example demonstrates:
1. Constant-power panning law
2. Vectorized panning operations
3. Modulated panning with an ADSR envelope
4. Power preservation across pan positions
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.engine import (
    ADSREnvelope,
    ModulatedPanner,
    Panner,
    SineOscillator,
    unipolar_to_bipolar,
)

OUTPUT_DIR = Path(__file__).resolve().parent


def demonstrate_constant_power_panning() -> None:
    """Demonstrate constant-power panning law."""
    print("=" * 70)
    print("Constant-Power Panning Demonstration")
    print("=" * 70)

    positions = np.linspace(-1.0, 1.0, 21)

    left_gains = []
    right_gains = []
    total_powers = []

    print("\nPan Position | Left Gain | Right Gain | Total Power")
    print("-" * 60)

    for position in positions:
        panner = Panner(position)
        left, right = panner(1.0)
        power = left**2 + right**2

        left_gains.append(left)
        right_gains.append(right)
        total_powers.append(power)

        print(f"{position:12.2f} | {left:9.4f} | {right:10.4f} | {power:11.6f}")

    print(f"\n[OK] Power variation: {np.std(total_powers):.10f}")
    print(f"[OK] All powers ~= 1.0: {np.allclose(total_powers, 1.0)}")

    plt.figure(figsize=(10.0, 6.0))
    plt.plot(positions, left_gains, "b-", label="Left Channel", linewidth=2)
    plt.plot(positions, right_gains, "r-", label="Right Channel", linewidth=2)
    plt.plot(positions, total_powers, "g--", label="Total Power", linewidth=2)
    plt.xlabel("Pan Position (-1=Left, 1=Right)")
    plt.ylabel("Gain / Power")
    plt.title("Constant-Power Panning Law")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color="k", linestyle=":", alpha=0.5)
    plt.axvline(x=0.0, color="k", linestyle=":", alpha=0.5)
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "panning_law.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n[OK] Saved plot to '{plot_path}'")


def demonstrate_vectorized_panning() -> None:
    """Demonstrate vectorized panning performance."""
    print("\n" + "=" * 70)
    print("Vectorized Panning Demonstration")
    print("=" * 70)

    sample_rate = 44100
    duration = 0.1
    num_samples = int(sample_rate * duration)

    time_axis = np.arange(num_samples) / sample_rate
    signal = np.sin(2 * np.pi * 440 * time_axis)

    panner = Panner(0.7)

    start = time.perf_counter()
    left_slow = []
    right_slow = []
    for sample in signal:
        left, right = panner(sample)
        left_slow.append(left)
        right_slow.append(right)
    time_slow = time.perf_counter() - start

    start = time.perf_counter()
    left_fast, right_fast = panner.pan_vectorized(signal)
    time_fast = time.perf_counter() - start

    start = time.perf_counter()
    left_direct, right_direct = panner(signal)
    time_direct = time.perf_counter() - start

    print(f"\nProcessing {num_samples} samples:")
    print(f"  Element-by-element: {time_slow * 1000:.2f} ms")
    print(f"  Vectorized method:  {time_fast * 1000:.2f} ms")
    print(f"  Direct call:        {time_direct * 1000:.2f} ms")
    if time_fast > 0:
        print(f"\n[OK] Speedup (vs element-by-element): {time_slow / time_fast:.1f}x")

    np.testing.assert_allclose(left_slow, left_fast, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(right_slow, right_fast, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(left_fast, left_direct, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(right_fast, right_direct, rtol=1e-6, atol=1e-7)
    print("[OK] All methods produce matching results")


def demonstrate_modulated_panning() -> None:
    """Demonstrate modulated panning with ADSR envelope."""
    print("\n" + "=" * 70)
    print("Modulated Panning Demonstration")
    print("=" * 70)

    sample_rate = 1000
    env = ADSREnvelope(
        attack_duration=0.2,
        decay_duration=0.2,
        sustain_level=0.5,
        release_duration=0.3,
        sample_rate=sample_rate,
    )
    env.trigger_note_on()

    # ADSR is unipolar [0, 1]. ModulatedPanner expects bipolar [-1, 1].
    panner = ModulatedPanner(unipolar_to_bipolar(env))

    num_samples = int(sample_rate * 1.0)
    release_sample = int(num_samples * 0.6)

    positions = []
    left_values = []
    right_values = []
    powers = []

    for sample_index in range(num_samples):
        left, right = panner(1.0)
        positions.append(panner.position)
        left_values.append(left)
        right_values.append(right)
        powers.append(left**2 + right**2)

        if sample_index == release_sample:
            env.trigger_release()

    positions = np.array(positions)
    left_values = np.array(left_values)
    right_values = np.array(right_values)
    powers = np.array(powers)
    time_axis = np.arange(num_samples) / sample_rate

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 8.0))

    axes[0].plot(time_axis, positions, "g-", linewidth=2)
    axes[0].set_ylabel("Pan Position")
    axes[0].set_title("Modulated Panning with ADSR Envelope")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color="k", linestyle=":", alpha=0.5)
    axes[0].set_ylim(-1.1, 1.1)

    axes[1].plot(time_axis, left_values, "b-", label="Left Channel", linewidth=2)
    axes[1].plot(time_axis, right_values, "r-", label="Right Channel", linewidth=2)
    axes[1].set_ylabel("Channel Gain")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, powers, "purple", linewidth=2)
    axes[2].axhline(y=1.0, color="g", linestyle="--", label="Expected Power = 1.0")
    axes[2].set_ylabel("Total Power")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0.95, 1.05)

    fig.tight_layout()
    plot_path = OUTPUT_DIR / "modulated_panning.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n[OK] Saved plot to '{plot_path}'")

    print(f"\n[OK] Power variation: {np.std(powers):.10f}")
    print(f"[OK] All powers ~= 1.0: {np.allclose(powers, 1.0)}")


def demonstrate_stereo_audio() -> None:
    """Demonstrate panning with an audio signal."""
    print("\n" + "=" * 70)
    print("Stereo Audio Generation")
    print("=" * 70)

    sample_rate = 44100
    duration = 2.0
    num_samples = int(sample_rate * duration)
    release_sample = int(num_samples * 0.75)

    osc = SineOscillator(440, amplitude=0.5, sample_rate=sample_rate)
    pan_env = ADSREnvelope(
        attack_duration=1.0,
        decay_duration=0.5,
        sustain_level=1.0,
        release_duration=0.5,
        sample_rate=sample_rate,
    )
    pan_env.trigger_note_on()
    panner = ModulatedPanner(unipolar_to_bipolar(pan_env))

    print(f"\nGenerating {duration} seconds of stereo audio...")

    mono_signal = osc.get_samples(num_samples, reset=True, mode="vectorized")
    left_before_release, right_before_release = panner(mono_signal[:release_sample])
    pan_env.trigger_release()
    left_after_release, right_after_release = panner(mono_signal[release_sample:])

    left_channel = np.concatenate((left_before_release, left_after_release))
    right_channel = np.concatenate((right_before_release, right_after_release))

    print(f"[OK] Generated {len(left_channel)} stereo samples")
    print(f"  Left channel range: [{left_channel.min():.4f}, {left_channel.max():.4f}]")
    print(
        f"  Right channel range: "
        f"[{right_channel.min():.4f}, {right_channel.max():.4f}]"
    )

    time_axis = np.arange(len(left_channel)) / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 6.0))
    axes[0].plot(time_axis, left_channel, "b-", linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel("Left Channel")
    axes[0].set_title("Stereo Audio with Modulated Panning")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, duration)

    axes[1].plot(time_axis, right_channel, "r-", linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel("Right Channel")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, duration)

    fig.tight_layout()
    plot_path = OUTPUT_DIR / "stereo_audio.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n[OK] Saved plot to '{plot_path}'")


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("PANNER DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo showcases the Panner implementation:")
    print("  - Constant-power panning law for perceptually uniform panning")
    print("  - Vectorized operations for better performance")
    print("  - Modulated panning with envelopes")
    print("  - Power preservation across all pan positions")

    demonstrate_constant_power_panning()
    demonstrate_vectorized_panning()
    demonstrate_modulated_panning()
    demonstrate_stereo_audio()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n[OK] Constant-power panning law implemented")
    print("[OK] Vectorized operations provide significant speedup")
    print("[OK] Modulated panning works correctly with envelopes")
    print("[OK] Power is preserved across all pan positions")
    print("[OK] Pan range: -1.0 (left), 0.0 (center), 1.0 (right)")
    print("\nGenerated plots:")
    print(f"  - {OUTPUT_DIR / 'panning_law.png'} - Constant-power law visualization")
    print(f"  - {OUTPUT_DIR / 'modulated_panning.png'} - ADSR-modulated panning")
    print(f"  - {OUTPUT_DIR / 'stereo_audio.png'} - Stereo audio waveforms")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
