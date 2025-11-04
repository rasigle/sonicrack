"""
Test pan and volume smoothing to verify click elimination.

Verifies that changing pan position and volume amplitude doesn't cause clicks.
"""

import numpy as np

from src.engine.modifier import Panner, Volume


def test_panner_smoothing():
    """Test that pan position changes are smoothed."""
    panner = Panner(position=0.0)  # Start at center

    # Generate some mono samples
    mono_samples = np.ones(1000) * 0.5

    # Pan at center (should be equal L/R)
    left1, right1 = panner.pan_vectorized(mono_samples[:100])
    print(f"Center pan - left: {left1[0]:.3f}, right: {right1[0]:.3f}")

    # Change pan position (should trigger smoothing)
    panner.position = 1.0  # Hard right
    print(f"After setting to hard right:")
    print(f"  Target gains: L={panner._target_left_gain:.3f}, R={panner._target_right_gain:.3f}")
    print(f"  Current gains: L={panner._current_left_gain:.3f}, R={panner._current_right_gain:.3f}")

    # Verify smoothing state is active
    assert panner._smoothing_samples_remaining > 0
    assert panner._target_left_gain != panner._current_left_gain

    # Generate samples - should smooth over 10ms (441 samples @ 44.1kHz)
    left2, right2 = panner.pan_vectorized(mono_samples[100:541])
    print(f"During smoothing - first: L={left2[0]:.3f}, R={right2[0]:.3f}")
    print(f"During smoothing - last: L={left2[-1]:.3f}, R={right2[-1]:.3f}")

    # Verify smoothing occurred (gradual transition, not instant)
    # First few samples should be closer to old value
    assert abs(left2[0] - left1[-1]) < 0.1  # Smooth transition

    # After smoothing completes, generate more samples to verify target reached
    left3, right3 = panner.pan_vectorized(mono_samples[541:641])
    print(f"After smoothing - L={left3[-1]:.3f}, R={right3[-1]:.3f}")
    print(f"  Final gains: L={panner._target_left_gain:.3f}, R={panner._target_right_gain:.3f}")

    # Now should be at target (hard right: low left, high right)
    # With input of 0.5, hard right should give: left ≈ 0 (0.5 * 0), right ≈ 0.5 (0.5 * 1.0)
    assert left3[-1] < 0.1  # Should be close to zero
    assert right3[-1] > 0.45  # Should be close to 0.5 (input * right_gain)

    print("✓ Panner smoothing works correctly")


def test_volume_smoothing():
    """Test that volume/gain changes are smoothed."""
    vol = Volume(amplitude=1.0)

    # Generate samples
    samples = np.ones(1000) * 0.5

    # Apply volume at 1.0
    result1 = vol.scale_vectorized(samples[:100])
    assert np.allclose(result1, 0.5)

    # Change volume (should trigger smoothing)
    vol.amplitude = 0.5

    # Verify smoothing state is active
    assert vol._smoothing_samples_remaining > 0
    assert vol._target_amplitude != vol._current_amplitude

    # Generate samples - should smooth
    result2 = vol.scale_vectorized(samples[100:541])

    # Verify smooth transition (not instant jump)
    assert abs(result2[0] - result1[-1]) < 0.1  # Smooth start

    # Should end near target (0.5 * 0.5 = 0.25)
    assert abs(result2[-1] - 0.25) < 0.05

    print("✓ Volume smoothing works correctly")


def test_volume_gain_db_smoothing():
    """Test that gain_db changes are smoothed."""
    vol = Volume(gain_db=0)  # Unity gain

    samples = np.ones(1000) * 0.5

    # Apply at 0 dB
    result1 = vol.scale_vectorized(samples[:100])
    assert np.allclose(result1, 0.5)

    # Change to -12 dB (0.25 linear)
    vol.gain_db = -12

    # Verify smoothing triggered
    assert vol._smoothing_samples_remaining > 0

    # Generate with smoothing
    result2 = vol.scale_vectorized(samples[100:541])

    # Smooth transition
    assert abs(result2[0] - result1[-1]) < 0.1

    # Ends at -12 dB (0.5 * 0.25 = 0.125)
    assert abs(result2[-1] - 0.125) < 0.05

    print("✓ Volume gain_db smoothing works correctly")


def test_no_clicking_on_rapid_changes():
    """Test rapid parameter changes don't accumulate discontinuities."""
    panner = Panner(position=-1.0)  # Start left

    samples = np.ones(2000) * 0.5
    all_left = []
    all_right = []

    # Make several rapid pan changes
    positions = [0.0, 0.5, -0.5, 0.0, 1.0]
    sample_idx = 0

    for pos in positions:
        panner.position = pos
        left, right = panner.pan_vectorized(samples[sample_idx:sample_idx+400])
        all_left.extend(left)
        all_right.extend(right)
        sample_idx += 400

    # Convert to arrays
    all_left = np.array(all_left)
    all_right = np.array(all_right)

    # Check for discontinuities (large jumps)
    left_diff = np.abs(np.diff(all_left))
    right_diff = np.abs(np.diff(all_right))

    # Max jump should be small (smooth transitions)
    max_left_jump = np.max(left_diff)
    max_right_jump = np.max(right_diff)

    # With smoothing, jumps should be limited (not instant)
    # Without smoothing, we'd see jumps close to 1.0
    assert max_left_jump < 0.05, f"Left channel has large jump: {max_left_jump}"
    assert max_right_jump < 0.05, f"Right channel has large jump: {max_right_jump}"

    print("✓ Rapid pan changes don't cause clicks")
    print(f"  Max left jump: {max_left_jump:.4f}")
    print(f"  Max right jump: {max_right_jump:.4f}")


if __name__ == "__main__":
    print("Testing Pan and Volume Smoothing...")
    print("="*60)

    test_panner_smoothing()
    test_volume_smoothing()
    test_volume_gain_db_smoothing()
    test_no_clicking_on_rapid_changes()

    print("="*60)
    print("All smoothing tests passed! ✅")
    print("\nPan and Volume changes are now click-free!")

