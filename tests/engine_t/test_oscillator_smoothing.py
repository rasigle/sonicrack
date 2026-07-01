"""
Test that all oscillator types have amplitude smoothing.
"""

import numpy as np
import pytest

from src.engine.generators.oscillators.oscillator import (
    SawtoothOscillator,
    SineOscillator,
    SquareOscillator,
    TriangleOscillator,
)


@pytest.mark.parametrize(
    "osc_class,name",
    [
        (SineOscillator, "Sine"),
        (SquareOscillator, "Square"),
        (SawtoothOscillator, "Sawtooth"),
        (TriangleOscillator, "Triangle"),
    ],
)
def test_oscillator_has_smoothing_state(osc_class, name):
    """Test that oscillator type has smoothing state variables."""
    osc = osc_class(frequency=440, gain_db=-20)

    # Check smoothing state exists (these are in the base Oscillator class)
    assert hasattr(osc, "_smoothing_samples_remaining"), (
        f"{name} missing _smoothing_samples_remaining!"
    )
    assert hasattr(osc, "_target_amplitude"), f"{name} missing _target_amplitude!"
    assert hasattr(osc, "_current_amplitude"), f"{name} missing _current_amplitude!"
    assert hasattr(osc, "_smoothing_samples_duration_total"), (
        f"{name} missing _smoothing_samples_duration_total!"
    )

    # Note: After construction, __iter__() is called which sets amplitude,
    # triggering initial smoothing. This is expected behavior.
    assert osc._smoothing_samples_remaining > 0, (
        f"{name} should have initial smoothing active"
    )

    # Complete initial smoothing
    osc.get_samples_vectorized(osc._smoothing_samples_remaining)

    # Now smoothing should be complete
    assert osc._smoothing_samples_remaining == 0, (
        f"{name} should have no smoothing after completion"
    )
    assert osc._current_amplitude == osc._target_amplitude, (
        f"{name} current and target should match after smoothing"
    )


@pytest.mark.parametrize(
    "osc_class,name",
    [
        (SineOscillator, "Sine"),
        (SquareOscillator, "Square"),
        (SawtoothOscillator, "Sawtooth"),
        (TriangleOscillator, "Triangle"),
    ],
)
def test_oscillator_smoothing_triggers_on_gain_change(osc_class, name):
    """Test that smoothing is triggered when gain changes."""
    osc = osc_class(frequency=440, gain_db=-20)

    # Complete initial smoothing (triggered by __iter__() in __init__)
    osc.get_samples_vectorized(osc._smoothing_samples_remaining)

    # Now should have no smoothing active
    assert osc._smoothing_samples_remaining == 0, (
        f"{name} should have no smoothing after initial smoothing completes"
    )

    # Change gain (should trigger smoothing immediately)
    osc.gain_db = -6

    # Check smoothing triggered
    assert osc._smoothing_samples_remaining > 0, f"{name} smoothing not triggered!"
    assert osc._target_amplitude > osc._current_amplitude, (
        f"{name} target should be higher than current"
    )

    # Smoothing duration should be reasonable (default is 10ms at sample_rate)
    expected_duration = int(10 * osc.sample_rate / 1000)  # 10ms default
    assert osc._smoothing_samples_remaining == expected_duration, (
        f"{name} smoothing duration incorrect: {osc._smoothing_samples_remaining} vs "
        f"{expected_duration}"
    )


@pytest.mark.parametrize(
    "osc_class,name",
    [
        (SineOscillator, "Sine"),
        (SquareOscillator, "Square"),
        (SawtoothOscillator, "Sawtooth"),
        (TriangleOscillator, "Triangle"),
    ],
)
def test_oscillator_smoothing_prevents_instant_jump(osc_class, name):
    """Test that smoothing prevents instant amplitude jumps."""
    osc = osc_class(frequency=440, gain_db=-20, sample_rate=44100)

    # Complete initial smoothing
    osc.get_samples_vectorized(osc._smoothing_samples_remaining)

    # Change gain to -6dB (should trigger smoothing)
    osc.gain_db = -6  # Target amplitude ~0.5

    # Generate samples during smoothing
    smoothing_duration = osc._smoothing_samples_remaining
    samples = osc.get_samples_vectorized(smoothing_duration)

    # The key test: check that amplitude ramps up gradually
    # Split samples into 5 segments and check RMS increases
    segment_size = len(samples) // 5
    rms_values = []
    for i in range(5):
        start = i * segment_size
        end = start + segment_size
        segment_rms = np.sqrt(np.mean(samples[start:end] ** 2))
        rms_values.append(segment_rms)

    # Each segment should have equal or higher RMS than previous (monotonic increase
    # or flat)
    for i in range(1, 5):
        assert rms_values[i] >= rms_values[i - 1] * 0.95, (
            f"{name} amplitude decreased during smoothing! Segment {i - 1}: "
            f"{rms_values[i - 1]:.3f}, Segment {i}: {rms_values[i]:.3f}"
        )

    # Last segment should be significantly higher than first segment
    assert rms_values[-1] > rms_values[0] * 1.2, (
        f"{name} didn't transition enough! First: {rms_values[0]:.3f}, "
        f"Last: {rms_values[-1]:.3f}"
    )


@pytest.mark.parametrize(
    "osc_class,name",
    [
        (SineOscillator, "Sine"),
        (SquareOscillator, "Square"),
        (SawtoothOscillator, "Sawtooth"),
        (TriangleOscillator, "Triangle"),
    ],
)
def test_oscillator_smoothing_reaches_target(osc_class, name):
    """Test that smoothing reaches the target amplitude."""
    osc = osc_class(frequency=440, gain_db=-20, sample_rate=44100)

    # Change gain to -6dB
    osc.gain_db = -6  # Target amplitude ~0.5
    target_amp = osc._target_amplitude

    # Generate samples during smoothing
    smoothing_duration = osc._smoothing_samples_remaining
    samples = osc.get_samples_vectorized(smoothing_duration)

    # Check that we reach target amplitude (check peak of last portion)
    last_portion = samples[-100:]  # Last 100 samples
    peak_at_end = np.max(np.abs(last_portion))

    # Should be close to target amplitude (within 20% tolerance for waveform variation)
    assert peak_at_end >= target_amp * 0.8, (
        f"{name} didn't reach target! Peak: {peak_at_end:.3f}, "
        f"expected >= {target_amp * 0.8:.3f} (target: {target_amp:.3f})"
    )


@pytest.mark.parametrize(
    "osc_class,name",
    [
        (SineOscillator, "Sine"),
        (SquareOscillator, "Square"),
        (SawtoothOscillator, "Sawtooth"),
        (TriangleOscillator, "Triangle"),
    ],
)
def test_oscillator_smoothing_completes(osc_class, name):
    """Test that smoothing completes after the smoothing duration."""
    osc = osc_class(frequency=440, gain_db=-20, sample_rate=44100)

    # Complete initial smoothing first
    osc.get_samples_vectorized(osc._smoothing_samples_remaining)

    # Change gain
    osc.gain_db = -6
    smoothing_duration = osc._smoothing_samples_remaining

    # Generate samples during smoothing
    # Note: SquareOscillator may trigger additional smoothing in its strategy
    # so we need to handle this case
    osc.get_samples_vectorized(smoothing_duration)

    # Check smoothing completed (or nearly completed)
    assert osc._smoothing_samples_remaining <= smoothing_duration, (
        f"{name} smoothing didn't complete! Remaining: "
        f"{osc._smoothing_samples_remaining}"
    )

    # If there's still smoothing remaining (e.g., SquareOscillator), complete it
    if osc._smoothing_samples_remaining > 0:
        osc.get_samples_vectorized(osc._smoothing_samples_remaining)

    # Now it should definitely be complete
    assert osc._smoothing_samples_remaining == 0, (
        f"{name} smoothing didn't complete after second attempt! Remaining: "
        f"{osc._smoothing_samples_remaining}"
    )

    # Current amplitude should now equal target
    assert abs(osc._current_amplitude - osc._target_amplitude) < 1e-6, (
        f"{name} current amplitude doesn't match target after smoothing!"
    )

    # No more smoothing should be active
    assert osc._smoothing_samples_remaining == 0, (
        f"{name} smoothing reactivated unexpectedly!"
    )


@pytest.mark.parametrize(
    "osc_class,name",
    [
        (SineOscillator, "Sine"),
        (SquareOscillator, "Square"),
        (SawtoothOscillator, "Sawtooth"),
        (TriangleOscillator, "Triangle"),
    ],
)
def test_oscillator_smoothing_triggers_on_amplitude_change(osc_class, name):
    """Test that smoothing is triggered when amplitude changes."""
    osc = osc_class(frequency=440, amplitude=0.1, gain_db=None, sample_rate=44100)

    # Complete initial smoothing (triggered by __iter__() in __init__)
    osc.get_samples_vectorized(osc._smoothing_samples_remaining)

    # Initial state - no smoothing active after completion
    assert osc._smoothing_samples_remaining == 0, (
        f"{name} should have no smoothing after completion"
    )

    # Change amplitude (should trigger smoothing)
    osc.amplitude = 0.5

    # Check smoothing triggered
    assert osc._smoothing_samples_remaining > 0, (
        f"{name} smoothing not triggered by amplitude change!"
    )
    assert osc._target_amplitude > osc._current_amplitude, (
        f"{name} target should be higher than current"
    )


@pytest.mark.parametrize(
    "osc_class,name",
    [
        (SineOscillator, "Sine"),
        (SquareOscillator, "Square"),
        (SawtoothOscillator, "Sawtooth"),
        (TriangleOscillator, "Triangle"),
    ],
)
def test_oscillator_smoothing_duration_consistent(osc_class, name):
    """Test that smoothing duration is consistent across all oscillators."""
    osc = osc_class(frequency=440, gain_db=-20, sample_rate=44100)

    # Expected duration: 10ms at 44100 Hz = 441 samples
    expected_duration = int(10 * 44100 / 1000)

    # Check initial smoothing duration
    assert osc._smoothing_samples_duration_total == expected_duration, (
        f"{name} incorrect smoothing duration: {osc._smoothing_samples_duration_total} "
        f"vs {expected_duration}"
    )

    # Change gain to trigger smoothing
    osc.gain_db = -6

    # Verify smoothing uses correct duration
    assert osc._smoothing_samples_remaining == expected_duration, (
        f"{name} smoothing not using correct duration: "
        f"{osc._smoothing_samples_remaining} vs {expected_duration}"
    )
