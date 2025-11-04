"""
Test that all oscillator types have amplitude smoothing.
"""

import pytest
import numpy as np
from src.engine.oscillator import (
    SineOscillator,
    SquareOscillator,
    SawtoothOscillator,
    TriangleOscillator
)


@pytest.mark.parametrize("osc_class,name", [
    (SineOscillator, "Sine"),
    (SquareOscillator, "Square"),
    (SawtoothOscillator, "Sawtooth"),
    (TriangleOscillator, "Triangle"),
])
def test_oscillator_has_smoothing_state(osc_class, name):
    """Test that oscillator type has smoothing state variables."""
    osc = osc_class(frequency=440, gain_db=-20)

    # Check smoothing state exists
    assert hasattr(osc, '_smoothing_samples_remaining'), f"{name} missing smoothing state!"
    assert hasattr(osc, '_target_amplitude'), f"{name} missing target amplitude!"
    assert hasattr(osc, '_current_amplitude'), f"{name} missing current amplitude!"


@pytest.mark.parametrize("osc_class,name", [
    (SineOscillator, "Sine"),
    (SquareOscillator, "Square"),
    (SawtoothOscillator, "Sawtooth"),
    (TriangleOscillator, "Triangle"),
])
def test_oscillator_smoothing_triggers_on_gain_change(osc_class, name):
    """Test that smoothing is triggered when gain changes."""
    osc = osc_class(frequency=440, gain_db=-20)

    # Complete initial smoothing (initialization triggers smoothing)
    osc.get_samples_vectorized(500)

    # Now smoothing should be complete
    assert osc._smoothing_samples_remaining == 0, f"{name} should have no smoothing after init"

    # Change gain (should trigger smoothing)
    osc.gain_db = -6

    # Check smoothing triggered
    assert osc._smoothing_samples_remaining > 0, f"{name} smoothing not triggered!"


@pytest.mark.parametrize("osc_class,name", [
    (SineOscillator, "Sine"),
    (SquareOscillator, "Square"),
    (SawtoothOscillator, "Sawtooth"),
    (TriangleOscillator, "Triangle"),
])
def test_oscillator_smoothing_prevents_instant_jump(osc_class, name):
    """Test that smoothing prevents instant amplitude jumps."""
    osc = osc_class(frequency=440, gain_db=-20)

    # Generate initial samples
    samples1 = osc.get_samples_vectorized(100)

    # Change gain (should trigger smoothing)
    osc.gain_db = -6  # Much louder

    # Generate samples during smoothing
    samples2 = osc.get_samples_vectorized(441)  # Full smoothing duration

    # Check that transition is smooth (no instant jump)
    first_sample = np.abs(samples2[0])
    # Should be close to old amplitude (0.1)
    assert first_sample < 0.15, f"{name} instant jump detected! First sample: {first_sample}"


@pytest.mark.parametrize("osc_class,name", [
    (SineOscillator, "Sine"),
    (SquareOscillator, "Square"),
    (SawtoothOscillator, "Sawtooth"),
    (TriangleOscillator, "Triangle"),
])
def test_oscillator_smoothing_reaches_target(osc_class, name):
    """Test that smoothing reaches the target amplitude."""
    osc = osc_class(frequency=440, gain_db=-20)

    # Change gain
    osc.gain_db = -6  # Target amplitude ~0.5

    # Generate samples during smoothing
    samples2 = osc.get_samples_vectorized(441)  # Full smoothing duration

    # Check that we reach target amplitude (use peak instead of last sample)
    peak_during_transition = np.max(np.abs(samples2))
    # Allow tolerance since waveform may not hit exact peak at arbitrary sample point
    assert peak_during_transition > 0.3, f"{name} didn't transition enough! Peak: {peak_during_transition}"


@pytest.mark.parametrize("osc_class,name", [
    (SineOscillator, "Sine"),
    (SquareOscillator, "Square"),
    (SawtoothOscillator, "Sawtooth"),
    (TriangleOscillator, "Triangle"),
])
def test_oscillator_smoothing_completes(osc_class, name):
    """Test that smoothing completes after the smoothing duration."""
    osc = osc_class(frequency=440, gain_db=-20)

    # Change gain
    osc.gain_db = -6

    # Generate samples during smoothing
    osc.get_samples_vectorized(441)  # Full smoothing duration

    # Generate more samples (smoothing should be complete)
    osc.get_samples_vectorized(100)

    # Check smoothing completed
    assert osc._smoothing_samples_remaining == 0, f"{name} smoothing didn't complete!"

