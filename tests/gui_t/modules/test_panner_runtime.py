"""Test panner modules in runtime processing context."""

from __future__ import annotations

from typing import Any

import numpy as np

from sonicrack.gui.audio_engine import AudioEngine
from sonicrack.gui.modules.modifier.pan_mod import PannerModule
from sonicrack.gui.modules.modifier.pan_simple import SimplePannerModule
from sonicrack.gui.modules.output.output import OutputModule
from sonicrack.gui.modules.source.oscillator import OscillatorModule


def test_simple_panner_runtime_center_position(qapp: Any):
    """Test SimplePannerModule runtime processing with center pan position."""
    del qapp

    # Create oscillator and panner modules
    osc = OscillatorModule()
    panner = SimplePannerModule()

    # Connect oscillator to panner
    osc.sine_port.connect(panner.in_port)

    # Set up runtime parameters
    params = {}

    # Set pan position to center
    panner.pan_knob.set_value(0.0)

    # Process a buffer through oscillator and panner
    num_samples = 256
    osc.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    # Verify output
    output = panner.out_port.value
    assert output is not None, "Output should not be None"
    assert output.shape == (
        num_samples,
        2,
    ), f"Output should be stereo, got {output.shape}"

    # At center, both channels should be approximately equal
    left = output[:, 0]
    right = output[:, 1]
    np.testing.assert_allclose(left, right, rtol=0.01)


def test_simple_panner_runtime_right_position(qapp: Any):
    """Test SimplePannerModule runtime processing with right pan position."""
    del qapp

    # Create and connect modules
    osc = OscillatorModule()
    panner = SimplePannerModule()
    osc.sine_port.connect(panner.in_port)

    params = {}
    num_samples = 256

    # Set pan position to hard right
    panner.pan_knob.set_value(1.0)
    osc.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    output = panner.out_port.value
    left = output[:, 0]
    right = output[:, 1]

    # Right channel should have more energy than left
    left_energy = np.sum(np.abs(left))
    right_energy = np.sum(np.abs(right))
    assert right_energy > left_energy, (
        f"Right channel energy ({right_energy:.2f}) should be greater than "
        f"left ({left_energy:.2f})"
    )


def test_simple_panner_runtime_left_position(qapp: Any):
    """Test SimplePannerModule runtime processing with left pan position."""
    del qapp

    # Create and connect modules
    osc = OscillatorModule()
    panner = SimplePannerModule()
    osc.sine_port.connect(panner.in_port)

    params = {}
    num_samples = 256

    # Set pan position to hard left
    panner.pan_knob.set_value(-1.0)
    osc.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    output = panner.out_port.value
    left = output[:, 0]
    right = output[:, 1]

    # Left channel should have more energy than right
    left_energy = np.sum(np.abs(left))
    right_energy = np.sum(np.abs(right))
    assert left_energy > right_energy, (
        f"Left channel energy ({left_energy:.2f}) should be greater than "
        f"right ({right_energy:.2f})"
    )


def test_modulated_panner_unmodulated_runtime(qapp: Any):
    """Test PannerModule runtime processing without modulation input."""
    del qapp

    # Create oscillator and panner modules
    osc = OscillatorModule()
    panner = PannerModule()

    # Connect oscillator to panner (no modulation)
    osc.sine_port.connect(panner.in_port)

    params = {}
    num_samples = 256

    # Set pan position to left
    panner.pan_knob.set_value(-0.5)
    osc.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    # Verify output
    output = panner.out_port.value
    assert output is not None, "Output should not be None"
    assert output.shape == (
        num_samples,
        2,
    ), f"Output should be stereo, got {output.shape}"

    left = output[:, 0]
    right = output[:, 1]

    # Left channel should have more energy
    left_energy = np.sum(np.abs(left))
    right_energy = np.sum(np.abs(right))
    assert left_energy > right_energy, (
        f"Left channel energy ({left_energy:.2f}) should be greater than right "
        f"({right_energy:.2f})"
    )


def test_modulated_panner_with_modulation_input(qapp: Any):
    """Test PannerModule runtime processing with modulation input."""
    del qapp

    # Create modules
    audio_osc = OscillatorModule()  # Audio signal
    lfo = OscillatorModule()  # Modulation source
    panner = PannerModule()

    # Connect: audio -> panner, LFO -> panner modulation
    audio_osc.sine_port.connect(panner.in_port)
    lfo.sine_port.connect(panner.mod_port)

    params = {}
    num_samples = 256

    # Set LFO to low frequency
    lfo.freq_knob.set_value(0.5)
    panner.pan_knob.set_value(0.0)  # Base position at center

    # Process buffers
    audio_osc.process_runtime(num_samples, params)
    lfo.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    # Verify output
    output = panner.out_port.value
    assert output is not None, "Output should not be None"
    assert output.shape == (
        num_samples,
        2,
    ), f"Output should be stereo, got {output.shape}"

    # With modulation, channels should vary over time
    left = output[:, 0]
    right = output[:, 1]

    # Channels should not be identical (due to modulation)
    # Use a more lenient check since they might be close at some samples
    assert not np.allclose(left, right, rtol=0.001), (
        "Channels should differ significantly with modulation"
    )


def test_simple_panner_runtime_without_input(qapp: Any):
    """Test SimplePannerModule outputs silence when no input connected."""
    del qapp

    panner = SimplePannerModule()
    params = {}
    num_samples = 256

    # Process without connecting input
    panner.process_runtime(num_samples, params)

    output = panner.out_port.value
    assert output is not None, "Output should not be None"

    # Output should be zeros (silence)
    expected_silence = np.zeros(num_samples, dtype=np.float32)
    np.testing.assert_array_equal(output, expected_silence)


def test_modulated_panner_runtime_without_input(qapp: Any):
    """Test PannerModule outputs silence when no input connected."""
    del qapp

    panner = PannerModule()
    params = {}
    num_samples = 256

    # Process without connecting input
    panner.process_runtime(num_samples, params)

    output = panner.out_port.value
    assert output is not None, "Output should not be None"

    # Output should be zeros (silence)
    expected_silence = np.zeros(num_samples, dtype=np.float32)
    np.testing.assert_array_equal(output, expected_silence)


def test_simple_panner_runtime_uses_engine_component(qapp: Any):
    """Test that SimplePannerModule actually uses its engine component in
    process_runtime."""
    del qapp

    osc = OscillatorModule()
    panner = SimplePannerModule()
    osc.sine_port.connect(panner.in_port)

    params = {}
    num_samples = 256

    # Set position via knob (this is what process_runtime will use)
    panner.pan_knob.set_value(0.8)

    # Process
    osc.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    output = panner.out_port.value
    left = output[:, 0]
    right = output[:, 1]

    # At position 0.8 (right), right should be louder
    assert np.sum(np.abs(right)) > np.sum(np.abs(left)), (
        "Component position should affect output"
    )


def test_simple_panner_position_reaches_output_render_path(qapp: Any):
    """Knob changes must update the cached runtime params used by AudioEngine."""
    del qapp

    engine = AudioEngine()
    osc = OscillatorModule()
    panner = SimplePannerModule()
    output = OutputModule()
    output.audio_engine = engine

    for module in (osc, panner, output):
        engine.add_module(module)

    osc.sine_port.connect(panner.in_port)
    panner.out_port.connect(output.inp_port_l)

    panner.pan_knob.set_value(1.0)
    output._generate_samples(512)
    samples = output._generate_samples(512)

    left_energy = np.sum(np.abs(samples[:, 0]))
    right_energy = np.sum(np.abs(samples[:, 1]))

    assert panner.get_parameters()["position"] == 1.0
    assert right_energy > left_energy * 10


def test_modulated_panner_runtime_uses_engine_component(qapp: Any):
    """Test that PannerModule uses its engine component in unmodulated mode."""
    del qapp

    osc = OscillatorModule()
    panner = PannerModule()
    osc.sine_port.connect(panner.in_port)
    # No modulation connected

    params = {}
    num_samples = 256

    # Set position
    panner.pan_knob.set_value(-0.8)

    # Process
    osc.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    output = panner.out_port.value
    left = output[:, 0]
    right = output[:, 1]

    # At position -0.8 (left), left should be louder
    assert np.sum(np.abs(left)) > np.sum(np.abs(right)), (
        "Component position should affect output in unmodulated mode"
    )
