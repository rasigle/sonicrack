"""Test Panner module in runtime processing context."""

from __future__ import annotations

from typing import Any

import numpy as np

from sonicrack.gui.modules.modifier.pan import PannerModule
from sonicrack.gui.modules.output.output import OutputModule
from sonicrack.gui.modules.source.oscillator import OscillatorModule
from sonicrack.runtime.engine import AudioEngine
from sonicrack.runtime.specs import RuntimeParameters

PortValue = float | np.ndarray


def _array_value(value: PortValue) -> np.ndarray:
    assert isinstance(value, np.ndarray)
    return value


def test_panner_runtime_center_position(qapp: Any):
    """Test PannerModule runtime processing with center pan position."""
    del qapp

    osc = OscillatorModule()
    panner = PannerModule()

    osc.sine_port.connect(panner.in_port)

    params: RuntimeParameters = {}
    panner.pan_knob.set_value(0.0)

    num_samples = 256
    osc.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    output = _array_value(panner.out_port.value)
    assert output is not None, "Output should not be None"
    assert output.shape == (
        num_samples,
        2,
    ), f"Output should be stereo, got {output.shape}"

    left = output[:, 0]
    right = output[:, 1]
    np.testing.assert_allclose(left, right, rtol=0.01)


def test_panner_runtime_right_position(qapp: Any):
    """Test PannerModule runtime processing with right pan position."""
    del qapp

    osc = OscillatorModule()
    panner = PannerModule()
    osc.sine_port.connect(panner.in_port)

    params: RuntimeParameters = {}
    num_samples = 256

    panner.pan_knob.set_value(1.0)
    osc.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    output = _array_value(panner.out_port.value)
    left = output[:, 0]
    right = output[:, 1]

    left_energy = np.sum(np.abs(left))
    right_energy = np.sum(np.abs(right))
    assert right_energy > left_energy, (
        f"Right channel energy ({right_energy:.2f}) should be greater than "
        f"left ({left_energy:.2f})"
    )


def test_panner_runtime_left_position(qapp: Any):
    """Test PannerModule runtime processing with left pan position."""
    del qapp

    osc = OscillatorModule()
    panner = PannerModule()
    osc.sine_port.connect(panner.in_port)

    params: RuntimeParameters = {}
    num_samples = 256

    panner.pan_knob.set_value(-1.0)
    osc.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    output = _array_value(panner.out_port.value)
    left = output[:, 0]
    right = output[:, 1]

    left_energy = np.sum(np.abs(left))
    right_energy = np.sum(np.abs(right))
    assert left_energy > right_energy, (
        f"Left channel energy ({left_energy:.2f}) should be greater than "
        f"right ({right_energy:.2f})"
    )


def test_panner_unmodulated_runtime(qapp: Any):
    """Test PannerModule runtime processing without modulation input."""
    del qapp

    osc = OscillatorModule()
    panner = PannerModule()

    osc.sine_port.connect(panner.in_port)

    params: RuntimeParameters = {}
    num_samples = 256

    panner.pan_knob.set_value(-0.5)
    osc.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    output = _array_value(panner.out_port.value)
    assert output is not None, "Output should not be None"
    assert output.shape == (
        num_samples,
        2,
    ), f"Output should be stereo, got {output.shape}"

    left = output[:, 0]
    right = output[:, 1]

    left_energy = np.sum(np.abs(left))
    right_energy = np.sum(np.abs(right))
    assert left_energy > right_energy, (
        f"Left channel energy ({left_energy:.2f}) should be greater than right "
        f"({right_energy:.2f})"
    )


def test_panner_with_modulation_input(qapp: Any):
    """Test PannerModule runtime processing with modulation input."""
    del qapp

    audio_osc = OscillatorModule()
    lfo = OscillatorModule()
    panner = PannerModule()

    audio_osc.sine_port.connect(panner.in_port)
    lfo.sine_port.connect(panner.mod_port)

    params: RuntimeParameters = {}
    num_samples = 256

    lfo.freq_knob.set_value(0.5)
    panner.pan_knob.set_value(0.0)

    audio_osc.process_runtime(num_samples, params)
    lfo.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    output = _array_value(panner.out_port.value)
    assert output is not None, "Output should not be None"
    assert output.shape == (
        num_samples,
        2,
    ), f"Output should be stereo, got {output.shape}"

    left = output[:, 0]
    right = output[:, 1]

    assert not np.allclose(left, right, rtol=0.001)


def test_panner_runtime_without_input(qapp: Any):
    """Test PannerModule outputs silence when no input connected."""
    del qapp

    panner = PannerModule()
    params: RuntimeParameters = {}
    num_samples = 256

    panner.process_runtime(num_samples, params)

    output = _array_value(panner.out_port.value)
    assert output is not None, "Output should not be None"

    expected_silence = np.zeros(num_samples, dtype=np.float32)
    np.testing.assert_array_equal(output, expected_silence)


def test_panner_position_reaches_output_render_path(qapp: Any):
    """Knob changes must update the cached runtime params used by AudioEngine."""
    del qapp

    engine = AudioEngine()
    osc = OscillatorModule()
    panner = PannerModule()
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


def test_panner_runtime_uses_engine_component(qapp: Any):
    """Test that PannerModule uses its engine component in unmodulated mode."""
    del qapp

    osc = OscillatorModule()
    panner = PannerModule()
    osc.sine_port.connect(panner.in_port)

    params: RuntimeParameters = {}
    num_samples = 256

    panner.pan_knob.set_value(-0.8)

    osc.process_runtime(num_samples, params)
    panner.process_runtime(num_samples, params)

    output = _array_value(panner.out_port.value)
    left = output[:, 0]
    right = output[:, 1]

    assert np.sum(np.abs(left)) > np.sum(np.abs(right))
