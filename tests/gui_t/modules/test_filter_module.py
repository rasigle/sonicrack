"""Tests for the GUI filter module behavior."""

from __future__ import annotations

import numpy as np

from src.engine.filter import BiquadResonantFilter, ButterworthFilter
from src.gui.core.port import Port
from src.gui.core.runtime import process_runtime_module
from src.gui.modules.modifier.filter import FilterModule
from src.gui.modules.modifier.resonant_filter import ResonantFilterModule


def test_filter_module_create_engine_component_returns_modifier(app):
    """The filter module should return only its modifier component.

    The patch compiler is responsible for wrapping modifiers in a Chain.
    """
    module = FilterModule()

    component = module.create_engine_component(input_components=[object()])

    assert isinstance(component, ButterworthFilter)


def test_filter_module_runtime_filters_signal(app):
    """Runtime processing should apply real filtering."""
    module = FilterModule()
    source = Port("output", "source")
    module.in_port.connect(source)

    sample_rate = 44100
    duration = 0.05
    num_samples = int(sample_rate * duration)
    time = np.arange(num_samples) / sample_rate

    # Mixed low + high frequency content so low-pass filtering has a clear effect.
    input_signal = (
        np.sin(2 * np.pi * 200 * time) + 0.6 * np.sin(2 * np.pi * 5000 * time)
    ).astype(np.float32)
    source.write(input_signal)

    module.cutoff_knob.set_value(500.0)
    module.type_combo.setCurrentText("Low-pass")
    module.component = module.create_engine_component()

    process_runtime_module(module, num_samples)
    output_signal = module.out_port.value

    assert isinstance(output_signal, np.ndarray)
    assert output_signal.shape == input_signal.shape
    assert not np.allclose(output_signal, input_signal)


def test_filter_module_runtime_outputs_silence_without_input(app):
    """Runtime processing should write silence when there is no connected input."""
    module = FilterModule()

    process_runtime_module(module, 16)

    assert isinstance(module.out_port.value, np.ndarray)
    assert np.allclose(module.out_port.value, np.zeros(16, dtype=np.float32))


def test_filter_module_bandpass_reorders_cutoffs(app):
    """Band-pass mode should normalize reversed knob ordering."""
    module = FilterModule()

    module.type_combo.setCurrentText("Band-pass")
    module.cutoff_knob.set_value(2000.0)
    module.high_cutoff_knob.set_value(200.0)

    component = module.create_engine_component()

    assert component.filter_type == "band"
    assert component.cutoff == (200.0, 2000.0)


def test_resonant_filter_module_create_engine_component(app):
    """Resonant filter module should create the synth filter component."""
    module = ResonantFilterModule()
    module.type_combo.setCurrentText("Band-pass")
    module.cutoff_knob.set_value(1200.0)
    module.resonance_knob.set_value(4.0)
    module.drive_knob.set_value(6.0)
    module.output_gain_knob.set_value(-3.0)

    component = module.create_engine_component()

    assert isinstance(component, BiquadResonantFilter)
    assert component.filter_type == "band"
    assert component.cutoff == 1200.0
    assert component.resonance == 4.0
    assert component.drive_db == 6.0
    assert component.output_gain_db == -3.0


def test_resonant_filter_module_runtime_filters_signal(app):
    """Runtime processing should apply the resonant filter."""
    module = ResonantFilterModule()
    source = Port("output", "source")
    module.in_port.connect(source)
    samples = np.ones(64, dtype=np.float32)
    source.write(samples)

    module.process_runtime(
        64,
        {
            "cutoff": 800.0,
            "resonance": 2.0,
            "filter_type": "Low-pass",
            "cv_depth_octaves": 0.0,
            "drive_db": 0.0,
            "output_gain_db": 0.0,
        },
    )

    assert isinstance(module.out_port.value, np.ndarray)
    assert module.out_port.value.shape == samples.shape
    assert not np.allclose(module.out_port.value, samples)
    assert module.component.cutoff == 800.0
    assert module.component.resonance == 2.0


def test_resonant_filter_module_reuses_component_for_cutoff_changes(app):
    """Knob moves should preserve filter state instead of rebuilding the filter."""
    module = ResonantFilterModule()
    source = Port("output", "source")
    module.in_port.connect(source)
    source.write(np.ones(64, dtype=np.float32))

    module.process_runtime(
        64,
        {
            "cutoff": 400.0,
            "resonance": 1.0,
            "filter_type": "Low-pass",
            "cv_depth_octaves": 0.0,
            "drive_db": 0.0,
            "output_gain_db": -6.0,
        },
    )
    component = module.component
    source.write(np.ones(64, dtype=np.float32))
    module.process_runtime(
        64,
        {
            "cutoff": 2000.0,
            "resonance": 1.0,
            "filter_type": "Low-pass",
            "cv_depth_octaves": 0.0,
            "drive_db": 0.0,
            "output_gain_db": -6.0,
        },
    )

    assert module.component is component
    assert module.component.cutoff == 2000.0


def test_resonant_filter_module_smooths_large_parameter_changes(app):
    """Large knob moves should not create a hard sample jump at the buffer edge."""
    module = ResonantFilterModule()
    source = Port("output", "source")
    module.in_port.connect(source)

    sample_rate = 44100
    num_samples = 512
    t = np.arange(num_samples) / sample_rate
    first_input = np.sin(2 * np.pi * 220 * t).astype(np.float32) * 0.25
    second_input = (
        np.sin(2 * np.pi * 220 * (t + num_samples / sample_rate)).astype(np.float32)
        * 0.25
    )

    source.write(first_input)
    module.process_runtime(
        num_samples,
        {
            "cutoff": 300.0,
            "resonance": 0.707,
            "filter_type": "Low-pass",
            "cv_depth_octaves": 0.0,
            "drive_db": 0.0,
            "output_gain_db": -12.0,
        },
    )
    first = np.asarray(module.out_port.value)

    source.write(second_input)
    module.process_runtime(
        num_samples,
        {
            "cutoff": 8000.0,
            "resonance": 8.0,
            "filter_type": "Low-pass",
            "cv_depth_octaves": 0.0,
            "drive_db": 18.0,
            "output_gain_db": -6.0,
        },
    )
    second = np.asarray(module.out_port.value)

    boundary_jump = abs(float(second[0] - first[-1]))
    ordinary_jumps = np.abs(np.diff(np.concatenate((first[-64:], second[:64]))))

    assert boundary_jump < 0.05
    assert boundary_jump <= float(np.percentile(ordinary_jumps, 95)) * 4.0


def test_resonant_filter_module_cutoff_cv_changes_output(app):
    """Cutoff CV should modulate the resonant filter cutoff per buffer."""
    module = ResonantFilterModule()
    source = Port("output", "source")
    cv_source = Port("output", "cv")
    module.in_port.connect(source)
    module.cutoff_cv_port.connect(cv_source)

    sample_rate = 44100
    t = np.arange(512) / sample_rate
    samples = np.sin(2 * np.pi * 1200 * t).astype(np.float32)
    source.write(samples)
    cv_source.write(np.linspace(0.0, 1.0, 512, dtype=np.float32))

    module.process_runtime(
        512,
        {
            "cutoff": 300.0,
            "resonance": 0.707,
            "filter_type": "Low-pass",
            "cv_depth_octaves": 4.0,
            "drive_db": 0.0,
            "output_gain_db": 0.0,
        },
    )
    modulated = np.asarray(module.out_port.value)

    module.component = None
    module._runtime_filter_params = None
    module.cutoff_cv_port.disconnect(cv_source)
    source.write(samples)
    module.process_runtime(
        512,
        {
            "cutoff": 300.0,
            "resonance": 0.707,
            "filter_type": "Low-pass",
            "cv_depth_octaves": 0.0,
            "drive_db": 0.0,
            "output_gain_db": 0.0,
        },
    )
    static = np.asarray(module.out_port.value)

    assert np.mean(np.abs(modulated)) > np.mean(np.abs(static))
