"""Tests for the GUI filter module behavior."""

from __future__ import annotations

import numpy as np

from src.engine.filter import ButterworthFilter
from src.gui.core.port import Port
from src.gui.modules.filter import FilterModule


def test_filter_module_create_engine_component_returns_modifier(app):
    """The filter module should return only its modifier component.

    The patch compiler is responsible for wrapping modifiers in a Chain.
    """
    module = FilterModule()

    component = module.create_engine_component(input_components=[object()])

    assert isinstance(component, ButterworthFilter)


def test_filter_module_process_filters_signal(app):
    """Process should apply real filtering instead of passing input through."""
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

    module.process(num_samples)
    output_signal = module.out_port.value

    assert isinstance(output_signal, np.ndarray)
    assert output_signal.shape == input_signal.shape
    assert not np.allclose(output_signal, input_signal)


def test_filter_module_process_outputs_silence_without_input(app):
    """Process should write silence when there is no connected input."""
    module = FilterModule()

    module.process(16)

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
