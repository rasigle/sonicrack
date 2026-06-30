"""GUI runtime coverage for sequencing modules."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.gui.core.port import Port
from src.gui.module_registry import discover_modules, get_registry
from src.gui.modules.sequencing.accent import AccentModule
from src.gui.modules.sequencing.clock import ClockModule
from src.gui.modules.sequencing.slide import SlideModule
from src.gui.modules.sequencing.step_sequencer import StepSequencerModule


def _connect_signal(input_port: Port, values: np.ndarray) -> Port:
    output_port = Port("output", "Test Out")
    output_port.write(values)
    output_port.connect(input_port)
    return output_port


def test_sequencing_modules_are_discoverable(qapp: Any):
    del qapp
    discover_modules("src.gui.modules", recursive=True)
    registered = get_registry().list_modules()

    assert "Clock" in registered
    assert "Step Sequencer" in registered
    assert "Slide" in registered
    assert "Accent" in registered
    assert "Acid Filter" in registered


def test_clock_module_writes_pulses(qapp: Any):
    del qapp
    module = ClockModule()

    module.process_runtime(8, {"bpm": 60.0, "division": "1/4", "swing": 0.0})

    output = np.asarray(module.clock_port.value)
    assert output.shape == (8,)
    assert output[0] == 1.0


def test_step_sequencer_module_writes_all_cv_outputs(qapp: Any):
    del qapp
    module = StepSequencerModule()
    _connect_signal(
        module.clock_input,
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )

    module.process_runtime(
        4,
        {
            "notes": "36,48",
            "accents": "1,0",
            "slides": "0,1",
            "gate_length": 1.0,
            "bpm": 120.0,
            "division": "1/16",
        },
    )

    assert np.asarray(module.freq_port.value).shape == (4,)
    np.testing.assert_allclose(module.gate_port.value, [1, 1, 1, 1])
    np.testing.assert_allclose(module.accent_port.value, [1, 1, 0, 0])
    np.testing.assert_allclose(module.slide_port.value, [0, 0, 1, 1])


def test_slide_module_processes_frequency_cv(qapp: Any):
    del qapp
    module = SlideModule()
    _connect_signal(
        module.freq_input,
        np.array([100.0, 200.0, 200.0, 200.0], dtype=np.float32),
    )
    _connect_signal(
        module.slide_input,
        np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )

    module.process_runtime(4, {"time": 1.0, "always_on": False})

    output = np.asarray(module.freq_output.value)
    assert output.shape == (4,)
    assert output[0] == 100.0
    assert 100.0 < output[1] < 200.0


def test_accent_module_writes_scaled_cv_outputs(qapp: Any):
    del qapp
    module = AccentModule()
    _connect_signal(
        module.accent_input,
        np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )

    module.process_runtime(
        4,
        {
            "amount": 1.0,
            "decay": 0.0,
            "amp_depth": 0.25,
            "cutoff_depth": 0.5,
            "envelope_depth": 0.75,
        },
    )

    np.testing.assert_allclose(module.amp_port.value, [0.25, 0.25, 0.0, 0.0])
    np.testing.assert_allclose(module.cutoff_port.value, [0.5, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(module.env_port.value, [0.75, 0.75, 0.0, 0.0])
