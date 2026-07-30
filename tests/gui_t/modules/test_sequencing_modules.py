"""GUI runtime coverage for sequencing modules."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from sonicrack.gui.modules.sequencing.accent import AccentModule
from sonicrack.gui.modules.sequencing.behringer_182 import Behringer182Module
from sonicrack.gui.modules.sequencing.clock import ClockModule
from sonicrack.gui.modules.sequencing.slide import SlideModule
from sonicrack.gui.modules.sequencing.step_sequencer import StepSequencerModule
from sonicrack.gui.widgets import ImagePushButton
from sonicrack.patching.port import Port
from sonicrack.patching.registry import discover_modules, get_registry


def _connect_signal(input_port: Port, values: np.ndarray) -> Port:
    output_port = Port("output", "Test Out")
    output_port.write(values)
    output_port.connect(input_port)
    return output_port


def test_sequencing_modules_are_discoverable(qapp: Any):
    del qapp
    discover_modules("sonicrack.gui.modules", recursive=True)
    registered = get_registry().list_modules()

    assert "Clock" in registered
    assert "Step Sequencer" in registered
    assert "Slide" in registered
    assert "Accent" in registered
    assert "Behringer 182" in registered
    assert "Acid Filter" in registered
    assert "TB-303 Voice" in registered


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
    np.testing.assert_allclose(module.freq_port.value, [-2.0, -2.0, -1.0, -1.0])
    np.testing.assert_allclose(module.gate_port.value, [1, 1, 1, 1])
    np.testing.assert_allclose(module.accent_port.value, [1, 1, 0, 0])
    np.testing.assert_allclose(module.slide_port.value, [0, 0, 1, 1])


def test_step_sequencer_step_toggles_follow_text_parameters(qapp: Any):
    del qapp
    module = StepSequencerModule()

    module.accent_edit.setText("0,1,0,1,0,1,0,1")
    module.slide_edit.setText("1,0,1,0,1,0,1,0")

    assert [button.isChecked() for button in module.accent_buttons] == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert [led.is_on() for led in module.slide_leds] == [
        True,
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_step_sequencer_step_toggles_update_text_parameters(qapp: Any):
    del qapp
    module = StepSequencerModule()

    module.accent_buttons[1].setChecked(True)
    module.slide_buttons[0].setChecked(True)

    assert module.accent_edit.text() == "1,1,0,1,0,0,1,0"
    assert module.slide_edit.text() == "1,0,1,0,0,0,1,0"


def test_step_sequencer_randomize_updates_pattern_and_toggles(qapp: Any):
    del qapp
    module = StepSequencerModule()
    original_notes = module.notes_edit.text()

    module.randomize_pattern(random.Random(42))

    notes = [item.strip() for item in module.notes_edit.text().split(",") if item.strip()]
    accents = module.accent_edit.text().split(",")
    slides = module.slide_edit.text().split(",")

    assert len(notes) == module.step_toggle_count
    assert len(accents) == module.step_toggle_count
    assert len(slides) == module.step_toggle_count
    assert module.notes_edit.text() != original_notes
    assert all(item == "-" or item.lstrip("-").isdigit() for item in notes)
    assert all(flag in {"0", "1"} for flag in accents)
    assert all(flag in {"0", "1"} for flag in slides)
    assert [button.isChecked() for button in module.accent_buttons] == [
        flag == "1" for flag in accents
    ]
    assert [button.isChecked() for button in module.slide_buttons] == [
        flag == "1" for flag in slides
    ]

    # Rests should not carry accent or slide flags.
    for index, note in enumerate(notes):
        if note == "-":
            assert accents[index] == "0"
            assert slides[index] == "0"


def test_step_sequencer_randomize_button_triggers_pattern_change(qapp: Any):
    del qapp
    module = StepSequencerModule()
    module.notes_edit.setText("36,36,36,36,36,36,36,36")
    module.accent_edit.setText("0,0,0,0,0,0,0,0")
    module.slide_edit.setText("0,0,0,0,0,0,0,0")
    before = (
        module.notes_edit.text(),
        module.accent_edit.text(),
        module.slide_edit.text(),
    )

    module.randomize_button.click()

    after = (
        module.notes_edit.text(),
        module.accent_edit.text(),
        module.slide_edit.text(),
    )
    assert after != before
    assert module.randomize_button.text() == "Randomize"


def test_behringer_182_module_writes_dual_cv_and_gate_outputs(qapp: Any):
    del qapp
    module = Behringer182Module()
    _connect_signal(
        module.clock_input,
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )

    module.process_runtime(
        4,
        {
            "cv_a": "0,0.5",
            "cv_b": "1,0.25",
            "gates": "1,0",
            "steps": "2",
            "direction": "forward",
            "gate_length": 1.0,
            "cv_a_range": 4.0,
            "cv_b_range": 2.0,
            "bpm": 60.0,
            "division": "1/4",
        },
    )

    np.testing.assert_allclose(module.cv_a_port.value, [0.0, 0.0, 2.0, 2.0])
    np.testing.assert_allclose(module.cv_b_port.value, [2.0, 2.0, 0.5, 0.5])
    np.testing.assert_allclose(module.gate_port.value, [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(module.trigger_port.value, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(module.end_port.value, [0.0, 0.0, 0.0, 0.0])


def test_behringer_182_uses_compact_panel_controls(qapp: Any):
    del qapp
    module = Behringer182Module()

    assert isinstance(module.run_button, ImagePushButton)
    assert module.run_button.text() == "Stop"
    assert module.run_button.width() == 54
    assert all(knob.knob_size == 34 for knob in module.cv_a_knobs)
    assert all(knob.knob_size == 34 for knob in module.cv_b_knobs)
    assert len(module.step_leds) == 8

    module.run_button.setChecked(False)

    assert module.run_button.text() == "Start"


def test_behringer_182_step_leds_follow_active_step(qapp: Any):
    del qapp
    module = Behringer182Module()
    _connect_signal(
        module.clock_input,
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )

    module.process_runtime(
        4,
        {
            "gates": "1,1",
            "steps": "2",
            "direction": "forward",
            "gate_length": 1.0,
            "cv_a_range": 4.0,
            "cv_b_range": 2.0,
            "bpm": 60.0,
            "division": "1/4",
            "running": True,
        },
    )

    assert [led.is_on() for led in module.step_leds] == [
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_behringer_182_module_stops_gate_outputs(qapp: Any):
    del qapp
    module = Behringer182Module()
    _connect_signal(
        module.clock_input,
        np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    )

    module.process_runtime(
        4,
        {
            "cv_a_1": 0.25,
            "cv_b_1": 0.75,
            "gates": "1,1",
            "steps": "2",
            "direction": "forward",
            "gate_length": 1.0,
            "cv_a_range": 4.0,
            "cv_b_range": 2.0,
            "bpm": 60.0,
            "division": "1/4",
            "running": False,
        },
    )

    np.testing.assert_allclose(module.cv_a_port.value, [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(module.cv_b_port.value, [1.5, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(module.gate_port.value, 0.0)
    np.testing.assert_allclose(module.trigger_port.value, 0.0)
    np.testing.assert_allclose(module.end_port.value, 0.0)


def test_behringer_182_cv_knob_changes_without_resetting_step(qapp: Any):
    del qapp
    module = Behringer182Module()
    clock_output = _connect_signal(
        module.clock_input,
        np.array([1.0, 0.0, 1.0], dtype=np.float32),
    )

    module.process_runtime(
        3,
        {
            "cv_a_1": 0.0,
            "cv_a_2": 0.25,
            "gates": "1,1",
            "steps": "2",
            "direction": "forward",
            "gate_length": 1.0,
            "cv_a_range": 4.0,
            "cv_b_range": 2.0,
            "bpm": 60.0,
            "division": "1/4",
            "running": True,
        },
    )

    clock_output.write(np.zeros(2, dtype=np.float32))
    module.process_runtime(
        2,
        {
            "cv_a_1": 0.0,
            "cv_a_2": 0.75,
            "gates": "1,1",
            "steps": "2",
            "direction": "forward",
            "gate_length": 1.0,
            "cv_a_range": 4.0,
            "cv_b_range": 2.0,
            "bpm": 60.0,
            "division": "1/4",
            "running": True,
        },
    )

    np.testing.assert_allclose(module.cv_a_port.value, [3.0, 3.0])


def test_slide_module_processes_frequency_cv(qapp: Any):
    del qapp
    module = SlideModule()
    _connect_signal(
        module.freq_input,
        np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )
    _connect_signal(
        module.slide_input,
        np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )

    module.process_runtime(4, {"time": 1.0, "always_on": False})

    output = np.asarray(module.freq_output.value)
    assert output.shape == (4,)
    assert output[0] == 0.0
    assert 0.0 < output[1] < 1.0


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
