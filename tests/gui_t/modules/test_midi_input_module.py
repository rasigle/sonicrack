"""Tests for the MIDI Input module CV/trigger outputs."""

from typing import Any

import numpy as np
import pytest
from soniclab.midi_io import (
    ControlChangeMessage,
    NoteOffMessage,
    NoteOnMessage,
    PitchBendMessage,
    midi_note_to_pitch_cv,
)

from sonicrack.gui.modules.input.midi_input import MIDIInputModule
from sonicrack.patching.module import ModuleCategory
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import initialize_module_registry


def test_midi_input_category_is_midi(qapp: Any):
    del qapp
    module = MIDIInputModule()
    assert module.metadata.category == ModuleCategory.MIDI


def test_midi_input_outputs_include_trigger_and_controllers(qapp: Any):
    del qapp
    module = MIDIInputModule()

    assert module.trigger_port.signal == PortSignal.TRIGGER
    assert module.mod_port.signal == PortSignal.CONTROL_CV
    assert module.expr_port.signal == PortSignal.CONTROL_CV
    assert module.bend_port.signal == PortSignal.CONTROL_CV
    assert module.get_output_component("Trig") is module.trigger_output
    assert module.get_output_component("Mod") is module.mod_output


def test_midi_input_note_on_arms_one_sample_trigger(qapp: Any):
    del qapp
    module = MIDIInputModule()

    module._on_midi_message(
        NoteOnMessage(timestamp=0.0, channel=0, note=60, velocity=100)
    )
    module.process_runtime(8, {})

    expected_trigger = np.zeros(8, dtype=np.float32)
    expected_trigger[0] = 1.0
    np.testing.assert_allclose(module.trigger_port.value, expected_trigger)
    np.testing.assert_allclose(module.gate_port.value, np.ones(8, dtype=np.float32))
    np.testing.assert_allclose(
        module.freq_port.value,
        np.full(8, midi_note_to_pitch_cv(60), dtype=np.float32),
    )
    np.testing.assert_allclose(
        module.vel_port.value,
        np.full(8, 100 / 127, dtype=np.float32),
    )

    # Held notes do not keep pulsing.
    module.process_runtime(8, {})
    np.testing.assert_allclose(
        module.trigger_port.value, np.zeros(8, dtype=np.float32)
    )


def test_midi_input_legato_note_change_rearms_trigger(qapp: Any):
    del qapp
    module = MIDIInputModule()

    module._on_midi_message(
        NoteOnMessage(timestamp=0.0, channel=0, note=60, velocity=100)
    )
    module.process_runtime(4, {})

    module._on_midi_message(
        NoteOnMessage(timestamp=0.0, channel=0, note=64, velocity=90)
    )
    assert module.cv_converter.gate == pytest.approx(1.0)
    module.process_runtime(4, {})

    expected = np.zeros(4, dtype=np.float32)
    expected[0] = 1.0
    np.testing.assert_allclose(module.trigger_port.value, expected)


def test_midi_input_note_off_does_not_arm_trigger(qapp: Any):
    del qapp
    module = MIDIInputModule()

    module._on_midi_message(
        NoteOnMessage(timestamp=0.0, channel=0, note=60, velocity=100)
    )
    module.process_runtime(4, {})
    module._on_midi_message(NoteOffMessage(timestamp=0.0, channel=0, note=60))
    module.process_runtime(4, {})

    np.testing.assert_allclose(
        module.trigger_port.value, np.zeros(4, dtype=np.float32)
    )
    assert module.cv_converter.gate == pytest.approx(0.0)


def test_midi_input_polyphonic_stack_last_priority(qapp: Any):
    """Multiple simultaneous notes use MIDIToCV note-stack (last priority)."""
    del qapp
    module = MIDIInputModule()
    module.priority_param.set_value("Last")

    module._on_midi_message(
        NoteOnMessage(timestamp=0.0, channel=0, note=60, velocity=100)
    )
    module._on_midi_message(
        NoteOnMessage(timestamp=0.0, channel=0, note=64, velocity=80)
    )
    module._on_midi_message(
        NoteOnMessage(timestamp=0.0, channel=0, note=67, velocity=90)
    )

    assert module.cv_converter.current_note == 67
    assert set(module.cv_converter.held_notes) == {60, 64, 67}

    # Release top note → falls back to previous last note (64)
    module._on_midi_message(NoteOffMessage(timestamp=0.0, channel=0, note=67))
    assert module.cv_converter.current_note == 64
    assert module.cv_converter.gate == pytest.approx(1.0)


def test_midi_input_high_priority_selects_highest_note(qapp: Any):
    del qapp
    module = MIDIInputModule()
    module.priority_param.set_value("High")

    module._on_midi_message(
        NoteOnMessage(timestamp=0.0, channel=0, note=64, velocity=100)
    )
    module._on_midi_message(
        NoteOnMessage(timestamp=0.0, channel=0, note=60, velocity=100)
    )
    # High priority keeps 64 even though 60 was pressed later
    assert module.cv_converter.current_note == 64

    module._on_midi_message(
        NoteOnMessage(timestamp=0.0, channel=0, note=72, velocity=100)
    )
    assert module.cv_converter.current_note == 72


def test_midi_input_mod_wheel_and_expression_cv(qapp: Any):
    del qapp
    module = MIDIInputModule()

    module._on_midi_message(
        ControlChangeMessage(timestamp=0.0, channel=0, controller=1, value=64)
    )
    module._on_midi_message(
        ControlChangeMessage(timestamp=0.0, channel=0, controller=11, value=127)
    )
    module.process_runtime(4, {})

    np.testing.assert_allclose(
        module.mod_port.value, np.full(4, 64 / 127, dtype=np.float32), atol=1e-5
    )
    np.testing.assert_allclose(
        module.expr_port.value, np.ones(4, dtype=np.float32), atol=1e-5
    )


def test_midi_input_pitch_bend_affects_pitch_and_bend_cv(qapp: Any):
    del qapp
    module = MIDIInputModule()
    module.bend_range_knob.set_value(2)
    module._on_bend_range_changed()

    module._on_midi_message(
        NoteOnMessage(timestamp=0.0, channel=0, note=60, velocity=100)
    )
    # Full up bend (+8191) ≈ +2 semitones with range 2
    module._on_midi_message(PitchBendMessage(timestamp=0.0, channel=0, value=8191))
    module.process_runtime(4, {})

    expected_pitch = midi_note_to_pitch_cv(62.0)  # ~C4 + 2 st
    np.testing.assert_allclose(
        module.freq_port.value,
        np.full(4, expected_pitch, dtype=np.float32),
        atol=0.02,
    )
    assert float(module.bend_port.value[0]) == pytest.approx(1.0, abs=0.01)


def test_midi_input_is_registered_under_midi_category(qapp: Any):
    del qapp
    registry = initialize_module_registry()
    assert registry.get("MIDI Input") is MIDIInputModule
    by_cat = registry.get_by_category(ModuleCategory.MIDI)
    assert "MIDI Input" in by_cat


def test_midi_input_device_selection_controls_worker_without_buttons(qapp: Any):
    """Selecting a device auto-starts; clearing it stops. No Start/Refresh buttons."""
    del qapp
    module = MIDIInputModule()
    assert not hasattr(module, "start_btn")
    assert not hasattr(module, "refresh_btn")

    # No device selected → idle
    assert module.device_combo.currentText() == "(No Device)"
    assert module._is_running is False

    # Sync with no device is a no-op (does not error)
    module._sync_midi_device("(No Device)")
    assert module._is_running is False

    # Selecting a missing/virtual name still attempts start path safely when
    # mocked: stop path after a fake running state.
    module._is_running = True
    module.midi_worker = None
    module._sync_midi_device("(No Device)")
    assert module._is_running is False
