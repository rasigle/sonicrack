"""Tests for the MIDI Input module CV/trigger outputs."""

from typing import Any

import numpy as np
import pytest
from soniclab.midi_io import NoteOffMessage, NoteOnMessage, midi_note_to_pitch_cv

from sonicrack.gui.modules.input.midi_input import MIDIInputModule
from sonicrack.patching.port import PortSignal


def test_midi_input_outputs_include_trigger(qapp: Any):
    del qapp
    module = MIDIInputModule()

    assert module.trigger_port.signal == PortSignal.TRIGGER
    assert module.get_output_component("Trig") is module.trigger_output


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
