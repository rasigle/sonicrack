"""Tests for the polyphonic MIDI → CV module."""

from typing import Any

import numpy as np
import pytest
from soniclab.midi_io import midi_note_to_pitch_cv

from sonicrack.gui.modules.input.midi_poly_cv import MIDIPolyCVModule
from sonicrack.patching.module import ModuleCategory
from sonicrack.patching.port import PortSignal
from sonicrack.patching.registry import initialize_module_registry


def test_midi_poly_cv_category_and_registration(qapp: Any):
    del qapp
    module = MIDIPolyCVModule()
    assert module.metadata.category == ModuleCategory.MIDI

    registry = initialize_module_registry()
    assert registry.get("MIDI Poly CV") is MIDIPolyCVModule
    assert "MIDI Poly CV" in registry.get_by_category(ModuleCategory.MIDI)


def test_midi_poly_cv_independent_voice_outputs(qapp: Any):
    del qapp
    module = MIDIPolyCVModule()

    module.note_on(60, 100)
    module.note_on(64, 80)
    module.note_on(67, 90)
    assert module.poly_cv.active_voice_count == 3

    module.process_runtime(8, {})

    pitches = sorted(
        float(np.asarray(module.voice_pitch_ports[i].value)[0]) for i in range(3)
    )
    expected = sorted(float(midi_note_to_pitch_cv(n)) for n in (60, 64, 67))
    for got, want in zip(pitches, expected, strict=True):
        assert got == pytest.approx(want, abs=1e-5)

    gates = [float(np.asarray(module.voice_gate_ports[i].value)[0]) for i in range(3)]
    assert all(g == pytest.approx(1.0) for g in gates)

    # Fourth voice free
    np.testing.assert_allclose(
        module.voice_gate_ports[3].value, np.zeros(8, dtype=np.float32)
    )


def test_midi_poly_cv_trigger_on_note_on(qapp: Any):
    del qapp
    module = MIDIPolyCVModule()
    assert module.trig_port.signal == PortSignal.TRIGGER

    module.note_on(60, 100)
    module.process_runtime(4, {})
    expected = np.zeros(4, dtype=np.float32)
    expected[0] = 1.0
    np.testing.assert_allclose(module.trig_port.value, expected)

    module.process_runtime(4, {})
    np.testing.assert_allclose(module.trig_port.value, np.zeros(4, dtype=np.float32))


def test_midi_poly_cv_note_off_frees_voice(qapp: Any):
    del qapp
    module = MIDIPolyCVModule()
    module.note_on(60, 100)
    module.note_on(64, 100)
    module.note_off(60)
    assert module.poly_cv.active_voice_count == 1
    assert module.poly_cv.get_playing_notes() == [64]


def test_midi_poly_cv_get_output_component(qapp: Any):
    del qapp
    module = MIDIPolyCVModule()
    assert module.get_output_component("V1 1V/Oct") is module.voice_pitch_outputs[0]
    assert module.get_output_component("V2 Gate") is module.voice_gate_outputs[1]
    assert module.get_output_component("Mod") is module.mod_output
