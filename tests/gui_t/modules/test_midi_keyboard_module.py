"""Tests for the on-screen MIDI keyboard module."""

from typing import Any

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent

from src.engine import Volume
from src.gui.core.port import Port
from src.gui.module_registry import initialize_module_registry
from src.gui.modules.input.midi_keyboard import MIDIKeyboardModule
from src.gui.modules.modifier.vca import VCAModule
from src.gui.modules.modulated_source.envelope_adsr import ADSRModule
from src.gui.modules.source.vco import ModulatedOscillatorModule
from src.midi_io import midi_note_to_pitch_cv, midi_to_frequency


def test_midi_keyboard_note_press_updates_cv_outputs(qapp: Any):
    del qapp
    module = MIDIKeyboardModule()

    module.octave_combo.setCurrentText("4")
    module.velocity_knob.set_value(64)
    module._note_on(0)

    assert module.cv_converter.current_note == 60
    assert module.cv_converter.gate == pytest.approx(1.0)
    assert module.cv_converter.pitch_cv == pytest.approx(midi_note_to_pitch_cv(60))
    assert module.cv_converter.frequency == pytest.approx(midi_to_frequency(60))
    assert module.cv_converter.velocity == pytest.approx(64 / 127)
    assert module.note_label.text() == "C4 (60)"


def test_midi_keyboard_note_release_clears_gate(qapp: Any):
    del qapp
    module = MIDIKeyboardModule()

    module.octave_combo.setCurrentText("4")
    module._note_on(7)
    module._note_off(7)

    assert module.cv_converter.gate == pytest.approx(0.0)
    assert module.note_label.text() == "--"


def test_midi_keyboard_computer_keys_trigger_notes(qapp: Any):
    del qapp
    module = MIDIKeyboardModule()

    module.octave_combo.setCurrentText("4")
    module.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_A.value,
            Qt.KeyboardModifier.NoModifier,
        )
    )

    assert module.cv_converter.current_note == 60
    assert module.cv_converter.gate == pytest.approx(1.0)
    assert module.cv_converter.pitch_cv == pytest.approx(midi_note_to_pitch_cv(60))
    assert module.cv_converter.frequency == pytest.approx(midi_to_frequency(60))
    assert module.note_label.text() == "C4 (60)"
    assert module.key_buttons[0].isDown()

    module.keyReleaseEvent(
        QKeyEvent(
            QEvent.Type.KeyRelease,
            Qt.Key.Key_A.value,
            Qt.KeyboardModifier.NoModifier,
        )
    )

    assert module.cv_converter.gate == pytest.approx(0.0)
    assert module.note_label.text() == "--"
    assert not module.key_buttons[0].isDown()


def test_midi_keyboard_computer_keys_keep_last_held_note(qapp: Any):
    del qapp
    module = MIDIKeyboardModule()

    module.octave_combo.setCurrentText("4")
    module.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_A.value,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    module.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_S.value,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    module.keyReleaseEvent(
        QKeyEvent(
            QEvent.Type.KeyRelease,
            Qt.Key.Key_S.value,
            Qt.KeyboardModifier.NoModifier,
        )
    )

    assert module.cv_converter.current_note == 60
    assert module.cv_converter.gate == pytest.approx(1.0)
    assert module.note_label.text() == "C4 (60)"


def test_midi_keyboard_runtime_writes_frequency_gate_velocity(qapp: Any):
    del qapp
    module = MIDIKeyboardModule()

    module.octave_combo.setCurrentText("4")
    module.velocity_knob.set_value(100)
    module._note_on(9)
    module.process_runtime(8, {})

    np.testing.assert_allclose(
        module.freq_port.value,
        np.full(8, midi_note_to_pitch_cv(69), dtype=np.float32),
    )
    np.testing.assert_allclose(module.gate_port.value, np.ones(8, dtype=np.float32))
    np.testing.assert_allclose(
        module.vel_port.value,
        np.full(8, 100 / 127, dtype=np.float32),
    )


def test_midi_keyboard_is_registered(qapp: Any):
    del qapp
    registry = initialize_module_registry()

    assert registry.get("MIDI Keyboard") is MIDIKeyboardModule


def test_midi_keyboard_frequency_can_drive_vco_runtime(qapp: Any):
    del qapp
    keyboard = MIDIKeyboardModule()
    vco = ModulatedOscillatorModule()

    keyboard.octave_combo.setCurrentText("4")
    keyboard._note_on(0)
    keyboard.process_runtime(128, {})
    keyboard.freq_port.connect(vco.freq_input)

    vco.process_runtime(
        128,
        {
            "waveform": "Sine",
            "mode": "analog",
            "frequency": midi_to_frequency(60),
            "gain_db": -12.0,
            "phase": 0.0,
        },
    )

    output = np.asarray(vco.out_port.value)
    assert output.shape == (128,)
    assert np.any(output != 0.0)
    assert vco.component.frequency == pytest.approx(midi_to_frequency(60), rel=0.01)


def test_midi_keyboard_frequency_can_drive_vco_vectorized_runtime(qapp: Any):
    del qapp
    keyboard = MIDIKeyboardModule()
    vco = ModulatedOscillatorModule()

    keyboard.octave_combo.setCurrentText("4")
    keyboard._note_on(0)
    keyboard.process_runtime(1024, {})
    keyboard.freq_port.connect(vco.freq_input)

    vco.process_runtime(
        1024,
        {
            "waveform": "Sine",
            "mode": "analog",
            "frequency": 440.0,
            "gain_db": -12.0,
            "phase": 0.0,
        },
    )

    output = np.asarray(vco.out_port.value)
    assert output.shape == (1024,)
    assert np.any(output != 0.0)


def test_midi_keyboard_vco_frequency_is_continuous_across_buffers(qapp: Any):
    del qapp
    single_buffer_keyboard = MIDIKeyboardModule()
    single_buffer_vco = ModulatedOscillatorModule()
    split_buffer_keyboard = MIDIKeyboardModule()
    split_buffer_vco = ModulatedOscillatorModule()
    parameters = {
        "waveform": "Sine",
        "mode": "analog",
        "frequency": 440.0,
        "gain_db": -12.0,
        "phase": 0.0,
    }

    single_buffer_keyboard.octave_combo.setCurrentText("4")
    single_buffer_keyboard._note_on(0)
    single_buffer_keyboard.process_runtime(2048, {})
    single_buffer_keyboard.freq_port.connect(single_buffer_vco.freq_input)
    single_buffer_vco.process_runtime(2048, parameters)
    expected = np.asarray(single_buffer_vco.out_port.value)

    split_buffer_keyboard.octave_combo.setCurrentText("4")
    split_buffer_keyboard._note_on(0)
    split_buffer_keyboard.process_runtime(1024, {})
    split_buffer_keyboard.freq_port.connect(split_buffer_vco.freq_input)
    split_buffer_vco.process_runtime(1024, parameters)
    first = np.asarray(split_buffer_vco.out_port.value)
    split_buffer_keyboard.process_runtime(1024, {})
    split_buffer_vco.process_runtime(1024, parameters)
    second = np.asarray(split_buffer_vco.out_port.value)

    actual = np.concatenate((first, second))
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_midi_keyboard_gate_can_trigger_adsr_runtime(qapp: Any):
    del qapp
    keyboard = MIDIKeyboardModule()
    adsr = ADSRModule()

    keyboard.octave_combo.setCurrentText("4")
    keyboard._note_on(0)
    keyboard.process_runtime(16, {})
    keyboard.gate_port.connect(adsr.gate_input)

    adsr.process_runtime(
        16,
        {
            "attack_duration": 4 / 44100,
            "decay_duration": 4 / 44100,
            "sustain_level": 0.5,
            "release_duration": 4 / 44100,
            "retrigger_mode": "Punch",
        },
    )

    output = np.asarray(adsr.out_port.value)
    assert output.shape == (16,)
    assert np.any(output > 0.0)


def test_midi_keyboard_velocity_can_drive_vca_runtime(qapp: Any):
    del qapp
    keyboard = MIDIKeyboardModule()
    vca = VCAModule()
    source = Port("output", "Audio")

    keyboard.octave_combo.setCurrentText("4")
    keyboard.velocity_knob.set_value(100)
    keyboard._note_on(0)
    keyboard.process_runtime(8, {})

    source.write(np.ones(8, dtype=np.float32))
    source.connect(vca.in_port)
    keyboard.vel_port.connect(vca.cv_port)

    vca.process_runtime(8, {"amplitude": 0.25})

    np.testing.assert_allclose(
        vca.out_port.value,
        np.full(8, 0.25 * (100 / 127), dtype=np.float32),
    )


def test_vca_cv_attenuation_blends_gain_and_cv_runtime(qapp: Any):
    del qapp
    vca = VCAModule()
    source = Port("output", "Audio")
    cv = Port("output", "CV")

    source.write(np.ones(8, dtype=np.float32))
    cv.write(np.full(8, 0.75, dtype=np.float32))
    source.connect(vca.in_port)
    cv.connect(vca.cv_port)

    vca.process_runtime(8, {"amplitude": 0.25, "cv_attenuation": 0.5})

    np.testing.assert_allclose(
        vca.out_port.value,
        np.full(8, 0.21875, dtype=np.float32),
    )


def test_vca_zero_cv_attenuation_uses_manual_gain_runtime(qapp: Any):
    del qapp
    vca = VCAModule()
    source = Port("output", "Audio")
    cv = Port("output", "CV")

    source.write(np.ones(8, dtype=np.float32))
    cv.write(np.full(8, 0.9, dtype=np.float32))
    source.connect(vca.in_port)
    cv.connect(vca.cv_port)

    vca.process_runtime(8, {"amplitude": 0.3, "cv_attenuation": 0.0})

    np.testing.assert_allclose(
        vca.out_port.value,
        np.full(8, 0.3, dtype=np.float32),
    )


def test_vca_registers_cv_attenuation_parameter(qapp: Any):
    del qapp
    vca = VCAModule()
    vca.cv_attn_knob.set_value(0.4)

    assert vca.get_parameters()["cv_attenuation"] == pytest.approx(0.4)


def test_vca_zero_cv_attenuation_component_uses_manual_gain(qapp: Any):
    del qapp
    vca = VCAModule()
    vca.gain_knob.set_value(0.4)
    vca.cv_attn_knob.set_value(0.0)

    component = vca.create_engine_component(modulation_components={"CV In": object()})

    assert isinstance(component, Volume)
    assert component.amplitude == pytest.approx(0.4)
