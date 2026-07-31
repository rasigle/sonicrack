"""Tests for the on-screen MIDI keyboard module."""

from typing import Any

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from soniclab import Volume
from soniclab.midi_io import midi_note_to_pitch_cv, midi_to_frequency

from sonicrack.gui.modules.input.midi_keyboard import MIDIKeyboardModule
from sonicrack.gui.modules.modifier.vca import VCAModule
from sonicrack.gui.modules.modulated_source.envelope_adsr import ADSRModule
from sonicrack.gui.modules.source.vco import ModulatedOscillatorModule
from sonicrack.patching.module import ModuleCategory
from sonicrack.patching.port import Port, PortSignal
from sonicrack.patching.registry import initialize_module_registry


def test_midi_keyboard_category_is_midi(qapp: Any):
    del qapp
    module = MIDIKeyboardModule()
    assert module.metadata.category == ModuleCategory.MIDI


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
    assert "C4 (60)" in module.note_label.text()


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
    assert "C4 (60)" in module.note_label.text()
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
    assert "C4 (60)" in module.note_label.text()


def test_midi_keyboard_polyphonic_input_uses_note_stack(qapp: Any):
    """Multiple simultaneous keys stay held; priority selects the driving note."""
    del qapp
    module = MIDIKeyboardModule()
    module.octave_combo.setCurrentText("4")
    module.priority_param.set_value("Last")

    module._note_on(0)  # C4 = 60
    module._note_on(4)  # E4 = 64
    module._note_on(7)  # G4 = 67

    assert set(module.cv_converter.held_notes) == {60, 64, 67}
    assert module.cv_converter.current_note == 67

    module.priority_param.set_value("Low")
    assert module.cv_converter.current_note == 60

    module._note_off(0)
    assert module.cv_converter.current_note == 64
    assert module.cv_converter.gate == pytest.approx(1.0)


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
    expected_trigger = np.zeros(8, dtype=np.float32)
    expected_trigger[0] = 1.0
    np.testing.assert_allclose(module.trigger_port.value, expected_trigger)
    np.testing.assert_allclose(
        module.vel_port.value,
        np.full(8, 100 / 127, dtype=np.float32),
    )


def test_midi_keyboard_trigger_is_one_sample_pulse_on_note_on(qapp: Any):
    del qapp
    module = MIDIKeyboardModule()

    assert module.trigger_port.signal == PortSignal.TRIGGER

    module.octave_combo.setCurrentText("4")
    module._note_on(0)
    module.process_runtime(8, {})

    expected = np.zeros(8, dtype=np.float32)
    expected[0] = 1.0
    np.testing.assert_allclose(module.trigger_port.value, expected)

    # Subsequent buffers stay low while the note remains held.
    module.process_runtime(8, {})
    np.testing.assert_allclose(module.trigger_port.value, np.zeros(8, dtype=np.float32))


def test_midi_keyboard_legato_note_change_rearms_trigger(qapp: Any):
    del qapp
    module = MIDIKeyboardModule()

    module.octave_combo.setCurrentText("4")
    module._note_on(0)
    module.process_runtime(4, {})

    # Second note while first is held keeps gate high but fires a new trigger.
    module._note_on(2)
    assert module.cv_converter.gate == pytest.approx(1.0)
    module.process_runtime(4, {})

    expected = np.zeros(4, dtype=np.float32)
    expected[0] = 1.0
    np.testing.assert_allclose(module.trigger_port.value, expected)
    np.testing.assert_allclose(module.gate_port.value, np.ones(4, dtype=np.float32))


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


def test_vca_runtime_silence_without_input(qapp: Any):
    del qapp
    vca = VCAModule()

    vca.process_runtime(16, {"amplitude": 1.0})

    assert np.allclose(vca.out_port.value, np.zeros(16, dtype=np.float32))


def test_vca_manual_gain_scales_audio_without_cv(qapp: Any):
    del qapp
    vca = VCAModule()
    source = Port("output", "Audio")
    source.write(np.full(8, 0.5, dtype=np.float32))
    source.connect(vca.in_port)

    vca.process_runtime(8, {"amplitude": 0.4})

    np.testing.assert_allclose(
        vca.out_port.value,
        np.full(8, 0.2, dtype=np.float32),
    )


def test_vca_update_knob_state_updates_tooltips(qapp: Any):
    del qapp
    vca = VCAModule()

    vca.update_knob_state()
    assert "when CV input is connected" in vca.cv_attn_knob.toolTip()
    assert "Final VCA output gain" in vca.gain_knob.toolTip()
    assert not vca.cv_attn_knob.isEnabled()

    # update_knob_state inspects PortWidget cables, not model Port.connect().
    from unittest.mock import MagicMock

    from sonicrack.gui.widgets.cable_widget import Cable

    cv_widget = next(p for p in vca.input_ports if p.port_name == "CV In")
    cv_widget.cables.append(MagicMock(spec=Cable))
    vca.update_knob_state()
    assert "0 = Gain only" in vca.cv_attn_knob.toolTip()
    assert "after CV modulation" in vca.gain_knob.toolTip()
    assert vca.cv_attn_knob.isEnabled()


def test_vca_ports_carry_signal_types(qapp: Any):
    del qapp
    vca = VCAModule()

    assert vca.in_port.signal == PortSignal.AUDIO
    assert vca.out_port.signal == PortSignal.AUDIO
    assert vca.cv_port.signal == PortSignal.CONTROL_CV


def test_vca_gain_changes_are_dezippered_across_buffers(qapp: Any):
    del qapp
    vca = VCAModule()
    source = Port("output", "Audio")
    source.write(np.ones(8, dtype=np.float32))
    source.connect(vca.in_port)

    vca.process_runtime(8, {"amplitude": 0.0})
    first = np.asarray(vca.out_port.value)
    source.write(np.ones(8, dtype=np.float32))
    vca.process_runtime(8, {"amplitude": 1.0})
    second = np.asarray(vca.out_port.value)

    assert np.allclose(first, 0.0)
    # Second buffer must ramp from previous gain, not jump immediately to 1.0.
    assert second[0] < 0.2
    assert second[-1] == pytest.approx(1.0)
    assert np.all(np.diff(second) > 0.0)


def test_vca_required_and_modulation_ports(qapp: Any):
    del qapp
    vca = VCAModule()

    assert vca.get_required_inputs() == ["In"]
    assert vca.get_modulation_inputs() == ["CV In"]
    assert vca.get_cv_range() == (0.0, 1.0)


def test_vca_create_engine_component_with_cv_influence(qapp: Any):
    del qapp
    from soniclab.dsp.modifiers import ModulatedVolume

    class _CvSource:
        def __init__(self):
            self.ended = False

        def __iter__(self):
            return self

        def __next__(self):
            return 0.5

        def get_samples(self, num_samples: int, **_kwargs):
            return np.full(num_samples, 0.5, dtype=np.float32)

        def get_samples_vectorized(self, num_samples: int):
            return np.full(num_samples, 0.5, dtype=np.float32)

        def trigger_release(self):
            self.ended = True

    vca = VCAModule()
    vca.gain_knob.set_value(0.8)
    vca.cv_attn_knob.set_value(0.5)
    cv_source = _CvSource()

    component = vca.create_engine_component(modulation_components={"CV In": cv_source})

    assert isinstance(component, ModulatedVolume)
    # Adapter blends gain with CV and exposes iterator / bulk sample APIs.
    adapter = component.modulator
    assert float(next(iter(adapter))) == pytest.approx(0.6)
    samples = adapter.get_samples(4)
    assert samples.shape == (4,)
    assert np.allclose(samples, 0.6)
    adapter.trigger_release()
    assert cv_source.ended is True
    assert adapter.ended is True
