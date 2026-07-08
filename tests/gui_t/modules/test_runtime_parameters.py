"""Runtime parameter snapshot coverage for GUI modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from soniclab import PITCH_CV_REFERENCE_FREQUENCY, frequency_to_pitch_cv
from soniclab.dsp.modulators import ADSREnvelope, DecayEnvelope
from soniclab.generators.oscillators.oscillator_ramp import SawtoothOscillator
from soniclab.generators.oscillators.oscillator_sine import SineOscillator
from soniclab.generators.oscillators.oscillator_square import SquareOscillator

from src.constants import DEFAULT_SAMPLE_RATE
from src.gui.core.port import Port
from src.gui.modules.effects.effects_compressor import CompressorModule
from src.gui.modules.effects.effects_distortion import DistortionModule
from src.gui.modules.mixer import MixerModule
from src.gui.modules.modifier.acid_filter import AcidFilterModule
from src.gui.modules.modifier.filter import FilterModule
from src.gui.modules.modulated_source.envelope_adsr import ADSRModule
from src.gui.modules.modulated_source.envelope_decay import DecayEnvelopeModule
from src.gui.modules.source._oscillator_runtime import (
    _frequency_slew_values,
    render_with_frequency_ramp,
)
from src.gui.modules.source.lfo import LFOModule
from src.gui.modules.source.oscillator import OscillatorModule
from src.gui.modules.source.vco import ModulatedOscillatorModule
from src.gui.modules.voice.tb303_voice import TB303VoiceModule
from src.gui.ui_constants import AUDIO_FREQUENCY_KNOB_CURVE


def _connect_constant_input(input_port: Port, value: float = 1.0) -> Port:
    output_port = Port("output", "Test Out")
    output_port.write(np.full(8, value, dtype=np.float32))
    output_port.connect(input_port)
    return output_port


def test_oscillator_runtime_applies_frequency_and_pulsewidth(qapp: Any):
    del qapp
    module = OscillatorModule()

    module.process_runtime(8, {"frequency": 880.0, "pulsewidth": 0.25})

    assert 120.0 < module._sine_oscillator.frequency < 880.0
    assert 120.0 < module._triangle_oscillator.frequency < 880.0
    assert 120.0 < module._sawtooth_oscillator.frequency < 880.0
    assert 120.0 < module._square_oscillator.frequency < 880.0
    assert module._square_oscillator.pulsewidth == pytest.approx(0.25)


def test_audio_rate_frequency_knobs_use_custom_curve(qapp: Any):
    oscillator = OscillatorModule()
    vco = ModulatedOscillatorModule()

    assert oscillator.freq_knob.curve_points == tuple(AUDIO_FREQUENCY_KNOB_CURVE)
    assert oscillator.freq_knob.min_value == pytest.approx(11.0)
    assert oscillator.freq_knob.max_value == pytest.approx(6000.0)
    assert vco.freq_knob.curve_points == tuple(AUDIO_FREQUENCY_KNOB_CURVE)
    assert vco.freq_knob.min_value == pytest.approx(11.0)
    assert vco.freq_knob.max_value == pytest.approx(6000.0)


def test_lfo_frequency_knob_uses_logarithmic_pitch_control(qapp: Any):
    del qapp
    module = LFOModule()

    assert module.freq_knob.logarithmic is True
    assert module.freq_knob.min_value == pytest.approx(0.01)
    assert module.freq_knob.max_value == pytest.approx(20.0)


def test_lfo_runtime_applies_frequency_and_pulsewidth(qapp: Any):
    del qapp
    module = LFOModule()

    module.process_runtime(8, {"frequency": 4.0, "pulsewidth": 0.75})

    assert 1.0 < module._sine_oscillator.frequency < 4.0
    assert 1.0 < module._triangle_oscillator.frequency < 4.0
    assert 1.0 < module._sawtooth_oscillator.frequency < 4.0
    assert 1.0 < module._square_oscillator.frequency < 4.0
    assert module._square_oscillator.pulsewidth == pytest.approx(0.75)


def test_lfo_clock_input_resets_cycle_on_trigger(qapp: Any):
    del qapp
    module = LFOModule()
    clock_source = _connect_constant_input(module.clock_input, 0.0)
    clock_source.write(
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    module.process_runtime(8, {"frequency": 1.0, "pulsewidth": 0.5})

    output = np.asarray(module.sine_port.value)
    assert module._sine_oscillator._sample_index == 4
    assert abs(float(output[4] - output[3])) < 1e-4


def test_lfo_clock_reset_is_smoothed_across_buffer_boundary(qapp: Any):
    del qapp
    module = LFOModule()
    clock_source = _connect_constant_input(module.clock_input, 0.0)
    clock_source.write(np.zeros(128, dtype=np.float32))

    module.process_runtime(128, {"frequency": 20.0, "pulsewidth": 0.5})
    first = np.asarray(module.sawtooth_port.value)

    clock_source.write(
        np.concatenate(
            (
                np.ones(1, dtype=np.float32),
                np.zeros(127, dtype=np.float32),
            )
        )
    )
    module.process_runtime(128, {"frequency": 20.0, "pulsewidth": 0.5})
    second = np.asarray(module.sawtooth_port.value)

    assert second[0] == pytest.approx(first[-1], abs=1e-7)


def test_oscillator_frequency_changes_are_ramped_across_buffer(qapp: Any):
    del qapp
    module = OscillatorModule()

    module.process_runtime(128, {"frequency": 120.0, "pulsewidth": 0.5})
    first = np.asarray(module.triangle_port.value)
    module.process_runtime(128, {"frequency": 2000.0, "pulsewidth": 0.5})
    second = np.asarray(module.triangle_port.value)

    boundary_jump = abs(float(second[0] - first[-1]))
    assert boundary_jump < 0.08
    assert module._triangle_oscillator.frequency < 2000.0
    assert module._triangle_oscillator.frequency > 120.0


def test_lfo_frequency_changes_are_ramped_across_buffer(qapp: Any):
    del qapp
    module = LFOModule()

    module.process_runtime(128, {"frequency": 1.0, "pulsewidth": 0.5})
    first = np.asarray(module.sine_port.value)
    module.process_runtime(128, {"frequency": 12.0, "pulsewidth": 0.5})
    second = np.asarray(module.sine_port.value)

    boundary_jump = abs(float(second[0] - first[-1]))
    assert boundary_jump < 0.01
    assert module._sine_oscillator.frequency < 12.0
    assert module._sine_oscillator.frequency > 1.0


def test_lfo_frequency_ramp_state_is_tracked_per_waveform(qapp: Any):
    del qapp
    module = LFOModule()

    module.process_runtime(128, {"frequency": 1.0, "pulsewidth": 0.5})
    module.process_runtime(128, {"frequency": 12.0, "pulsewidth": 0.5})

    for index, oscillator in enumerate(module.oscs):
        assert module._last_runtime_frequencies[index] == pytest.approx(
            oscillator.frequency
        )


def test_lfo_can_drive_vco_without_click_on_lfo_frequency_change(qapp: Any):
    del qapp
    lfo = LFOModule()
    vco = ModulatedOscillatorModule()
    lfo.sine_port.connect(vco.freq_input)
    parameters = {
        "waveform": "Sine",
        "mode": "analog",
        "frequency": 440.0,
        "gain_db": -12.0,
        "phase": 0.0,
    }

    lfo.process_runtime(512, {"frequency": 1.0, "pulsewidth": 0.5})
    vco.process_runtime(512, parameters)
    first = np.asarray(vco.out_port.value)

    lfo.process_runtime(512, {"frequency": 12.0, "pulsewidth": 0.5})
    vco.process_runtime(512, parameters)
    second = np.asarray(vco.out_port.value)

    boundary_jump = abs(float(second[0] - first[-1]))
    assert boundary_jump < 0.08


def test_vco_pitch_cv_input_is_dezippered(qapp: Any):
    del qapp
    module = ModulatedOscillatorModule()
    pitch_source = _connect_constant_input(module.freq_input, 0.0)
    parameters = {
        "waveform": "Sine",
        "mode": "analog",
        "frequency": 440.0,
        "gain_db": -12.0,
        "phase": 0.0,
    }

    pitch_source.write(np.zeros(32, dtype=np.float32))
    module.process_runtime(32, parameters)

    pitch_source.write(np.ones(32, dtype=np.float32))
    module.process_runtime(32, parameters)

    assert module.component.frequency < 880.0
    assert module.component.frequency > 440.0
    assert module._last_pitch_cv is not None
    assert 0.0 < module._last_pitch_cv < 1.0


def test_vco_v_oct_input_transposes_base_frequency(qapp: Any):
    del qapp
    module = ModulatedOscillatorModule()
    pitch_source = _connect_constant_input(module.freq_input, 0.0)
    parameters = {
        "waveform": "Sine",
        "mode": "analog",
        "frequency": 330.0,
        "gain_db": -12.0,
        "phase": 0.0,
    }

    module.process_runtime(8, parameters)

    assert module.component.frequency == pytest.approx(330.0)
    assert module.freq_knob.isEnabled()

    pitch_source.write(np.ones(8, dtype=np.float32))
    module._last_pitch_cv = None
    module.process_runtime(8, parameters)

    assert module.component.frequency == pytest.approx(660.0)


def test_vco_fm_input_defaults_to_vcv_exponential_mode(qapp: Any):
    del qapp
    module = ModulatedOscillatorModule()
    fm_source = _connect_constant_input(module.fm_input, 1.0)
    fm_source.write(np.ones(16, dtype=np.float32))

    assert module.fm_amount_knob.min_value == pytest.approx(-100.0)
    assert module.fm_amount_knob.max_value == pytest.approx(100.0)
    assert module.fm_mode_combo.currentText() == "1V/octave"

    module.process_runtime(
        16,
        {
            "waveform": "Sine",
            "mode": "analog",
            "frequency": 330.0,
            "gain_db": -12.0,
            "phase": 0.0,
            "fm_amount": 50.0,
        },
    )

    output = np.asarray(module.out_port.value)
    assert output.shape == (16,)
    assert np.all(np.isfinite(output))
    assert module.component.frequency == pytest.approx(330.0 * np.sqrt(2.0))
    assert module.get_modulation_inputs() == ["Gain", "FM"]
    assert module.get_cv_range("FM") == pytest.approx((-1.0, 1.0))

    module.process_runtime(
        16,
        {
            "waveform": "Sine",
            "mode": "analog",
            "frequency": 330.0,
            "gain_db": -12.0,
            "phase": 0.0,
            "fm_amount": -50.0,
        },
    )

    assert module.component.frequency == pytest.approx(330.0 / np.sqrt(2.0))


def test_vco_linear_fm_uses_vcv_c4_scaled_hz_offset(qapp: Any):
    del qapp
    module = ModulatedOscillatorModule()
    fm_source = _connect_constant_input(module.fm_input, 1.0)
    fm_source.write(np.ones(16, dtype=np.float32))

    module.process_runtime(
        16,
        {
            "waveform": "Sine",
            "mode": "analog",
            "fm_mode": "Linear",
            "frequency": 330.0,
            "gain_db": -12.0,
            "phase": 0.0,
            "fm_amount": 50.0,
        },
    )

    assert module.component.frequency == pytest.approx(
        330.0 + (PITCH_CV_REFERENCE_FREQUENCY * 0.5)
    )


def test_vco_lfo_fm_differs_from_v_oct_pitch_input(qapp: Any):
    del qapp
    lfo_for_pitch = LFOModule()
    lfo_for_fm = LFOModule()
    pitch_vco = ModulatedOscillatorModule()
    fm_vco = ModulatedOscillatorModule()
    parameters = {
        "waveform": "Sine",
        "mode": "analog",
        "fm_mode": "Linear",
        "frequency": 440.0,
        "gain_db": -12.0,
        "phase": 0.0,
        "fm_amount": 25.0,
    }

    lfo_parameters = {"frequency": 5.0, "pulsewidth": 0.5}
    lfo_for_pitch.sine_port.connect(pitch_vco.freq_input)
    lfo_for_fm.sine_port.connect(fm_vco.fm_input)

    lfo_for_pitch.process_runtime(1024, lfo_parameters)
    pitch_vco._last_pitch_cv = None
    pitch_vco.process_runtime(1024, parameters)
    pitch_output = np.asarray(pitch_vco.out_port.value)

    lfo_for_fm.process_runtime(1024, lfo_parameters)
    fm_vco.process_runtime(1024, parameters)
    fm_output = np.asarray(fm_vco.out_port.value)

    assert pitch_vco.get_runtime_spec().input_names == ("V/Oct", "FM", "Gain")
    assert fm_vco.component.frequency != pytest.approx(pitch_vco.component.frequency)
    assert np.max(np.abs(pitch_output - fm_output)) > 0.01


def test_lfo_frequency_changes_are_ramped_with_clock_input_connected(qapp: Any):
    del qapp
    module = LFOModule()
    clock_source = _connect_constant_input(module.clock_input, 0.0)
    clock_source.write(np.zeros(128, dtype=np.float32))

    module.process_runtime(128, {"frequency": 1.0, "pulsewidth": 0.5})
    first = np.asarray(module.sine_port.value)
    clock_source.write(np.zeros(128, dtype=np.float32))
    module.process_runtime(128, {"frequency": 12.0, "pulsewidth": 0.5})
    second = np.asarray(module.sine_port.value)

    boundary_jump = abs(float(second[0] - first[-1]))
    assert boundary_jump < 0.01
    assert module._sine_oscillator.frequency < 1.2
    assert module._sine_oscillator.frequency > 1.0


def test_frequency_slew_continues_across_buffers():
    frequencies = _frequency_slew_values(
        previous_frequency=120.0,
        target_frequency=2000.0,
        num_samples=128,
        sample_rate=44100.0,
        slew_time_ms=35.0,
    )

    assert np.all(np.diff(frequencies) > 0.0)
    assert frequencies[-1] < 2000.0
    assert frequencies[0] / 120.0 == pytest.approx(
        frequencies[1] / frequencies[0],
        rel=1e-3,
    )


def test_render_frequency_ramp_keeps_oscillator_at_smoothed_state():
    oscillator = SineOscillator(frequency=120.0, sample_rate=44100, gain_db=0)

    _, rendered_frequency = render_with_frequency_ramp(
        oscillator,
        previous_frequency=120.0,
        target_frequency=2000.0,
        num_samples=128,
    )

    assert rendered_frequency < 2000.0
    assert rendered_frequency > 120.0
    assert oscillator.frequency == pytest.approx(rendered_frequency)


def test_frequency_slew_is_continuous_when_split_across_buffers():
    continuous_osc = SawtoothOscillator(
        frequency=120.0,
        sample_rate=44100,
        gain_db=0,
        mode="analog",
    )
    split_osc = SawtoothOscillator(
        frequency=120.0,
        sample_rate=44100,
        gain_db=0,
        mode="analog",
    )

    continuous, _ = render_with_frequency_ramp(
        continuous_osc,
        previous_frequency=120.0,
        target_frequency=2000.0,
        num_samples=256,
    )

    first, rendered_frequency = render_with_frequency_ramp(
        split_osc,
        previous_frequency=120.0,
        target_frequency=2000.0,
        num_samples=128,
    )
    second, _ = render_with_frequency_ramp(
        split_osc,
        previous_frequency=rendered_frequency,
        target_frequency=2000.0,
        num_samples=128,
    )

    np.testing.assert_allclose(
        np.concatenate((first, second)),
        continuous,
        rtol=1e-6,
        atol=1e-6,
    )


def test_mixer_runtime_applies_channel_gains(qapp: Any):
    del qapp
    module = MixerModule()
    _connect_constant_input(module.in1_port, 1.0)
    _connect_constant_input(module.in2_port, 1.0)

    module.process_runtime(
        8,
        {"gain1": 0.25, "gain2": 0.5, "gain3": 0.75, "gain4": 1.0},
    )

    assert module._volume_components[0].amplitude == pytest.approx(0.25)
    assert module._volume_components[1].amplitude == pytest.approx(0.5)


def test_mixer_gain_knobs_update_runtime_parameters(qapp: Any):
    del qapp
    module = MixerModule()

    assert module.get_parameters()["gain1"] == pytest.approx(0.7)

    module.ch1_gain_knob.set_value(0.33)

    assert module._volume_components[0].amplitude == pytest.approx(0.33)
    assert module.get_parameters()["gain1"] == pytest.approx(0.33)


def test_filter_runtime_applies_filter_parameters(qapp: Any):
    del qapp
    module = FilterModule()
    _connect_constant_input(module.in_port, 1.0)

    module.process_runtime(
        8,
        {
            "cutoff": 300.0,
            "high_cutoff": 1200.0,
            "order": 2,
            "filter_type": "Band-pass",
        },
    )

    assert module.component.cutoff == pytest.approx((300.0, 1200.0))
    assert module.component.order == 2
    assert module.component.filter_type == "band"


def test_acid_filter_runtime_applies_parameters(qapp: Any):
    del qapp
    module = AcidFilterModule()
    _connect_constant_input(module.in_port, 0.2)

    module.process_runtime(
        16,
        {
            "cutoff": 600.0,
            "resonance": 6.0,
            "env_amount": 2.0,
            "accent_amount": 1.5,
            "drive_db": 9.0,
            "output_gain_db": -3.0,
        },
    )

    assert module.component.cutoff == pytest.approx(600.0)
    assert module.component.resonance == pytest.approx(6.0)
    assert module.component.env_amount == pytest.approx(2.0)
    assert module.component.accent_amount == pytest.approx(1.5)
    assert module.component.drive_db == pytest.approx(9.0)
    assert module.component.output_gain_db == pytest.approx(-3.0)
    assert np.asarray(module.out_port.value).shape == (16,)
    assert np.all(np.isfinite(module.out_port.value))


def test_adsr_runtime_applies_envelope_parameters(qapp: Any):
    del qapp
    module = ADSRModule()

    module.process_runtime(
        8,
        {
            "attack_duration": 0.2,
            "decay_duration": 0.3,
            "sustain_level": 0.4,
            "release_duration": 0.5,
            "retrigger_mode": "Legato",
        },
    )

    assert module._adsr_component is not None
    assert isinstance(module._adsr_component, ADSREnvelope)
    assert module._adsr_component.attack_duration == pytest.approx(0.2)
    assert module._adsr_component.decay_duration == pytest.approx(0.3)
    assert module._adsr_component.sustain_level == pytest.approx(0.4)
    assert module._adsr_component.release_duration == pytest.approx(0.5)
    assert module._adsr_component.retrigger_mode == "legato"


def test_adsr_runtime_outputs_zero_until_triggered(qapp: Any):
    del qapp
    module = ADSRModule()

    module.process_runtime(8, {})

    np.testing.assert_allclose(module.out_port.value, np.zeros(8), atol=1e-7)


def test_adsr_runtime_renders_expected_gate_attack_decay_sustain(qapp: Any):
    del qapp
    module = ADSRModule()
    gate_source = _connect_constant_input(module.gate_input, 1.0)
    gate_source.write(np.ones(12, dtype=np.float32))

    module.process_runtime(
        12,
        {
            "attack_duration": 4 / DEFAULT_SAMPLE_RATE,
            "decay_duration": 4 / DEFAULT_SAMPLE_RATE,
            "sustain_level": 0.5,
            "release_duration": 4 / DEFAULT_SAMPLE_RATE,
        },
    )

    expected = np.array(
        [0.0, 0.25, 0.5, 0.75, 1.0, 0.875, 0.75, 0.625, 0.5, 0.5, 0.5, 0.5],
        dtype=np.float32,
    )
    np.testing.assert_allclose(module.out_port.value, expected, atol=1e-7)


def test_adsr_runtime_renders_expected_gate_release(qapp: Any):
    del qapp
    module = ADSRModule()
    gate_source = _connect_constant_input(module.gate_input, 1.0)
    parameters = {
        "attack_duration": 4 / DEFAULT_SAMPLE_RATE,
        "decay_duration": 4 / DEFAULT_SAMPLE_RATE,
        "sustain_level": 0.5,
        "release_duration": 4 / DEFAULT_SAMPLE_RATE,
    }

    gate_source.write(np.ones(12, dtype=np.float32))
    module.process_runtime(12, parameters)
    gate_source.write(np.zeros(6, dtype=np.float32))
    module.process_runtime(6, parameters)

    expected = np.array([0.5, 0.375, 0.25, 0.125, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(module.out_port.value, expected, atol=1e-7)


def test_adsr_manual_gate_holds_long_attack(qapp: Any):
    del qapp
    module = ADSRModule()
    parameters = {
        "attack_duration": 8 / DEFAULT_SAMPLE_RATE,
        "decay_duration": 4 / DEFAULT_SAMPLE_RATE,
        "sustain_level": 0.5,
        "release_duration": 4 / DEFAULT_SAMPLE_RATE,
    }

    module.trigger_button.setChecked(True)
    module.process_runtime(4, parameters)
    first_chunk = np.asarray(module.out_port.value)
    module.process_runtime(4, parameters)
    second_chunk = np.asarray(module.out_port.value)

    assert module.trigger_button.text() == "trig on"
    assert second_chunk[-1] > first_chunk[-1]
    assert second_chunk[-1] == pytest.approx(0.875)


def test_adsr_manual_on_off_trigger_releases_on_button_release(qapp: Any):
    del qapp
    module = ADSRModule()
    module.trigger_mode_combo.setCurrentText("On/Off")
    parameters = {
        "attack_duration": 4 / DEFAULT_SAMPLE_RATE,
        "decay_duration": 4 / DEFAULT_SAMPLE_RATE,
        "sustain_level": 0.5,
        "release_duration": 4 / DEFAULT_SAMPLE_RATE,
    }

    module._on_trigger_pressed()
    module.process_runtime(8, parameters)
    held_chunk = np.asarray(module.out_port.value)
    module._on_trigger_released()
    module.process_runtime(5, parameters)
    released_chunk = np.asarray(module.out_port.value)

    assert module.trigger_button.text() == "trig"
    assert not module.trigger_button.isCheckable()
    assert held_chunk[-1] == pytest.approx(0.625)
    np.testing.assert_allclose(
        released_chunk, [0.5, 0.375, 0.25, 0.125, 0.0], atol=1e-7
    )


def test_adsr_switching_from_latched_to_on_off_releases_gate(qapp: Any):
    del qapp
    module = ADSRModule()
    parameters = {
        "attack_duration": 4 / DEFAULT_SAMPLE_RATE,
        "decay_duration": 4 / DEFAULT_SAMPLE_RATE,
        "sustain_level": 0.5,
        "release_duration": 4 / DEFAULT_SAMPLE_RATE,
    }

    module.trigger_button.setChecked(True)
    module.process_runtime(8, parameters)
    module.trigger_mode_combo.setCurrentText("On/Off")
    module.process_runtime(5, parameters)

    assert module.trigger_button.text() == "trig"
    assert module.get_parameters()["trigger_mode"] == "On/Off"
    np.testing.assert_allclose(
        module.out_port.value, [0.5, 0.375, 0.25, 0.125, 0.0], atol=1e-7
    )


def test_decay_envelope_runtime_applies_parameters(qapp: Any):
    del qapp
    module = DecayEnvelopeModule()

    module.process_runtime(
        8,
        {
            "attack_duration": 0.05,
            "decay_duration": 0.2,
            "amount": 0.6,
            "accent_amount": 0.4,
        },
    )

    assert isinstance(module.component, DecayEnvelope)
    assert module.component.attack_duration == pytest.approx(0.05)
    assert module.component.decay_duration == pytest.approx(0.2)
    assert module.component.amount == pytest.approx(0.6)
    assert module.trigger_button.text() == "trig"


def test_decay_envelope_runtime_renders_gate_pluck(qapp: Any):
    del qapp
    module = DecayEnvelopeModule()
    gate_source = _connect_constant_input(module.gate_input, 1.0)
    gate_source.write(np.ones(6, dtype=np.float32))

    module.process_runtime(
        6,
        {
            "attack_duration": 0.0,
            "decay_duration": 4 / DEFAULT_SAMPLE_RATE,
            "amount": 1.0,
            "accent_amount": 0.0,
        },
    )

    np.testing.assert_allclose(
        module.out_port.value,
        [1.0, 0.75, 0.5, 0.25, 0.0, 0.0],
        atol=1e-7,
    )


def test_decay_envelope_runtime_applies_accent_amount(qapp: Any):
    del qapp
    module = DecayEnvelopeModule()
    gate_source = _connect_constant_input(module.gate_input, 1.0)
    gate_source.write(np.ones(3, dtype=np.float32))
    accent_source = _connect_constant_input(module.accent_input, 1.0)
    accent_source.write(np.ones(3, dtype=np.float32))

    module.process_runtime(
        3,
        {
            "attack_duration": 0.0,
            "decay_duration": 4 / DEFAULT_SAMPLE_RATE,
            "amount": 0.5,
            "accent_amount": 1.0,
        },
    )

    np.testing.assert_allclose(module.out_port.value, [1.0, 0.75, 0.5], atol=1e-7)


def test_distortion_runtime_applies_drive_and_mix_cv(qapp: Any):
    del qapp
    module = DistortionModule()
    input_source = _connect_constant_input(module.in_port, 0.5)
    input_source.write(np.full(3, 0.5, dtype=np.float32))
    drive_source = _connect_constant_input(module.drive_cv_port, 0.0)
    drive_source.write(np.array([0.0, 1.0, 1.0], dtype=np.float32))
    mix_source = _connect_constant_input(module.mix_cv_port, 0.0)
    mix_source.write(np.array([-1.0, 0.0, 0.0], dtype=np.float32))

    module.process_runtime(
        3,
        {"drive": 0.0, "mix": 1.0, "distortion_type": "soft"},
    )

    output = np.asarray(module.out_port.value)
    assert output[0] == pytest.approx(0.25)
    assert output[1] > output[0]
    assert output[2] == pytest.approx(output[1])
    assert module.component.drive == pytest.approx(0.0)
    assert module.component.mix == pytest.approx(1.0)


def test_compressor_runtime_applies_parameters(qapp: Any):
    del qapp
    module = CompressorModule()
    input_source = _connect_constant_input(module.in_port, 1.0)
    input_source.write(np.ones(64, dtype=np.float32))

    module.process_runtime(
        64,
        {
            "threshold_db": -30.0,
            "ratio": 12.0,
            "attack_ms": 0.1,
            "release_ms": 50.0,
            "makeup_gain_db": 0.0,
            "mix": 1.0,
        },
    )

    assert module.component.threshold_db == pytest.approx(-30.0)
    assert module.component.ratio == pytest.approx(12.0)
    assert module.component.attack_ms == pytest.approx(0.1)
    assert module.component.release_ms == pytest.approx(50.0)
    assert module.component.makeup_gain_db == pytest.approx(0.0)
    assert module.component.mix == pytest.approx(1.0)
    output = np.asarray(module.out_port.value)
    assert output.shape == (64,)
    assert output[-1] < 1.0
    assert np.all(np.isfinite(output))


def test_tb303_voice_runtime_outputs_silence_without_required_inputs(qapp: Any):
    del qapp
    module = TB303VoiceModule()

    module.process_runtime(8, {})

    np.testing.assert_allclose(module.out_port.value, np.zeros(8), atol=1e-7)


def test_tb303_voice_runtime_applies_parameters_and_renders(qapp: Any):
    del qapp
    module = TB303VoiceModule()
    assert module.tune_knob.min_value == pytest.approx(-24.0)
    assert module.tune_knob.max_value == pytest.approx(24.0)
    pitch_cv = frequency_to_pitch_cv(110.0)
    freq_source = _connect_constant_input(module.freq_input, pitch_cv)
    freq_source.write(np.full(32, pitch_cv, dtype=np.float32))
    gate_source = _connect_constant_input(module.gate_input, 1.0)
    gate_source.write(np.ones(32, dtype=np.float32))

    module.process_runtime(
        32,
        {
            "waveform": "Square",
            "tuning": 1.0,
            "pulsewidth": 0.35,
            "cutoff": 600.0,
            "resonance": 7.0,
            "env_amount": 2.0,
            "decay": 0.12,
            "accent": 0.5,
            "slide_time": 0.05,
            "drive_db": 8.0,
            "volume": 0.7,
        },
    )

    assert module.component.waveform == "Square"
    assert module.component.pulsewidth == pytest.approx(0.35)
    assert module.component.cutoff == pytest.approx(600.0)
    assert module.component.decay == pytest.approx(0.12)
    output = np.asarray(module.out_port.value)
    assert output.shape == (32,)
    assert np.all(np.isfinite(output))
    assert np.max(np.abs(output)) > 0.0


def test_vco_runtime_applies_oscillator_parameters(qapp: Any):
    del qapp
    module = ModulatedOscillatorModule()

    module.process_runtime(
        8,
        {
            "waveform": "Square",
            "mode": "ideal",
            "frequency": 330.0,
            "gain_db": -6.0,
            "phase": 45.0,
            "pulsewidth": 0.25,
        },
    )

    assert isinstance(module.component, SquareOscillator)
    assert module.component.frequency == pytest.approx(330.0)
    assert module.component.gain_db == pytest.approx(-6.0)
    assert module.component.phase == pytest.approx(np.deg2rad(45.0))
    assert module.component.pulsewidth == pytest.approx(0.25)


def test_vco_waveform_change_updates_cached_mode(qapp: Any):
    del qapp
    module = ModulatedOscillatorModule()
    module.wave_combo.setCurrentText("Square")
    module.mode_combo.setCurrentText("vcv")
    assert module.get_parameters()["mode"] == "vcv"

    module.wave_combo.setCurrentText("Sine")

    assert module.mode_combo.currentText() == "pure"
    assert module.get_parameters()["waveform"] == "Sine"
    assert module.get_parameters()["mode"] == "pure"


def test_vco_runtime_normalizes_stale_mode_after_waveform_change(qapp: Any):
    del qapp
    module = ModulatedOscillatorModule()

    module.process_runtime(
        32,
        {
            "waveform": "Sine",
            "mode": "vcv",
            "frequency": 330.0,
            "gain_db": -6.0,
            "phase": 0.0,
            "pulsewidth": 0.5,
        },
    )

    assert isinstance(module.component, SineOscillator)
    assert module.component.mode == "pure"
    output = np.asarray(module.out_port.value)
    assert output.shape == (32,)
    assert np.all(np.isfinite(output))
    assert np.max(np.abs(output)) > 0.0
