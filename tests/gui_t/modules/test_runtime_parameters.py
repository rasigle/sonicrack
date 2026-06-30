"""Runtime parameter snapshot coverage for GUI modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.constants import DEFAULT_SAMPLE_RATE
from src.engine.modulator import ADSREnvelope, DecayEnvelope
from src.engine.oscillator_square import SquareOscillator
from src.gui.core.port import Port
from src.gui.modules.effects.effects_distortion import DistortionModule
from src.gui.modules.mixer import MixerModule
from src.gui.modules.modifier.acid_filter import AcidFilterModule
from src.gui.modules.modifier.filter import FilterModule
from src.gui.modules.modulated_source.envelope_adsr import ADSRModule
from src.gui.modules.modulated_source.envelope_decay import DecayEnvelopeModule
from src.gui.modules.source.lfo import LFOModule
from src.gui.modules.source.oscillator import OscillatorModule
from src.gui.modules.source.vco import ModulatedOscillatorModule
from src.gui.modules.voice.tb303_voice import TB303VoiceModule


def _connect_constant_input(input_port: Port, value: float = 1.0) -> Port:
    output_port = Port("output", "Test Out")
    output_port.write(np.full(8, value, dtype=np.float32))
    output_port.connect(input_port)
    return output_port


def test_oscillator_runtime_applies_frequency_and_pulsewidth(qapp: Any):
    del qapp
    module = OscillatorModule()

    module.process_runtime(8, {"frequency": 880.0, "pulsewidth": 0.25})

    assert module._sine_oscillator.frequency == pytest.approx(880.0)
    assert module._triangle_oscillator.frequency == pytest.approx(880.0)
    assert module._sawtooth_oscillator.frequency == pytest.approx(880.0)
    assert module._square_oscillator.frequency == pytest.approx(880.0)
    assert module._square_oscillator.pulsewidth == pytest.approx(0.25)


def test_lfo_runtime_applies_frequency_and_pulsewidth(qapp: Any):
    del qapp
    module = LFOModule()

    module.process_runtime(8, {"frequency": 4.0, "pulsewidth": 0.75})

    assert module._sine_oscillator.frequency == pytest.approx(4.0)
    assert module._triangle_oscillator.frequency == pytest.approx(4.0)
    assert module._sawtooth_oscillator.frequency == pytest.approx(4.0)
    assert module._square_oscillator.frequency == pytest.approx(4.0)
    assert module._square_oscillator.pulsewidth == pytest.approx(0.75)


def test_oscillator_frequency_changes_are_ramped_across_buffer(qapp: Any):
    del qapp
    module = OscillatorModule()

    module.process_runtime(128, {"frequency": 120.0, "pulsewidth": 0.5})
    first = np.asarray(module.triangle_port.value)
    module.process_runtime(128, {"frequency": 2000.0, "pulsewidth": 0.5})
    second = np.asarray(module.triangle_port.value)

    boundary_jump = abs(float(second[0] - first[-1]))
    assert boundary_jump < 0.08
    assert module._triangle_oscillator.frequency == pytest.approx(2000.0)


def test_lfo_frequency_changes_are_ramped_across_buffer(qapp: Any):
    del qapp
    module = LFOModule()

    module.process_runtime(128, {"frequency": 1.0, "pulsewidth": 0.5})
    first = np.asarray(module.sine_port.value)
    module.process_runtime(128, {"frequency": 12.0, "pulsewidth": 0.5})
    second = np.asarray(module.sine_port.value)

    boundary_jump = abs(float(second[0] - first[-1]))
    assert boundary_jump < 0.01
    assert module._sine_oscillator.frequency == pytest.approx(12.0)


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

    assert module.trigger_button.text() == "Gate On"
    assert second_chunk[-1] > first_chunk[-1]
    assert second_chunk[-1] == pytest.approx(0.875)


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


def test_tb303_voice_runtime_outputs_silence_without_required_inputs(qapp: Any):
    del qapp
    module = TB303VoiceModule()

    module.process_runtime(8, {})

    np.testing.assert_allclose(module.out_port.value, np.zeros(8), atol=1e-7)


def test_tb303_voice_runtime_applies_parameters_and_renders(qapp: Any):
    del qapp
    module = TB303VoiceModule()
    freq_source = _connect_constant_input(module.freq_input, 110.0)
    freq_source.write(np.full(32, 110.0, dtype=np.float32))
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
