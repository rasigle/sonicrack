"""Coverage for recommended-priority modules (effects, poly, utilities, UX)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from soniclab import Quantizer, SampleAndHold, SlewLimiter

from sonicrack.gui.modules.effects.effects_chorus import ChorusModule
from sonicrack.gui.modules.effects.effects_eq import EQModule
from sonicrack.gui.modules.effects.effects_limiter import LimiterModule
from sonicrack.gui.modules.effects.effects_phaser import PhaserModule
from sonicrack.gui.modules.modifier.attenuverter import AttenuverterModule
from sonicrack.gui.modules.modifier.mod_matrix import ModMatrixModule
from sonicrack.gui.modules.modifier.mult import MultModule
from sonicrack.gui.modules.modifier.multiple import MultipleModule
from sonicrack.gui.modules.modifier.quantizer import QuantizerModule
from sonicrack.gui.modules.modifier.sample_hold import SampleHoldModule
from sonicrack.gui.modules.modifier.slew import SlewModule
from sonicrack.gui.modules.sequencing.step_sequencer import StepSequencerModule
from sonicrack.gui.modules.source.lfo import LFOModule
from sonicrack.gui.modules.source.wavetable import WavetableModule
from sonicrack.gui.modules.voice.poly_voice import PolyVoiceModule
from sonicrack.patching.port import Port
from sonicrack.patching.registry import discover_modules, get_registry


def _connect_signal(input_port: Port, values: np.ndarray) -> Port:
    output_port = Port("output", "Test Out")
    output_port.write(values)
    output_port.connect(input_port)
    return output_port


def test_priority_modules_are_discoverable(qapp: Any) -> None:
    del qapp
    discover_modules("sonicrack.gui.modules", recursive=True)
    registered = get_registry().list_modules()
    for title in (
        "Chorus",
        "Phaser",
        "Limiter",
        "EQ",
        "Wavetable",
        "Poly Voice",
        "Attenuverter",
        "Mult",
        "Multiple",
        "Sample & Hold",
        "Slew",
        "Quantizer",
        "Mod Matrix",
    ):
        assert title in registered


def test_step_sequencer_visual_notes_and_playhead(qapp: Any) -> None:
    del qapp
    module = StepSequencerModule()
    assert len(module.note_buttons) == module.step_toggle_count
    assert len(module.step_leds) == module.step_toggle_count

    module.set_step_note(0, 48)
    assert "48" in module.notes_edit.text().split(",")[0]
    assert module.note_buttons[0].text() != "—"

    module.cycle_step_note(1)
    assert module.notes_edit.text()

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
    assert any(led.is_on() for led in module.step_leds)


def test_chorus_and_phaser_process_audio(qapp: Any) -> None:
    del qapp
    x = np.sin(np.linspace(0, 20, 256, dtype=np.float32))
    chorus = ChorusModule()
    _connect_signal(chorus.in_port, x)
    chorus.process_runtime(256, {"rate": 0.5, "depth": 2.0, "delay": 10.0, "mix": 0.5})
    out = np.asarray(chorus.out_port.value)
    assert out.shape == (256,)
    assert np.std(out) > 0.01

    phaser = PhaserModule()
    _connect_signal(phaser.in_port, x)
    phaser.process_runtime(
        256, {"rate": 0.3, "depth": 0.8, "feedback": 0.3, "mix": 0.5}
    )
    assert np.asarray(phaser.out_port.value).shape == (256,)


def test_limiter_and_eq_process_audio(qapp: Any) -> None:
    del qapp
    x = np.full(128, 1.5, dtype=np.float32)
    limiter = LimiterModule()
    _connect_signal(limiter.in_port, x)
    limiter.process_runtime(128, {"threshold": 0.8, "release": 40.0, "makeup": 1.0})
    limited = np.asarray(limiter.out_port.value)
    assert limited.shape == (128,)
    assert float(np.max(np.abs(limited))) <= 0.85

    eq = EQModule()
    _connect_signal(eq.in_port, np.sin(np.linspace(0, 40, 128, dtype=np.float32)))
    eq.process_runtime(
        128, {"frequency": 1000.0, "gain_db": 6.0, "q": 1.0, "mode": "peak"}
    )
    assert np.asarray(eq.out_port.value).shape == (128,)


def test_wavetable_and_utilities(qapp: Any) -> None:
    del qapp
    wt = WavetableModule()
    wt.process_runtime(64, {"frequency": 220.0, "morph": 0.5, "gain": 0.4})
    assert np.asarray(wt.out_port.value).shape == (64,)
    assert np.std(wt.out_port.value) > 0.01

    att = AttenuverterModule()
    _connect_signal(att.in_port, np.ones(16, dtype=np.float32))
    att.process_runtime(16, {"amount": -0.5, "offset": 0.25})
    np.testing.assert_allclose(att.out_port.value, -0.25)

    mult = MultModule()
    _connect_signal(mult.a_port, np.full(8, 0.5, dtype=np.float32))
    _connect_signal(mult.b_port, np.full(8, 0.4, dtype=np.float32))
    mult.process_runtime(8, {"amount": 1.0})
    np.testing.assert_allclose(mult.out_port.value, 0.2)


def test_poly_voice_plays_chord(qapp: Any) -> None:
    del qapp
    module = PolyVoiceModule()
    module.note_on_midi(60, 100)
    module.note_on_midi(64, 100)
    module.note_on_midi(67, 100)
    module.process_runtime(
        512,
        {
            "waveform": "Sine",
            "attack": 0.005,
            "decay": 0.1,
            "sustain": 0.8,
            "release": 0.2,
            "voices": 8.0,
        },
    )
    audio = np.asarray(module.out_port.value)
    assert audio.shape == (512,)
    assert np.std(audio) > 0.01
    module.note_off_midi(60)
    module.note_off_midi(64)
    module.note_off_midi(67)


def test_lfo_polarity_and_amount(qapp: Any) -> None:
    del qapp
    module = LFOModule()
    module.process_runtime(
        128,
        {"frequency": 2.0, "pulsewidth": 0.5, "amount": 0.5, "polarity": "Unipolar"},
    )
    samples = np.asarray(module.sine_port.value)
    assert samples.shape == (128,)
    assert float(np.min(samples)) >= -0.01
    assert float(np.max(samples)) <= 0.51
    assert module._sine_lfo.depth == pytest.approx(0.5)
    assert module._sine_lfo.bipolar is False


def test_lfo_engine_shapes(qapp: Any) -> None:
    del qapp
    module = LFOModule()
    params = {
        "frequency": 20.0,
        "pulsewidth": 0.5,
        "amount": 1.0,
        "polarity": "Bipolar",
    }
    module.process_runtime(512, params)
    module.process_runtime(512, params)
    sine = np.asarray(module.sine_port.value)
    triangle = np.asarray(module.triangle_port.value)
    square = np.asarray(module.square_port.value)
    assert sine.shape == triangle.shape == square.shape == (512,)
    assert float(np.max(np.abs(sine))) > 0.5
    assert float(np.max(np.abs(triangle))) > 0.5
    assert float(np.max(np.abs(square))) > 0.5

    # S&H hold length at 20 Hz is ~sample_rate/20 samples; collect enough
    # buffers to span multiple holds and confirm stepped random output.
    random_chunks: list[np.ndarray] = []
    for _ in range(12):
        module.process_runtime(512, params)
        random_chunks.append(np.asarray(module.random_port.value).copy())
    random = np.concatenate(random_chunks)
    assert random.shape[0] == 12 * 512
    assert float(np.std(random)) > 0.01


def test_cv_utilities_and_mod_matrix(qapp: Any) -> None:
    del qapp
    # Multiple fans one signal to four outs
    mult = MultipleModule()
    src = np.linspace(-1, 1, 32, dtype=np.float32)
    _connect_signal(mult.in_port, src)
    mult.process_runtime(32, {})
    for port in mult.out_ports:
        np.testing.assert_allclose(port.value, src)

    # Sample & hold on rising edges
    sh = SampleHoldModule()
    signal = np.array([0.1, 0.2, 0.9, 0.3, 0.4], dtype=np.float32)
    clock = np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    _connect_signal(sh.in_port, signal)
    _connect_signal(sh.clock_port, clock)
    sh.process_runtime(5, {})
    held = np.asarray(sh.out_port.value)
    assert float(held[0]) == 0.0  # nothing held yet until first edge
    assert float(held[2]) == float(signal[2])  # sampled on rising edge
    assert float(held[3]) == float(signal[2])  # held
    assert float(held[4]) == float(signal[4])  # re-sampled

    # Slew limits step response
    slew = SlewModule()
    step = np.concatenate(
        [np.zeros(8, dtype=np.float32), np.ones(24, dtype=np.float32)]
    )
    _connect_signal(slew.in_port, step)
    slew.process_runtime(32, {"rise_ms": 50.0, "fall_ms": 50.0})
    out = np.asarray(slew.out_port.value)
    assert float(out[8]) < 1.0  # not an instant jump

    # Quantizer snaps to chromatic
    quant = QuantizerModule()
    # MIDI 60.5 ≈ +0.0417V → should snap toward C4 (0V) or C# 
    _connect_signal(quant.in_port, np.full(8, 0.5 / 12.0, dtype=np.float32))
    quant.process_runtime(8, {"scale": "Chromatic", "root": "C"})
    qout = np.asarray(quant.out_port.value)
    assert np.allclose(qout, qout[0])  # constant
    # Nearest semitone to 60.5 is 60 or 61
    midi = qout[0] * 12.0 + 60.0
    assert abs(midi - round(midi)) < 1e-4

    # Mod matrix mixes A and B
    matrix = ModMatrixModule()
    _connect_signal(matrix.src_ports[0], np.ones(16, dtype=np.float32))
    _connect_signal(matrix.src_ports[1], np.full(16, 0.5, dtype=np.float32))
    matrix.process_runtime(
        16,
        {
            "amt_a_0": 1.0,
            "amt_b_0": 0.0,
            "offset_0": 0.0,
            "amt_a_1": 0.0,
            "amt_b_1": 1.0,
            "offset_1": 0.25,
            "amt_a_2": 0.5,
            "amt_b_2": 0.5,
            "offset_2": 0.0,
            "amt_a_3": 0.0,
            "amt_b_3": 0.0,
            "offset_3": -0.5,
        },
    )
    np.testing.assert_allclose(matrix.dest_ports[0].value, 1.0)
    np.testing.assert_allclose(matrix.dest_ports[1].value, 0.75)
    np.testing.assert_allclose(matrix.dest_ports[2].value, 0.75)
    np.testing.assert_allclose(matrix.dest_ports[3].value, -0.5)


def test_dsp_utility_units() -> None:
    sh = SampleAndHold()
    x = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    clk = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    y = sh(x, clk)
    assert float(y[1]) == 1.0
    assert float(y[2]) == 1.0

    slew = SlewLimiter(sample_rate=1000.0, rise_ms=10.0, fall_ms=10.0)
    step = np.ones(20, dtype=np.float32)
    out = slew(step)
    assert float(out[0]) < 1.0
    assert float(out[-1]) <= 1.0 + 1e-5

    quant = Quantizer(scale="Octaves", root=0)
    # A4 = 69 → 0.75V → nearest octave to C (…48, 60, 72…)
    q = quant(np.array([0.75], dtype=np.float32))
    midi = float(q[0]) * 12.0 + 60.0
    assert abs(midi - 72.0) < 0.01 or abs(midi - 60.0) < 0.01


def test_poly_voice_has_midi_controls(qapp: Any) -> None:
    del qapp
    module = PolyVoiceModule()
    assert hasattr(module, "midi_device_combo")
    assert not hasattr(module, "midi_start_btn")
    assert not hasattr(module, "midi_refresh_btn")
    assert module.midi_device_combo.count() >= 1
    # note_on_midi still works without a hardware device
    module.note_on_midi(60, 100)
    module.process_runtime(
        512,
        {
            "waveform": "Sine",
            "attack": 0.005,
            "decay": 0.1,
            "sustain": 0.8,
            "release": 0.2,
            "voices": 8.0,
        },
    )
    assert np.std(module.out_port.value) > 0.001
    module.note_off_midi(60)
    module.shutdown()
