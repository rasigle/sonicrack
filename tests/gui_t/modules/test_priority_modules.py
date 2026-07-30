"""Coverage for recommended-priority modules (effects, poly, utilities, UX)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sonicrack.gui.modules.effects.effects_chorus import ChorusModule
from sonicrack.gui.modules.effects.effects_eq import EQModule
from sonicrack.gui.modules.effects.effects_limiter import LimiterModule
from sonicrack.gui.modules.effects.effects_phaser import PhaserModule
from sonicrack.gui.modules.modifier.attenuverter import AttenuverterModule
from sonicrack.gui.modules.modifier.mult import MultModule
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
