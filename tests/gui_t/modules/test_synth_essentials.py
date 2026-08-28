"""Tests for high-impact synth modules added around soniclab primitives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from soniclab.utils.cv import frequency_to_pitch_cv

pytest.importorskip("soniclab.dsp.filters.ladder")
pytest.importorskip("soniclab.generators.sample_player")
pytest.importorskip("soniclab.voices.subtractive")

from sonicrack.gui.modules.effects.effects_bitcrusher import BitcrusherModule
from sonicrack.gui.modules.effects.effects_flanger import FlangerModule
from sonicrack.gui.modules.effects.effects_ring_mod import RingModModule
from sonicrack.gui.modules.mixer.mixer import MixerModule
from sonicrack.gui.modules.modifier.ladder_filter import LadderFilterModule
from sonicrack.gui.modules.modifier.stereo_width import StereoWidthModule
from sonicrack.gui.modules.modifier.svf_filter import SVFFilterModule
from sonicrack.gui.modules.modulated_source.envelope_follower import (
    EnvelopeFollowerModule,
)
from sonicrack.gui.modules.source.sample_player import SamplePlayerModule
from sonicrack.gui.modules.source.unison import UnisonModule
from sonicrack.gui.modules.voice.subtractive_voice import SubtractiveVoiceModule
from sonicrack.patching.port import Port
from sonicrack.patching.registry import discover_modules, get_registry


def _connect_signal(input_port: Port, values: np.ndarray) -> Port:
    output_port = Port("output", "Test Out")
    output_port.write(values)
    output_port.connect(input_port)
    return output_port


def test_new_modules_are_discoverable(qapp: Any) -> None:
    del qapp
    discover_modules("sonicrack.gui.modules", recursive=True)
    registered = get_registry().list_modules()
    for title in (
        "Flanger",
        "Bitcrusher",
        "Ring Mod",
        "Ladder Filter",
        "SVF Filter",
        "Stereo Width",
        "Env Follower",
        "Sample Player",
        "Subtractive Voice",
        "Unison",
    ):
        assert title in registered


def test_flanger_bitcrusher_ring_mod(qapp: Any) -> None:
    del qapp
    x = np.sin(np.linspace(0, 40, 256, dtype=np.float32))
    flanger = FlangerModule()
    _connect_signal(flanger.in_port, x)
    flanger.process_runtime(
        256,
        {"rate": 0.4, "depth": 1.2, "delay": 2.0, "feedback": 0.5, "mix": 0.7},
    )
    out = np.asarray(flanger.out_port.value)
    assert out.shape[0] == 256
    assert np.std(out) > 0.01

    crush = BitcrusherModule()
    _connect_signal(crush.in_port, x)
    crush.process_runtime(256, {"bits": 4.0, "downsample": 4.0, "mix": 1.0})
    crushed = np.asarray(crush.out_port.value)
    assert crushed.shape[0] == 256
    assert np.std(crushed) > 0.01

    ring = RingModModule()
    carrier = np.sin(np.linspace(0, 80, 256, dtype=np.float32))
    _connect_signal(ring.in_port, x)
    _connect_signal(ring.carrier_port, carrier)
    ring.process_runtime(256, {"amount": 1.0, "mix": 1.0})
    product = np.asarray(ring.out_port.value)
    assert product.shape == (256,)
    assert np.std(product) > 0.01


def test_ladder_and_svf_filter_audio(qapp: Any) -> None:
    del qapp
    x = np.sin(np.linspace(0, 60, 256, dtype=np.float32))
    ladder = LadderFilterModule()
    _connect_signal(ladder.in_port, x)
    ladder.process_runtime(
        256,
        {"cutoff": 600.0, "resonance": 0.6, "drive": 1.4, "cv_depth_octaves": 0.0},
    )
    ladder_out = np.asarray(ladder.out_port.value)
    assert ladder_out.shape[0] == 256
    assert np.std(ladder_out) > 0.001

    svf = SVFFilterModule()
    _connect_signal(svf.in_port, x)
    svf.process_runtime(
        256,
        {
            "cutoff": 800.0,
            "resonance": 1.5,
            "filter_type": "Low-pass",
            "cv_depth_octaves": 0.0,
        },
    )
    svf_out = np.asarray(svf.out_port.value)
    assert svf_out.shape[0] == 256
    assert np.std(svf_out) > 0.001


def test_stereo_width_and_envelope_follower(qapp: Any) -> None:
    del qapp
    left = np.sin(np.linspace(0, 20, 128, dtype=np.float32))
    right = np.sin(np.linspace(0, 27, 128, dtype=np.float32))
    stereo = np.column_stack((left, right))
    width = StereoWidthModule()
    _connect_signal(width.in_port, stereo)
    width.process_runtime(128, {"width": 0.0})
    collapsed = np.asarray(width.out_port.value)
    assert collapsed.shape == (128, 2)
    np.testing.assert_allclose(collapsed[:, 0], collapsed[:, 1], atol=1e-5)

    follower = EnvelopeFollowerModule()
    burst = np.concatenate(
        [np.zeros(32, dtype=np.float32), np.ones(96, dtype=np.float32)]
    )
    _connect_signal(follower.in_port, burst)
    follower.process_runtime(128, {"attack": 1.0, "release": 80.0, "gain": 1.0})
    env = np.asarray(follower.env_port.value)
    thru = np.asarray(follower.thru_port.value)
    assert env.shape == (128,)
    assert float(env[-1]) > float(env[0])
    np.testing.assert_allclose(thru, burst)


def test_sample_player_and_unison(qapp: Any) -> None:
    del qapp
    player = SamplePlayerModule()
    player.process_runtime(256, {"frequency": 440.0, "level": 0.8, "loop": "On"})
    audio = np.asarray(player.out_port.value)
    assert audio.shape == (256,)
    assert np.std(audio) > 0.01

    unison = UnisonModule()
    unison.process_runtime(
        256,
        {
            "waveform": "Saw",
            "frequency": 110.0,
            "voices": 3.0,
            "detune": 12.0,
            "spread": 0.8,
            "level": 0.4,
        },
    )
    stacked = np.asarray(unison.out_port.value)
    assert stacked.shape == (256, 2)
    assert np.std(stacked) > 0.01


def test_subtractive_voice_gated(qapp: Any) -> None:
    del qapp
    voice = SubtractiveVoiceModule()
    freq = np.full(512, float(frequency_to_pitch_cv(220.0)), dtype=np.float32)
    gate = np.concatenate(
        [np.ones(300, dtype=np.float32), np.zeros(212, dtype=np.float32)]
    )
    _connect_signal(voice.freq_input, freq)
    _connect_signal(voice.gate_input, gate)
    voice.process_runtime(
        512,
        {
            "waveform": "Saw",
            "filter_mode": "Low",
            "cutoff": 1200.0,
            "resonance": 1.0,
            "env_amount": 2.0,
            "attack": 0.005,
            "decay": 0.1,
            "sustain": 0.6,
            "release": 0.2,
            "lfo_rate": 0.0,
            "lfo_filter": 0.0,
            "volume": 0.8,
        },
    )
    out = np.asarray(voice.out_port.value)
    assert out.shape == (512,)
    assert np.std(out) > 0.01


def test_mixer_pan_mute_and_meters(qapp: Any) -> None:
    del qapp
    mixer = MixerModule()
    leftish = np.ones(64, dtype=np.float32)
    _connect_signal(mixer.in1_port, leftish)
    mixer.mute_buttons[0].setChecked(False)
    mixer.process_runtime(
        64,
        {"gain1": 1.0, "pan1": -1.0, "mute1": False},
    )
    out = np.asarray(mixer.out_port.value)
    assert out.shape == (64, 2)
    assert float(np.mean(out[:, 0])) > float(np.mean(out[:, 1]))

    mixer.process_runtime(64, {"gain1": 1.0, "pan1": 0.0, "mute1": False})
    centered = np.asarray(mixer.out_port.value)
    assert centered.ndim == 1
    assert centered.shape == (64,)

    mixer.process_runtime(64, {"gain1": 1.0, "pan1": 0.0, "mute1": True})
    muted = np.asarray(mixer.out_port.value)
    assert np.allclose(muted, 0.0)
    mixer._refresh_meters()
    assert mixer.meters[0].width() > 0
