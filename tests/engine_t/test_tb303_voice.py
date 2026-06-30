"""Tests for the composed TB-303 style voice."""

import numpy as np

from src.engine.voices import TB303Voice


def test_tb303_voice_renders_finite_audio():
    voice = TB303Voice(sample_rate=44100)

    output = voice.process(
        frequency=np.full(128, 110.0, dtype=np.float32),
        gate=np.ones(128, dtype=np.float32),
        accent=np.zeros(128, dtype=np.float32),
        slide=np.zeros(128, dtype=np.float32),
    )

    assert output.shape == (128,)
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))
    assert np.max(np.abs(output)) > 0.0


def test_tb303_voice_accepts_short_optional_cv():
    voice = TB303Voice(waveform="Square", sample_rate=44100)

    output = voice.process(
        frequency=np.full(32, 110.0, dtype=np.float32),
        gate=np.ones(32, dtype=np.float32),
        accent=np.array([1.0], dtype=np.float32),
        slide=np.array([0.0, 1.0], dtype=np.float32),
    )

    assert output.shape == (32,)
    assert np.all(np.isfinite(output))
