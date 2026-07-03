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


def test_tb303_voice_uses_click_safe_attack_on_note_start():
    voice = TB303Voice(sample_rate=44100)

    output = voice.process(
        frequency=np.full(128, 110.0, dtype=np.float32),
        gate=np.ones(128, dtype=np.float32),
        accent=np.zeros(128, dtype=np.float32),
        slide=np.zeros(128, dtype=np.float32),
    )

    assert abs(float(output[0])) < 1e-4


def test_tb303_voice_advances_phase_with_modulated_pitch_buffer():
    voice = TB303Voice(waveform="Square", sample_rate=44100)
    initial_phase = voice._oscillator._i

    output = voice.process(
        frequency=np.linspace(110.0, 220.0, 128, dtype=np.float32),
        gate=np.ones(128, dtype=np.float32),
        accent=np.zeros(128, dtype=np.float32),
        slide=np.ones(128, dtype=np.float32),
    )

    assert output.shape == (128,)
    assert np.all(np.isfinite(output))
    assert voice._oscillator._i != initial_phase


def test_tb303_tuning_is_semitone_offset_not_frequency_multiplier():
    voice = TB303Voice(sample_rate=44100)
    captured = []

    def capture_pitch(frequencies, slides):
        del slides
        captured.append(np.asarray(frequencies).copy())
        return np.asarray(frequencies)

    voice._slide.process = capture_pitch
    voice.process(
        frequency=np.full(8, 110.0, dtype=np.float32),
        gate=np.ones(8, dtype=np.float32),
        accent=np.zeros(8, dtype=np.float32),
        slide=np.zeros(8, dtype=np.float32),
    )

    np.testing.assert_allclose(captured[-1], 110.0)

    voice.tuning = 12.0
    voice.process(
        frequency=np.full(8, 110.0, dtype=np.float32),
        gate=np.ones(8, dtype=np.float32),
        accent=np.zeros(8, dtype=np.float32),
        slide=np.zeros(8, dtype=np.float32),
    )

    np.testing.assert_allclose(captured[-1], 220.0)
