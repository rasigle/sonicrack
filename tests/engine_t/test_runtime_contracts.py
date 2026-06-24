"""Tests for shared engine runtime contracts."""

import math

import numpy as np
import pytest

from src.engine import (
    ADSREnvelope,
    Chain,
    Delay,
    NoiseGenerator,
    Reverb,
    SineOscillator,
    WaveAdder,
)
from src.engine.cv_utils import CVScaler
from src.engine.filter import ButterworthFilter
from src.engine.validation import validate_numeric_range


@pytest.mark.parametrize(
    ("factory", "expected_error"),
    [
        (lambda: SineOscillator(sample_rate=0), ValueError),
        (lambda: ADSREnvelope(sample_rate=-1), ValueError),
        (lambda: ButterworthFilter(sample_rate=0), ValueError),
        (lambda: Delay(sample_rate=math.nan), ValueError),
        (lambda: Reverb(sample_rate=math.inf), ValueError),
        (lambda: NoiseGenerator(sample_rate=True), TypeError),
    ],
)
def test_engine_components_reject_invalid_sample_rates(factory, expected_error):
    with pytest.raises(expected_error):
        factory()


def test_adsr_preserves_explicit_sample_rate():
    envelope = ADSREnvelope(
        attack_duration=0.01,
        decay_duration=0.02,
        sustain_level=0.5,
        release_duration=0.03,
        sample_rate=1000,
    )

    assert envelope.sample_rate == 1000
    assert envelope._attack_samples == 10
    assert envelope._decay_samples == 20
    assert envelope._release_samples == 30


def test_noise_smoothing_duration_uses_explicit_sample_rate():
    noise = NoiseGenerator(sample_rate=48000)

    assert noise._smoothing_duration_samples == 480


def test_validate_numeric_range_accepts_inclusive_bounds():
    assert validate_numeric_range(0.0, 0.0, 1.0, name="mix") == 0.0
    assert validate_numeric_range(1, 0.0, 1.0, name="mix") == 1.0


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf])
def test_validate_numeric_range_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        validate_numeric_range(value, 0.0, 1.0, name="mix")


@pytest.mark.parametrize("value", [True, "0.5", object()])
def test_validate_numeric_range_rejects_non_real_values(value):
    with pytest.raises(TypeError):
        validate_numeric_range(value, 0.0, 1.0, name="mix")


@pytest.mark.parametrize(
    "renderer",
    [
        lambda n: SineOscillator().get_samples(n),
        lambda n: SineOscillator().get_samples_vectorized(n),
        lambda n: ADSREnvelope().get_samples(n),
        lambda n: Chain(SineOscillator()).get_samples_vectorized(n),
        lambda n: WaveAdder(SineOscillator(), SineOscillator()).get_samples(n),
        lambda n: Delay(SineOscillator()).get_samples_vectorized(n),
        lambda n: Reverb(SineOscillator()).get_samples_vectorized(n),
        lambda n: NoiseGenerator().get_samples(n),
        lambda n: CVScaler(SineOscillator()).get_samples(n),
    ],
)
def test_render_apis_reject_negative_sample_counts(renderer):
    with pytest.raises(ValueError, match="non-negative"):
        renderer(-1)


@pytest.mark.parametrize(
    "renderer",
    [
        lambda n: SineOscillator().get_samples(n),
        lambda n: SineOscillator().get_samples_vectorized(n),
        lambda n: ADSREnvelope().get_samples(n),
        lambda n: Chain(SineOscillator()).get_samples_vectorized(n),
        lambda n: Delay(SineOscillator()).get_samples_vectorized(n),
        lambda n: Reverb(SineOscillator()).get_samples_vectorized(n),
        lambda n: NoiseGenerator().get_samples(n),
    ],
)
def test_render_apis_reject_non_integer_sample_counts(renderer):
    with pytest.raises(TypeError, match="integer"):
        renderer(1.5)


def test_zero_sample_render_returns_empty_array():
    samples = Chain(SineOscillator(), ButterworthFilter()).get_samples_vectorized(0)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (0,)
    assert samples.dtype == np.float32
