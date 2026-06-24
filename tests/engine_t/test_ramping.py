"""Tests for shared ramp and smoothing helpers."""

import numpy as np
import pytest

from src.engine.ramping import (
    consume_linear_ramp,
    duration_ms_to_samples,
    fill_linear_ramp,
)


def test_duration_ms_to_samples_uses_sample_rate():
    assert duration_ms_to_samples(48000, 5.0) == 240
    assert duration_ms_to_samples(44100, 10.0) == 441


@pytest.mark.parametrize("duration", [-1.0, float("nan"), float("inf")])
def test_duration_ms_to_samples_rejects_invalid_durations(duration):
    with pytest.raises(ValueError, match="finite and non-negative"):
        duration_ms_to_samples(44100, duration)


def test_fill_linear_ramp_reuses_buffers():
    out = np.empty(8, dtype=np.float32)
    indices = np.arange(8, dtype=np.float32)

    ramp = fill_linear_ramp(0.0, 1.0, 5, out=out, index_buffer=indices)

    assert ramp.base is out or np.shares_memory(ramp, out)
    np.testing.assert_allclose(ramp, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_consume_linear_ramp_spans_multiple_chunks():
    current = 0.0
    remaining = 10

    first, current, remaining = consume_linear_ramp(current, 1.0, remaining, 4)
    second, current, remaining = consume_linear_ramp(current, 1.0, remaining, 6)

    np.testing.assert_allclose(first, [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(second, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert current == pytest.approx(1.0)
    assert remaining == 0


def test_consume_linear_ramp_fills_tail_after_completion():
    envelope, current, remaining = consume_linear_ramp(0.0, 1.0, 2, 5)

    np.testing.assert_allclose(envelope, [0.5, 1.0, 1.0, 1.0, 1.0])
    assert current == pytest.approx(1.0)
    assert remaining == 0
