"""Tests for acid-style filter components."""

import numpy as np

from src.engine.filter_303 import AcidResonantFilter


def test_acid_filter_processes_finite_output():
    filt = AcidResonantFilter(sample_rate=44100)
    samples = np.linspace(-0.5, 0.5, 128, dtype=np.float32)

    output = filt.process_modulated(
        samples,
        env_cv=np.linspace(0.0, 1.0, 128, dtype=np.float32),
        accent_cv=np.ones(128, dtype=np.float32),
    )

    assert output.shape == samples.shape
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))


def test_acid_filter_accepts_short_cv_buffers():
    filt = AcidResonantFilter(sample_rate=44100)
    samples = np.ones(16, dtype=np.float32) * 0.1

    output = filt.process_modulated(
        samples,
        cutoff_cv=np.array([0.0, 0.5], dtype=np.float32),
        env_cv=np.array([1.0], dtype=np.float32),
    )

    assert output.shape == samples.shape
    assert np.all(np.isfinite(output))
