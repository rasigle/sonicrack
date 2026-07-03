"""Tests for acid-style filter components."""

import numpy as np

from src.engine.dsp.filters.acid_303 import AcidResonantFilter
from src.engine.dsp.filters.butterworth import BiquadResonantFilter


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


def test_biquad_modulated_coefficients_match_scalar_design():
    filt = BiquadResonantFilter(sample_rate=44100)
    cutoffs = np.array([100.0, 1000.0, 5000.0], dtype=np.float32)
    resonances = np.array([0.7, 2.0, 8.0], dtype=np.float32)

    b0, b1, b2, a1, a2 = filt._design_filter_values(cutoffs, resonances)

    for index, (cutoff, resonance) in enumerate(zip(cutoffs, resonances)):
        b, a = filt._design_filter(float(cutoff), float(resonance))
        np.testing.assert_allclose([b0[index], b1[index], b2[index]], b)
        np.testing.assert_allclose([1.0, a1[index], a2[index]], a)
