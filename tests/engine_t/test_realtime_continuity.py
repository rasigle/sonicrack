"""Realtime buffer-boundary continuity tests for stateful DSP components."""

from collections.abc import Callable

import numpy as np
import pytest

from src.engine import (
    ADSREnvelope,
    Chain,
    Delay,
    ModulatedOscillator,
    ModulatedPanner,
    ModulatedVolume,
    Panner,
    Reverb,
    SawtoothOscillator,
    SineOscillator,
    SquareOscillator,
    TriangleOscillator,
    Volume,
)
from src.engine.filter import ButterworthFilter

CHUNKS = [64, 128, 320, 511, 1024, 2049]
TOTAL_SAMPLES = sum(CHUNKS)


def _render_chunked(render: Callable[[int], np.ndarray]) -> np.ndarray:
    return np.concatenate([render(chunk_size) for chunk_size in CHUNKS])


def test_butterworth_filter_preserves_state_across_buffers():
    rng = np.random.default_rng(1234)
    signal = rng.normal(0.0, 0.25, TOTAL_SAMPLES).astype(np.float32)

    continuous_filter = ButterworthFilter(cutoff=1200, order=4, sample_rate=44100)
    chunked_filter = ButterworthFilter(cutoff=1200, order=4, sample_rate=44100)

    continuous = continuous_filter.scale_vectorized(signal)

    offset = 0
    chunks = []
    for chunk_size in CHUNKS:
        chunk = signal[offset : offset + chunk_size]
        chunks.append(chunked_filter.scale_vectorized(chunk))
        offset += chunk_size
    chunked = np.concatenate(chunks)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


def test_butterworth_filter_preserves_stereo_state_across_buffers():
    rng = np.random.default_rng(5678)
    signal = rng.normal(0.0, 0.25, (TOTAL_SAMPLES, 2)).astype(np.float32)

    continuous_filter = ButterworthFilter(cutoff=1200, order=4, sample_rate=44100)
    chunked_filter = ButterworthFilter(cutoff=1200, order=4, sample_rate=44100)

    continuous = continuous_filter.scale_vectorized(signal)

    offset = 0
    chunks = []
    for chunk_size in CHUNKS:
        chunk = signal[offset : offset + chunk_size]
        chunks.append(chunked_filter.scale_vectorized(chunk))
        offset += chunk_size
    chunked = np.concatenate(chunks)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


def test_stereo_chain_with_filter_preserves_state_across_buffers():
    def make_chain() -> Chain:
        return Chain(
            SineOscillator(frequency=440, gain_db=-12),
            Panner(position=0.35),
            ButterworthFilter(cutoff=1800, order=4, sample_rate=44100),
        )

    continuous = make_chain().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_chain = make_chain()
    chunked = _render_chunked(chunked_chain.get_samples_vectorized)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


def _make_release_ready_adsr() -> ADSREnvelope:
    adsr = ADSREnvelope(
        attack_duration=0.002,
        decay_duration=0.003,
        sustain_level=0.4,
        release_duration=0.05,
        sample_rate=44100,
    )
    adsr.trigger_note_on()
    adsr.get_samples(512, mode="vectorized")
    adsr.trigger_release()
    return adsr


def test_adsr_release_preserves_state_across_buffers():
    continuous_adsr = _make_release_ready_adsr()
    chunked_adsr = _make_release_ready_adsr()

    continuous = continuous_adsr.get_samples(TOTAL_SAMPLES, mode="vectorized")
    chunked = _render_chunked(
        lambda chunk_size: chunked_adsr.get_samples(chunk_size, mode="vectorized")
    )

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


def test_modulated_oscillator_adsr_release_preserves_state_across_buffers():
    def make_voice() -> ModulatedOscillator:
        envelope = _make_release_ready_adsr()
        return ModulatedOscillator(
            SineOscillator(frequency=220, gain_db=-12),
            envelope,
            amp_mod=lambda base_amp, env_value: base_amp * env_value,
        )

    continuous = make_voice().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_voice = make_voice()
    chunked = _render_chunked(chunked_voice.get_samples_vectorized)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


def test_delay_preserves_state_across_buffers():
    def make_delay() -> Delay:
        return Delay(
            SineOscillator(frequency=220, gain_db=-9),
            delay_time=0.01,
            feedback=0.35,
            mix=0.55,
            sample_rate=44100,
        )

    continuous = make_delay().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_delay = make_delay()
    chunked = _render_chunked(chunked_delay.get_samples_vectorized)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


def test_reverb_preserves_state_across_buffers():
    def make_reverb() -> Reverb:
        return Reverb(
            SineOscillator(frequency=330, gain_db=-12),
            room_size=0.65,
            damping=0.35,
            mix=0.4,
            sample_rate=44100,
        )

    continuous = make_reverb().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_reverb = make_reverb()
    chunked = _render_chunked(chunked_reverb.get_samples_vectorized)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", ["pure", "warm", "bright", "analog"])
def test_sine_oscillator_preserves_state_across_buffers(mode: str):
    def make_sine() -> SineOscillator:
        return SineOscillator(
            frequency=137,
            gain_db=-12,
            mode=mode,
            sample_rate=44100,
        )

    continuous = make_sine().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_sine = make_sine()
    chunked = _render_chunked(chunked_sine.get_samples_vectorized)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", ["pure", "analog"])
def test_sawtooth_oscillator_preserves_state_across_buffers(mode: str):
    def make_sawtooth() -> SawtoothOscillator:
        return SawtoothOscillator(
            frequency=173,
            gain_db=-12,
            mode=mode,
            sample_rate=44100,
        )

    continuous = make_sawtooth().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_sawtooth = make_sawtooth()
    chunked = _render_chunked(chunked_sawtooth.get_samples_vectorized)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", ["pure", "analog"])
def test_triangle_oscillator_preserves_state_across_buffers(mode: str):
    def make_triangle() -> TriangleOscillator:
        return TriangleOscillator(
            frequency=211,
            gain_db=-12,
            mode=mode,
            sample_rate=44100,
        )

    continuous = make_triangle().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_triangle = make_triangle()
    chunked = _render_chunked(chunked_triangle.get_samples_vectorized)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "mode",
    ["ideal", "ideal_smooth", "bandlimited", "vcv", "soft", "comparator"],
)
def test_square_oscillator_preserves_state_across_buffers(mode: str):
    def make_square() -> SquareOscillator:
        return SquareOscillator(
            frequency=137,
            gain_db=-12,
            pulsewidth=0.37,
            mode=mode,
            sample_rate=44100,
        )

    continuous = make_square().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_square = make_square()
    chunked = _render_chunked(chunked_square.get_samples_vectorized)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


def test_modulated_oscillator_preserves_state_across_buffers():
    def make_modulated() -> ModulatedOscillator:
        carrier = SineOscillator(frequency=220, gain_db=-12, sample_rate=44100)
        lfo = SineOscillator(frequency=5, amplitude=0.35, gain_db=None)
        return ModulatedOscillator(
            carrier,
            lfo,
            amp_mod=lambda base_amp, mod_value: base_amp * (0.65 + mod_value),
        )

    continuous = make_modulated().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_modulated = make_modulated()
    chunked = _render_chunked(chunked_modulated.get_samples_vectorized)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


def test_chain_with_modulated_volume_preserves_state_across_buffers():
    def make_chain() -> Chain:
        lfo = SineOscillator(frequency=3, amplitude=0.25, gain_db=None)
        return Chain(
            SineOscillator(frequency=330, gain_db=-12),
            ModulatedVolume(lfo),
            Volume(gain_db=-3),
        )

    continuous = make_chain().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_chain = make_chain()
    chunked = _render_chunked(chunked_chain.get_samples_vectorized)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


def test_stereo_chain_with_modulated_panner_preserves_state_across_buffers():
    def make_chain() -> Chain:
        lfo = SineOscillator(frequency=2, amplitude=1.0, gain_db=None)
        return Chain(
            SineOscillator(frequency=440, gain_db=-12),
            ModulatedPanner(lfo),
            Panner(position=0.0),
        )

    continuous = make_chain().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_chain = make_chain()
    chunked = _render_chunked(chunked_chain.get_samples_vectorized)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)
