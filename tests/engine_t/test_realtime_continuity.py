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
    WaveAdder,
)
from src.engine.dsp.filters.butterworth import BiquadResonantFilter, ButterworthFilter
from src.engine.presets import PresetBuilder

CHUNKS = [64, 128, 320, 511, 1024, 2049]
TOTAL_SAMPLES = sum(CHUNKS)


def _render_chunked(render: Callable[[int], np.ndarray]) -> np.ndarray:
    return np.concatenate([render(chunk_size) for chunk_size in CHUNKS])


def _render_chunked_total(
    render: Callable[[int], np.ndarray], total_samples: int
) -> np.ndarray:
    chunks = []
    remaining = total_samples
    index = 0
    while remaining > 0:
        chunk_size = min(CHUNKS[index % len(CHUNKS)], remaining)
        chunks.append(render(chunk_size))
        remaining -= chunk_size
        index += 1
    return np.concatenate(chunks)


def _trigger(component, method_name: str) -> None:
    method = getattr(component, method_name, None)
    if callable(method):
        method()

    oscillator = getattr(component, "oscillator", None)
    if oscillator is not None:
        _trigger(oscillator, method_name)

    for child_name in ("modulators", "modifiers", "generators"):
        children = getattr(component, child_name, ())
        for child in children:
            _trigger(child, method_name)


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


def test_butterworth_filter_preserves_multichannel_state_across_buffers():
    rng = np.random.default_rng(9012)
    signal = rng.normal(0.0, 0.25, (TOTAL_SAMPLES, 4)).astype(np.float32)

    continuous_filter = ButterworthFilter(cutoff=1800, order=4, sample_rate=48000)
    chunked_filter = ButterworthFilter(cutoff=1800, order=4, sample_rate=48000)

    continuous = continuous_filter.scale_vectorized(signal)

    offset = 0
    chunks = []
    for chunk_size in CHUNKS:
        chunk = signal[offset : offset + chunk_size]
        chunks.append(chunked_filter.scale_vectorized(chunk))
        offset += chunk_size
    chunked = np.concatenate(chunks)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


def test_biquad_resonant_filter_preserves_state_across_buffers():
    rng = np.random.default_rng(2468)
    signal = rng.normal(0.0, 0.25, TOTAL_SAMPLES).astype(np.float32)

    continuous_filter = BiquadResonantFilter(
        cutoff=1200,
        resonance=3.0,
        filter_type="low",
        output_gain_db=-6.0,
        sample_rate=44100,
    )
    chunked_filter = BiquadResonantFilter(
        cutoff=1200,
        resonance=3.0,
        filter_type="low",
        output_gain_db=-6.0,
        sample_rate=44100,
    )

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


def test_nested_composer_graph_preserves_state_across_buffers():
    def make_graph() -> Chain:
        bass = Chain(
            SawtoothOscillator(frequency=110, gain_db=-15, sample_rate=44100),
            ButterworthFilter(cutoff=900, order=3, sample_rate=44100),
            Volume(gain_db=-3),
        )
        shimmer_lfo = SineOscillator(
            frequency=0.75,
            amplitude=0.25,
            gain_db=None,
            sample_rate=44100,
        )
        shimmer = Chain(
            TriangleOscillator(frequency=330, gain_db=-18, sample_rate=44100),
            ModulatedVolume(shimmer_lfo),
        )
        echo = Delay(
            SquareOscillator(
                frequency=220,
                gain_db=-20,
                pulsewidth=0.42,
                mode="bandlimited",
                sample_rate=44100,
            ),
            delay_time=0.006,
            feedback=0.22,
            mix=0.35,
            sample_rate=44100,
        )
        pan_lfo = SineOscillator(
            frequency=1.25,
            amplitude=0.8,
            gain_db=None,
            sample_rate=44100,
        )
        room = Reverb(
            WaveAdder(bass, shimmer, echo, mix_mode="average"),
            room_size=0.45,
            damping=0.35,
            mix=0.25,
            sample_rate=44100,
        )
        return Chain(
            room,
            ButterworthFilter(cutoff=2400, order=4, sample_rate=44100),
            ModulatedPanner(pan_lfo),
            ButterworthFilter(cutoff=3200, order=2, sample_rate=44100),
        )

    continuous = make_graph().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_graph = make_graph()
    chunked = _render_chunked(chunked_graph.get_samples_vectorized)

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


def test_adsr_retrigger_during_release_starts_from_current_level():
    adsr = ADSREnvelope(
        attack_duration=0.01,
        decay_duration=0.01,
        sustain_level=0.5,
        release_duration=0.2,
        sample_rate=1000,
    )

    adsr.trigger_note_on()
    adsr.get_samples(40, mode="vectorized")
    adsr.trigger_note_off()
    release_samples = adsr.get_samples(30, mode="vectorized")
    current_release_level = release_samples[-1]

    adsr.trigger_note_on()
    retriggered_attack = adsr.get_samples(8, mode="vectorized")

    assert retriggered_attack[0] == pytest.approx(current_release_level)
    assert retriggered_attack[0] > 0.0
    assert np.all(np.diff(retriggered_attack) >= 0.0)


def test_adsr_punch_retrigger_smoothly_resets_before_attack():
    adsr = ADSREnvelope(
        attack_duration=0.01,
        decay_duration=0.01,
        sustain_level=0.5,
        release_duration=0.2,
        sample_rate=1000,
        retrigger_mode="punch",
    )

    adsr.trigger_note_on()
    adsr.get_samples(40, mode="vectorized")
    adsr.trigger_note_off()
    release_samples = adsr.get_samples(30, mode="vectorized")

    adsr.trigger_note_on()
    retriggered = adsr.get_samples(8, mode="vectorized")

    assert retriggered[0] == pytest.approx(release_samples[-1])
    assert retriggered[1] < retriggered[0]
    assert retriggered[2] == pytest.approx(0.0)
    assert retriggered[-1] > retriggered[2]


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


@pytest.mark.parametrize("sample_rate", [44100, 48000, 96000])
def test_stateful_chain_preserves_state_across_sample_rates(sample_rate: int):
    def make_chain() -> Chain:
        return Chain(
            SawtoothOscillator(
                frequency=sample_rate / 320,
                gain_db=-16,
                mode="analog",
                sample_rate=sample_rate,
            ),
            Delay(
                delay_time=0.0025,
                feedback=0.25,
                mix=0.35,
                sample_rate=sample_rate,
            ),
            Reverb(
                room_size=0.35,
                damping=0.45,
                mix=0.2,
                sample_rate=sample_rate,
            ),
            ButterworthFilter(
                cutoff=min(3200, sample_rate * 0.2),
                order=3,
                sample_rate=sample_rate,
            ),
        )

    continuous = make_chain().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_chain = make_chain()
    chunked = _render_chunked(chunked_chain.get_samples_vectorized)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("sample_rate", [44100, 48000, 96000])
def test_adsr_note_events_preserve_state_across_sample_rates(sample_rate: int):
    pre_event_samples = int(sample_rate * 0.017)
    post_event_samples = int(sample_rate * 0.041)

    def make_envelope() -> ADSREnvelope:
        envelope = ADSREnvelope(
            attack_duration=0.003,
            decay_duration=0.007,
            sustain_level=0.45,
            release_duration=0.025,
            sample_rate=sample_rate,
        )
        envelope.trigger_note_on()
        return envelope

    continuous_envelope = make_envelope()
    continuous_pre = continuous_envelope.get_samples(
        pre_event_samples, mode="vectorized"
    )
    continuous_envelope.trigger_note_off()
    continuous_post = continuous_envelope.get_samples(
        post_event_samples, mode="vectorized"
    )
    continuous = np.concatenate((continuous_pre, continuous_post))

    chunked_envelope = make_envelope()
    chunked_pre = _render_chunked_total(
        lambda n: chunked_envelope.get_samples(n, mode="vectorized"),
        pre_event_samples,
    )
    chunked_envelope.trigger_note_off()
    chunked_post = _render_chunked_total(
        lambda n: chunked_envelope.get_samples(n, mode="vectorized"),
        post_event_samples,
    )
    chunked = np.concatenate((chunked_pre, chunked_post))

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("sample_rate", [44100, 48000, 96000])
def test_preset_built_event_graph_preserves_state_across_sample_rates(
    sample_rate: int,
):
    pre_event_samples = int(sample_rate * 0.019)
    post_event_samples = int(sample_rate * 0.037)

    def make_patch() -> Chain:
        return (
            PresetBuilder("Cross-rate event graph")
            .set_sample_rate(sample_rate)
            .sine(frequency=sample_rate / 160, gain_db=-12, mode="analog")
            .adsr(
                attack_duration=0.004,
                decay_duration=0.006,
                sustain_level=0.5,
                release_duration=0.022,
            )
            .volume(gain_db=-3, sample_rate=sample_rate, smoothing_time_ms=4.0)
            .panner(position=0.2, sample_rate=sample_rate, smoothing_time_ms=4.0)
            .build()
        )

    continuous_patch = make_patch()
    _trigger(continuous_patch, "trigger_note_on")
    continuous_pre = continuous_patch.get_samples_vectorized(pre_event_samples)
    _trigger(continuous_patch, "trigger_note_off")
    continuous_post = continuous_patch.get_samples_vectorized(post_event_samples)
    continuous = np.concatenate((continuous_pre, continuous_post))

    chunked_patch = make_patch()
    _trigger(chunked_patch, "trigger_note_on")
    chunked_pre = _render_chunked_total(
        chunked_patch.get_samples_vectorized,
        pre_event_samples,
    )
    _trigger(chunked_patch, "trigger_note_off")
    chunked_post = _render_chunked_total(
        chunked_patch.get_samples_vectorized,
        post_event_samples,
    )
    chunked = np.concatenate((chunked_pre, chunked_post))

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("sample_rate", [44100, 48000, 96000])
def test_nested_graph_runtime_parameter_changes_preserve_state_across_buffers(
    sample_rate: int,
):
    segment_lengths = (
        int(sample_rate * 0.011),
        int(sample_rate * 0.013),
        int(sample_rate * 0.017),
    )

    def make_graph() -> tuple[Chain, Delay, Reverb]:
        bass = Chain(
            SawtoothOscillator(
                frequency=sample_rate / 360,
                gain_db=-15,
                mode="analog",
                sample_rate=sample_rate,
            ),
            ButterworthFilter(
                cutoff=min(1400, sample_rate * 0.06),
                order=3,
                sample_rate=sample_rate,
            ),
            Volume(gain_db=-4),
        )
        shimmer_lfo = SineOscillator(
            frequency=0.8,
            amplitude=0.2,
            gain_db=None,
            sample_rate=sample_rate,
        )
        shimmer = Chain(
            TriangleOscillator(
                frequency=sample_rate / 160,
                gain_db=-18,
                mode="analog",
                sample_rate=sample_rate,
            ),
            ModulatedVolume(
                shimmer_lfo,
                sample_rate=sample_rate,
                smoothing_time_ms=3.0,
            ),
        )
        echo = Delay(
            SquareOscillator(
                frequency=sample_rate / 220,
                gain_db=-20,
                pulsewidth=0.41,
                mode="bandlimited",
                sample_rate=sample_rate,
            ),
            delay_time=0.004,
            feedback=0.22,
            mix=0.3,
            sample_rate=sample_rate,
        )
        room = Reverb(
            WaveAdder(bass, shimmer, echo, mix_mode="average"),
            room_size=0.4,
            damping=0.35,
            mix=0.25,
            sample_rate=sample_rate,
        )
        graph = Chain(
            room,
            ButterworthFilter(
                cutoff=min(2600, sample_rate * 0.18),
                order=4,
                sample_rate=sample_rate,
            ),
            Panner(position=0.15, sample_rate=sample_rate, smoothing_time_ms=2.0),
        )
        return graph, echo, room

    def apply_first_change(delay: Delay, reverb: Reverb) -> None:
        delay.feedback = 0.48
        delay.mix = 0.42
        reverb.damping = 0.2

    def apply_second_change(delay: Delay, reverb: Reverb) -> None:
        delay.delay_time = 0.007
        reverb.room_size = 0.72
        reverb.mix = 0.38

    continuous_graph, continuous_delay, continuous_reverb = make_graph()
    continuous_segments = [
        continuous_graph.get_samples_vectorized(segment_lengths[0]),
    ]
    apply_first_change(continuous_delay, continuous_reverb)
    continuous_segments.append(
        continuous_graph.get_samples_vectorized(segment_lengths[1])
    )
    apply_second_change(continuous_delay, continuous_reverb)
    continuous_segments.append(
        continuous_graph.get_samples_vectorized(segment_lengths[2])
    )
    continuous = np.concatenate(continuous_segments)

    chunked_graph, chunked_delay, chunked_reverb = make_graph()
    chunked_segments = [
        _render_chunked_total(
            chunked_graph.get_samples_vectorized,
            segment_lengths[0],
        ),
    ]
    apply_first_change(chunked_delay, chunked_reverb)
    chunked_segments.append(
        _render_chunked_total(
            chunked_graph.get_samples_vectorized,
            segment_lengths[1],
        )
    )
    apply_second_change(chunked_delay, chunked_reverb)
    chunked_segments.append(
        _render_chunked_total(
            chunked_graph.get_samples_vectorized,
            segment_lengths[2],
        )
    )
    chunked = np.concatenate(chunked_segments)

    np.testing.assert_allclose(chunked, continuous, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("audio_rate", "modulator_rate"),
    [(44100, 48000), (48000, 44100), (96000, 48000)],
)
def test_modulation_heavy_graph_with_mixed_component_sample_rates_preserves_state(
    audio_rate: int,
    modulator_rate: int,
):
    def make_graph() -> Chain:
        amplitude_lfo = SineOscillator(
            frequency=3.5,
            amplitude=0.25,
            gain_db=None,
            sample_rate=modulator_rate,
        )
        pan_lfo = SineOscillator(
            frequency=1.25,
            amplitude=0.9,
            gain_db=None,
            sample_rate=modulator_rate,
        )
        carrier = SineOscillator(
            frequency=audio_rate / 190,
            gain_db=-12,
            mode="analog",
            sample_rate=audio_rate,
        )
        modulator = TriangleOscillator(
            frequency=4.0,
            amplitude=0.2,
            gain_db=None,
            mode="analog",
            sample_rate=modulator_rate,
        )
        voice = ModulatedOscillator(
            carrier,
            modulator,
            amp_mod=lambda base_amp, mod_value: base_amp * (0.75 + mod_value),
        )
        return Chain(
            voice,
            ModulatedVolume(
                amplitude_lfo,
                sample_rate=audio_rate,
                smoothing_time_ms=4.0,
            ),
            Delay(
                delay_time=0.003,
                feedback=0.18,
                mix=0.22,
                sample_rate=audio_rate,
            ),
            Reverb(
                room_size=0.32,
                damping=0.48,
                mix=0.18,
                sample_rate=audio_rate,
            ),
            ModulatedPanner(
                pan_lfo,
                sample_rate=audio_rate,
                smoothing_time_ms=4.0,
            ),
        )

    continuous = make_graph().get_samples_vectorized(TOTAL_SAMPLES)
    chunked_graph = make_graph()
    chunked = _render_chunked(chunked_graph.get_samples_vectorized)

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


@pytest.mark.parametrize("mode", ["pure", "analog", "vcv"])
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
