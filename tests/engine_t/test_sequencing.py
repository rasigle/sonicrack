"""Tests for reusable sequencing primitives."""

import numpy as np
import pytest

from src.engine.sequencing import (
    AccentProcessor,
    Behringer182Sequencer,
    SlideProcessor,
    StepClock,
    StepEvent,
    StepSequencer,
)
from src.engine.utils.cv import midi_note_to_pitch_cv


def test_step_clock_emits_initial_and_subsequent_pulses():
    clock = StepClock(bpm=60.0, division="1/4", sample_rate=10)

    first = clock.process(12)

    np.testing.assert_allclose(first, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])


def test_step_clock_preserves_phase_across_buffers():
    clock = StepClock(bpm=60.0, division="1/4", sample_rate=10)

    first = clock.process(6)
    second = clock.process(6)

    np.testing.assert_allclose(first, [1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(second, [0, 0, 0, 0, 1, 0])


def test_step_sequencer_renders_pitch_gate_accent_and_slide():
    sequencer = StepSequencer(
        [
            StepEvent(36, accent=True),
            StepEvent(None, gate=False),
            StepEvent(39, slide=True),
        ],
        bpm=60.0,
        division="1/4",
        sample_rate=4,
    )

    frame = sequencer.process(10)

    np.testing.assert_allclose(frame.frequency[:4], midi_note_to_pitch_cv(36))
    np.testing.assert_allclose(frame.gate[:4], [1, 1, 1, 0])
    np.testing.assert_allclose(frame.accent[:4], 1.0)
    np.testing.assert_allclose(frame.gate[4:8], 0.0)
    np.testing.assert_allclose(frame.frequency[8:], midi_note_to_pitch_cv(39))
    np.testing.assert_allclose(frame.slide[8:], 1.0)


def test_step_sequencer_accepts_external_clock_pulses():
    sequencer = StepSequencer(
        [StepEvent(36), StepEvent(48)],
        sample_rate=10,
    )
    clock = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)

    frame = sequencer.process(6, clock)

    np.testing.assert_allclose(frame.frequency[:3], midi_note_to_pitch_cv(36))
    np.testing.assert_allclose(frame.frequency[3:], midi_note_to_pitch_cv(48))


def test_slide_processor_smooths_only_when_slide_is_active():
    slide = SlideProcessor(time=0.25, sample_rate=4)

    output = slide.process(
        np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float32),
        np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )

    np.testing.assert_allclose(output, [0.0, 1.0, 1.0, 2.0])


def test_slide_processor_uses_gradual_smoothing_for_longer_times():
    slide = SlideProcessor(time=1.0, sample_rate=4)

    output = slide.process(
        np.full(4, 1.0, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    )

    assert output[0] == pytest.approx(1.0)
    second = slide.process(
        np.full(4, 0.0, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    )
    assert second[0] < 1.0
    assert second[0] > 0.0
    assert second[-1] > 0.0


def test_accent_processor_shapes_depth_outputs():
    accent = AccentProcessor(
        amount=1.0,
        decay=0.25,
        amp_depth=0.25,
        cutoff_depth=0.5,
        envelope_depth=0.75,
        sample_rate=4,
    )

    frame = accent.process(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))

    np.testing.assert_allclose(frame.amp, [0.25, 0.25, 0.0, 0.0])
    np.testing.assert_allclose(frame.cutoff, [0.5, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(frame.envelope, [0.75, 0.75, 0.0, 0.0])


def test_accent_processor_preserves_decay_across_buffers():
    accent = AccentProcessor(amount=1.0, decay=1.0, amp_depth=1.0, sample_rate=4)

    first = accent.process(np.array([1.0], dtype=np.float32))
    second = accent.process(np.zeros(2, dtype=np.float32))

    np.testing.assert_allclose(first.amp, [1.0])
    np.testing.assert_allclose(second.amp, [0.75, 0.5])


def test_behringer_182_renders_dual_cv_rows_gate_and_triggers():
    sequencer = Behringer182Sequencer(
        cv_a=[0.0, 0.5, 1.0],
        cv_b=[1.0, 0.5, 0.0],
        gates=[True, False, True],
        steps=3,
        gate_length=1.0,
        cv_a_range=5.0,
        cv_b_range=2.0,
        bpm=60.0,
        division="1/4",
        sample_rate=10,
    )
    clock = np.array([1, 0, 1, 0, 1, 0, 1], dtype=np.float32)

    frame = sequencer.process(7, clock)

    np.testing.assert_allclose(frame.cv_a, [0.0, 0.0, 2.5, 2.5, 5.0, 5.0, 0.0])
    np.testing.assert_allclose(frame.cv_b, [2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 2.0])
    np.testing.assert_allclose(frame.gate, [1, 1, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(frame.trigger, [1, 0, 0, 0, 1, 0, 1])
    np.testing.assert_allclose(frame.end, [0, 0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(frame.step, [0, 0, 1, 1, 2, 2, 0])


def test_behringer_182_reset_and_hold_control_position():
    sequencer = Behringer182Sequencer(
        cv_a=[0.0, 0.25, 0.5],
        steps=3,
        cv_a_range=4.0,
        sample_rate=10,
    )
    clock = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    reset = np.array([0, 0, 1, 0, 0], dtype=np.float32)
    hold = np.array([0, 1, 0, 0, 0], dtype=np.float32)

    frame = sequencer.process(5, clock, reset_pulses=reset, hold_signal=hold)

    np.testing.assert_allclose(frame.cv_a, [0.0, 0.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(frame.step, [0, 0, 0, 1, 2])


def test_behringer_182_is_registered_component():
    from src.engine.core.registry import audio_registry

    assert audio_registry.get("Behringer182Sequencer") is Behringer182Sequencer
