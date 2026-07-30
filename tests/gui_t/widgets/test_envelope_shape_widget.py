"""Tests for the envelope shape preview widget."""

from __future__ import annotations

from typing import Any

import pytest

from sonicrack.gui.modules.modulated_source.envelope_adsr import ADSRModule
from sonicrack.gui.modules.modulated_source.envelope_decay import DecayEnvelopeModule
from sonicrack.gui.widgets.envelope_shape_widget import EnvelopeShapeWidget
from sonicrack.patching.module import ModuleCategory


def test_shape_points_include_attack_peak_sustain_and_release():
    points = EnvelopeShapeWidget.shape_points(
        attack=0.1,
        decay=0.2,
        sustain=0.5,
        release=0.3,
    )

    assert points[0] == (0.0, 0.0)
    assert points[1][1] == pytest.approx(1.0)
    assert points[2][1] == pytest.approx(0.5)
    assert points[3][1] == pytest.approx(0.5)
    assert points[-1][1] == pytest.approx(0.0)
    assert points[-1][0] == pytest.approx(1.0)
    # Time progresses left to right.
    xs = [p[0] for p in points]
    assert xs == sorted(xs)


def test_ad_shape_points_peak_and_return_to_zero():
    points = EnvelopeShapeWidget.ad_shape_points(attack=0.1, decay=0.3, amount=0.8)

    assert points[0] == (0.0, 0.0)
    assert points[1][0] == pytest.approx(0.25)
    assert points[1][1] == pytest.approx(0.8)
    assert points[-1] == (1.0, 0.0)


def test_phase_to_xy_adsr_segments():
    points = EnvelopeShapeWidget.shape_points(0.1, 0.2, 0.5, 0.3)

    attack_xy = EnvelopeShapeWidget.phase_to_xy("attack", 0.5, 0.6, points, "adsr")
    assert points[0][0] < attack_xy[0] < points[1][0]
    assert attack_xy[1] == pytest.approx(0.6)

    sustain_xy = EnvelopeShapeWidget.phase_to_xy("sustain", 0.0, 0.5, points, "adsr")
    assert points[2][0] <= sustain_xy[0] <= points[3][0]
    assert sustain_xy[1] == pytest.approx(0.5)

    release_xy = EnvelopeShapeWidget.phase_to_xy("release", 1.0, 0.0, points, "adsr")
    assert release_xy[0] == pytest.approx(points[4][0])


def test_phase_to_xy_ad_segments():
    points = EnvelopeShapeWidget.ad_shape_points(0.2, 0.2, amount=1.0)

    attack_xy = EnvelopeShapeWidget.phase_to_xy("attack", 1.0, 1.0, points, "ad")
    assert attack_xy[0] == pytest.approx(points[1][0])
    assert attack_xy[1] == pytest.approx(1.0)

    decay_xy = EnvelopeShapeWidget.phase_to_xy("decay", 1.0, 0.0, points, "ad")
    assert decay_xy[0] == pytest.approx(1.0)


def test_shape_widget_updates_from_set_envelope(qapp: Any):
    del qapp
    widget = EnvelopeShapeWidget()
    widget.set_envelope(0.2, 0.4, 0.25, 0.5)
    assert widget.kind() == "adsr"
    assert widget.envelope() == pytest.approx((0.2, 0.4, 0.25, 0.5))


def test_shape_widget_ad_mode_and_live_indicator(qapp: Any):
    del qapp
    widget = EnvelopeShapeWidget()
    widget.set_ad_envelope(0.05, 0.15, amount=0.9)
    assert widget.kind() == "ad"
    assert widget.ad_envelope() == pytest.approx((0.05, 0.15, 0.9))

    assert widget.indicator() is None
    widget.set_live_state("attack", 0.5, 0.45)
    indicator = widget.indicator()
    assert indicator is not None
    assert 0.0 < indicator[0] < 1.0
    assert indicator[1] == pytest.approx(0.45)

    widget.clear_live_state()
    assert widget.indicator() is None


def test_adsr_module_shape_tracks_knobs_and_parameters(qapp: Any):
    del qapp
    module = ADSRModule()

    assert module.metadata.category == ModuleCategory.ENVELOPE
    assert module.shape_widget.envelope() == pytest.approx(
        (
            module.attack_knob.get_value(),
            module.decay_knob.get_value(),
            module.sustain_knob.get_value(),
            module.release_knob.get_value(),
        )
    )

    module.set_parameters(
        {
            "attack_duration": 0.5,
            "decay_duration": 0.8,
            "sustain_level": 0.2,
            "release_duration": 1.0,
        }
    )

    assert module.shape_widget.envelope() == pytest.approx((0.5, 0.8, 0.2, 1.0))


def test_adsr_module_live_position_tracks_component(qapp: Any):
    del qapp
    module = ADSRModule()
    module.set_parameters(
        {
            "attack_duration": 0.5,
            "decay_duration": 0.5,
            "sustain_level": 0.5,
            "release_duration": 0.5,
        }
    )
    adsr = module._current_adsr()
    adsr.trigger_note_on()
    adsr.get_samples(100)

    module._update_live_position()
    indicator = module.shape_widget.indicator()
    assert indicator is not None
    assert indicator[1] > 0.0


def test_decay_module_shape_tracks_knobs_and_category(qapp: Any):
    del qapp
    module = DecayEnvelopeModule()

    assert module.metadata.category == ModuleCategory.ENVELOPE
    assert module.shape_widget.kind() == "ad"
    assert module.shape_widget.ad_envelope() == pytest.approx(
        (
            module.attack_knob.get_value(),
            module.decay_knob.get_value(),
            module.amount_knob.get_value(),
        )
    )

    module.set_parameters(
        {
            "attack_duration": 0.05,
            "decay_duration": 0.4,
            "amount": 0.75,
        }
    )
    assert module.shape_widget.ad_envelope() == pytest.approx((0.05, 0.4, 0.75))


def test_decay_module_live_position_tracks_component(qapp: Any):
    del qapp
    module = DecayEnvelopeModule()
    module.set_parameters(
        {
            "attack_duration": 0.05,
            "decay_duration": 0.2,
            "amount": 1.0,
        }
    )
    module.component.attack_duration = 0.05
    module.component.decay_duration = 0.2
    module.component.amount = 1.0
    module.component.trigger_note_on()
    module.component.get_samples(50)

    module._update_live_position()
    indicator = module.shape_widget.indicator()
    assert indicator is not None
    assert indicator[1] > 0.0
