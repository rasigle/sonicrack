"""Tests for the ADSR envelope shape preview widget."""

from __future__ import annotations

from typing import Any

import pytest

from sonicrack.gui.modules.modulated_source.envelope_adsr import ADSRModule
from sonicrack.gui.widgets.envelope_shape_widget import EnvelopeShapeWidget


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


def test_shape_widget_updates_from_set_envelope(qapp: Any):
    del qapp
    widget = EnvelopeShapeWidget()
    widget.set_envelope(0.2, 0.4, 0.25, 0.5)
    assert widget.envelope() == pytest.approx((0.2, 0.4, 0.25, 0.5))


def test_adsr_module_shape_tracks_knobs_and_parameters(qapp: Any):
    del qapp
    module = ADSRModule()

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
