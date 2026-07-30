"""Tests for simple and modulated Volume modules."""

from __future__ import annotations

import math

import pytest

from sonicrack.gui.modules.modifier.volume_simple import SimpleVolumeModule


def test_simple_volume_set_parameters_negative_gain_db(qapp):
    """Negative gain_db must update Volume.gain_db without crashing.

    Regression: the gain knob callback used to write dB into amplitude
    (linear, must be >= 0), which aborted loading patches like demo_wobble_bass.
    """
    del qapp
    module = SimpleVolumeModule()

    module.set_parameters({"active": True, "gain_db": -8.0})

    assert module.get_parameters()["gain_db"] == pytest.approx(-8.0)
    assert module.component is not None
    assert module.component.gain_db == pytest.approx(-8.0)
    # Linear amplitude for -8 dB is 10^(-8/20) ≈ 0.398
    assert module.component.amplitude == pytest.approx(math.pow(10.0, -8.0 / 20.0))


def test_simple_volume_knob_change_updates_gain_db(qapp):
    del qapp
    module = SimpleVolumeModule()
    module.gain_knob.set_value(-12.0)

    assert module.component is not None
    assert module.component.gain_db == pytest.approx(-12.0)
