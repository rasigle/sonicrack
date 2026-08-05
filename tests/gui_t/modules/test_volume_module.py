"""Tests for the Volume module (gain control with optional modulation)."""

from __future__ import annotations

import math

import pytest

from sonicrack.gui.modules.modifier.volume import VolumeModule


def test_volume_set_parameters_negative_gain_db(qapp):
    """Negative gain_db must update Volume.gain_db without crashing.

    Regression: the gain knob callback used to write dB into amplitude
    (linear, must be >= 0), which aborted loading patches like demo_wobble_bass.
    """
    del qapp
    module = VolumeModule()

    module.set_parameters({"active": True, "gain_db": -8.0})

    assert module.get_parameters()["gain_db"] == pytest.approx(-8.0)
    assert module.component is not None
    assert module.component.gain_db == pytest.approx(-8.0)
    # Linear amplitude for -8 dB is 10^(-8/20) ≈ 0.398
    assert module.component.amplitude == pytest.approx(math.pow(10.0, -8.0 / 20.0))


def test_volume_knob_change_updates_gain_db(qapp):
    del qapp
    module = VolumeModule()
    module.gain_knob.set_value(-12.0)

    assert module.component is not None
    assert module.component.gain_db == pytest.approx(-12.0)


def test_volume_has_mod_port(qapp):
    """Combined Volume module always exposes a Mod input for optional CV."""
    del qapp
    module = VolumeModule()
    assert "Mod" in module.inputs
    assert "In" in module.inputs
    assert "Out" in module.outputs
