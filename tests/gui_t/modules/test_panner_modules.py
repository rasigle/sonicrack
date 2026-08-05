"""Test that the Panner module properly uses engine components."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from soniclab import Panner

from sonicrack.gui.modules.modifier.pan import PannerModule


def test_panner_creates_engine_component(qapp: Any):
    """Test that PannerModule creates and uses Panner component."""
    del qapp
    module = PannerModule()

    assert hasattr(module, "component"), "Module should have component attribute"
    assert isinstance(module.component, Panner), "Component should be a Panner"
    assert "Mod" in module.inputs


def test_panner_component_position_updates(qapp: Any):
    """Test that PannerModule component position updates work."""
    del qapp
    module = PannerModule()

    module.component.position = 0.5
    assert module.component.position == pytest.approx(0.5)

    module.component.position = -0.75
    assert module.component.position == pytest.approx(-0.75)


def test_panner_component_pan_vectorized(qapp: Any):
    """Test that PannerModule component pan_vectorized method works."""
    del qapp
    module = PannerModule()

    module.component.position = 0.5

    samples = np.ones(100, dtype=np.float32)
    left, right = module.component.pan_vectorized(samples)

    assert left.shape == (100,), "Left channel should have correct shape"
    assert right.shape == (100,), "Right channel should have correct shape"

    # At position 0.5 (right), right channel should have more gain
    assert np.all(right > left), "Right channel should have more gain at position 0.5"


def test_panner_knob_updates_component_when_unmodulated(qapp: Any):
    """Knob should update engine component when Mod is not connected."""
    del qapp
    module = PannerModule()
    module.pan_knob.set_value(-0.5)
    assert module.component.position == pytest.approx(-0.5)


def test_panner_component_property_returns_target_value(qapp: Any):
    """Test that Panner position property returns target value, not smoothed value."""
    del qapp
    panner = Panner(position=0.0)

    panner.position = 0.75
    assert panner.position == pytest.approx(0.75)

    panner.position = -0.25
    assert panner.position == pytest.approx(-0.25)


def test_panner_constant_power_law(qapp: Any):
    """Test that Panner uses constant-power panning law."""
    del qapp
    panner = Panner(position=0.0)

    samples = np.ones(100, dtype=np.float32)
    left, right = panner.pan_vectorized(samples)

    input_power = np.sum(samples**2)
    output_power = np.sum(left**2) + np.sum(right**2)
    assert output_power == pytest.approx(input_power, rel=0.01)
