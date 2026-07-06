"""Test that panner modules properly use engine components."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.engine import Panner
from src.gui.modules.modifier.pan_mod import PannerModule
from src.gui.modules.modifier.pan_simple import SimplePannerModule


def test_simple_panner_creates_engine_component(qapp: Any):
    """Test that SimplePannerModule creates and uses Panner component."""
    del qapp
    module = SimplePannerModule()

    # Verify component is created
    assert hasattr(module, "component"), "Module should have component attribute"
    assert isinstance(module.component, Panner), "Component should be a Panner"


def test_simple_panner_component_position_updates(qapp: Any):
    """Test that SimplePannerModule component position updates work."""
    del qapp
    module = SimplePannerModule()

    # Test that position updates work
    module.component.position = 0.5
    assert module.component.position == pytest.approx(0.5), "Position should be updated"

    # Test another value
    module.component.position = -0.75
    assert module.component.position == pytest.approx(
        -0.75
    ), "Position should be updated"


def test_simple_panner_component_pan_vectorized(qapp: Any):
    """Test that SimplePannerModule component pan_vectorized method works."""
    del qapp
    module = SimplePannerModule()

    # Set position to right
    module.component.position = 0.5

    # Test that pan_vectorized works
    samples = np.ones(100, dtype=np.float32)
    left, right = module.component.pan_vectorized(samples)

    assert left.shape == (100,), "Left channel should have correct shape"
    assert right.shape == (100,), "Right channel should have correct shape"

    # At position 0.5 (right), right channel should have more gain
    assert np.all(right > left), "Right channel should have more gain at position 0.5"


def test_modulated_panner_creates_engine_component(qapp: Any):
    """Test that PannerModule creates and uses Panner component."""
    del qapp
    module = PannerModule()

    # Verify component is created
    assert hasattr(module, "component"), "Module should have component attribute"
    assert isinstance(module.component, Panner), "Component should be a Panner"


def test_modulated_panner_component_position_updates(qapp: Any):
    """Test that PannerModule component position updates work."""
    del qapp
    module = PannerModule()

    # Test that position updates work
    module.component.position = -0.5
    assert module.component.position == pytest.approx(
        -0.5
    ), "Position should be updated"


def test_modulated_panner_component_pan_vectorized(qapp: Any):
    """Test that PannerModule component pan_vectorized method works."""
    del qapp
    module = PannerModule()

    # Set position to left
    module.component.position = -0.5

    # Test that pan_vectorized works
    samples = np.ones(100, dtype=np.float32)
    left, right = module.component.pan_vectorized(samples)

    assert left.shape == (100,), "Left channel should have correct shape"
    assert right.shape == (100,), "Right channel should have correct shape"

    # At position -0.5 (left), left channel should have more gain
    assert np.all(left > right), "Left channel should have more gain at position -0.5"


def test_panner_component_property_returns_target_value(qapp: Any):
    """Test that Panner position property returns target value, not smoothed value."""
    del qapp
    panner = Panner(position=0.0)

    # Set position and immediately read it back
    panner.position = 0.75
    assert panner.position == pytest.approx(
        0.75
    ), "Position property should return target value"

    # Change again
    panner.position = -0.25
    assert panner.position == pytest.approx(
        -0.25
    ), "Position property should return new target value"


def test_panner_constant_power_law(qapp: Any):
    """Test that Panner uses constant-power panning law."""
    del qapp
    panner = Panner(position=0.0)

    # At center, both channels should have equal gain (sqrt(0.5) ≈ 0.707)
    samples = np.ones(100, dtype=np.float32)
    left, right = panner.pan_vectorized(samples)

    # Check power is preserved (left^2 + right^2 should equal input^2)
    input_power = np.sum(samples**2)
    output_power = np.sum(left**2) + np.sum(right**2)
    assert output_power == pytest.approx(
        input_power, rel=0.01
    ), "Constant-power law: output power should equal input power"
