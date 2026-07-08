"""Unit tests for knob callback functionality."""

from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from sonicrack.gui.widgets.knob_style import ProceduralKnobStyle
from sonicrack.gui.widgets.knob_widget import Knob

AUDIO_FREQUENCY_CURVE = (
    (0.0, 11.0),
    (0.2, 40.0),
    (0.5, 282.0),
    (0.8, 1715.0),
    (1.0, 6000.0),
)


def test_knob_callback_on_set_value():
    """Test that callback is called when set_value is used."""
    callback = Mock()
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0,
        callback=callback,
    )

    # Set a new value
    knob.set_value(75.0)

    # Verify callback was called with the new value
    callback.assert_called_once_with(75.0)


def test_knob_callback_multiple_calls():
    """Test that callback is called for each value change."""
    callback = Mock()
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=0.0,
        callback=callback,
    )

    # Change value multiple times
    knob.set_value(25.0)
    knob.set_value(50.0)
    knob.set_value(75.0)

    # Verify callback was called 3 times
    assert callback.call_count == 3
    callback.assert_any_call(25.0)
    callback.assert_any_call(50.0)
    callback.assert_any_call(75.0)


def test_knob_no_callback_on_same_value():
    """Test that callback is not called when setting the same value."""
    callback = Mock()
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0,
        callback=callback,
    )

    # Set the same value
    knob.set_value(50.0)

    # Callback should not be called (value didn't change)
    callback.assert_not_called()


def test_knob_without_callback():
    """Test that knob works without a callback."""
    knob = Knob(label="Test", min_value=0.0, max_value=100.0, default_value=50.0)

    # Should not raise any errors
    knob.set_value(75.0)
    assert knob.get_value() == 75.0


def test_knob_uses_style_geometry():
    """Test that visual styles control knob dimensions."""
    knob = Knob(
        label="Small",
        min_value=0.0,
        max_value=1.0,
        default_value=0.5,
        style=ProceduralKnobStyle.small(),
    )

    assert knob.knob_size == 34
    assert knob.minimumWidth() == 64
    assert knob.minimumHeight() == 66


def test_styled_knob_preserves_value_behavior():
    """Test that style changes do not affect value handling."""
    callback = Mock()
    knob = Knob(
        label="Large",
        min_value=0.0,
        max_value=10.0,
        default_value=5.0,
        style=ProceduralKnobStyle.large(),
        callback=callback,
    )

    knob.set_value(12.0)

    assert knob.get_value() == 10.0
    callback.assert_called_once_with(10.0)


def test_knob_callback_with_clamping():
    """Test that callback receives clamped values."""
    callback = Mock()
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0,
        callback=callback,
    )

    # Set value beyond max
    knob.set_value(150.0)

    # Callback should receive clamped value
    callback.assert_called_once_with(100.0)

    callback.reset_mock()

    # Set value below min
    knob.set_value(-50.0)

    # Callback should receive clamped value
    callback.assert_called_once_with(0.0)


def test_knob_callback_with_db_range():
    """Test callback with dB range (like master gain)."""
    received_values = []

    def on_gain_changed(value):
        received_values.append(value)

    knob = Knob(
        label="Gain",
        min_value=-80.0,
        max_value=12.0,
        default_value=0.0,
        callback=on_gain_changed,
    )

    # Test various gain values
    knob.set_value(-12.0)
    knob.set_value(0.0)
    knob.set_value(6.0)

    assert received_values == [-12.0, 0.0, 6.0]


def test_knob_double_click_reset():
    """Test that double-clicking resets the knob to default value."""
    callback = Mock()
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0,
        callback=callback,
    )

    # Change value away from default
    knob.set_value(75.0)
    assert knob.get_value() == 75.0
    callback.assert_called_with(75.0)
    callback.reset_mock()

    # Simulate double-click at center of widget
    center = knob.rect().center()
    QTest.mouseDClick(
        knob, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center
    )

    # Should reset to default
    assert knob.get_value() == 50.0
    callback.assert_called_with(50.0)


def test_knob_double_click_with_custom_default():
    """Test double-click reset with non-midpoint default."""
    callback = Mock()
    knob = Knob(
        label="Gain",
        min_value=-80.0,
        max_value=12.0,
        default_value=0.0,  # Custom default (not midpoint)
        callback=callback,
    )

    # Change to different value
    knob.set_value(6.0)
    assert knob.get_value() == 6.0
    callback.reset_mock()

    # Double-click should reset to 0.0, not midpoint (-34.0)
    center = knob.rect().center()
    QTest.mouseDClick(
        knob, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center
    )

    assert knob.get_value() == 0.0
    callback.assert_called_with(0.0)


def test_knob_double_click_at_default():
    """Test double-clicking when already at default value."""
    callback = Mock()
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0,
        callback=callback,
    )

    # Already at default, double-click should not call callback
    center = knob.rect().center()
    QTest.mouseDClick(
        knob, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center
    )

    # Callback should not be called since value didn't change
    callback.assert_not_called()
    assert knob.get_value() == 50.0


def test_knob_default_value_storage():
    """Test that default_value is properly stored."""
    knob = Knob(label="Test", min_value=0.0, max_value=100.0, default_value=75.0)

    # Check default value is stored
    assert knob.default_value == 75.0
    assert knob.get_value() == 75.0

    # Change value
    knob.set_value(25.0)
    assert knob.get_value() == 25.0

    # Default should still be stored
    assert knob.default_value == 75.0


def test_knob_default_value_none():
    """Test that default_value defaults to min_value when None."""
    knob = Knob(label="Test", min_value=10.0, max_value=100.0, default_value=None)

    # Should default to min_value
    assert knob.default_value == 10.0
    assert knob.get_value() == 10.0


def test_knob_custom_curve_maps_frequency_anchors():
    """Test custom knob curves map normalized clock positions to values."""
    knob = Knob(
        label="Freq",
        min_value=11.0,
        max_value=6000.0,
        default_value=282.0,
        curve_points=AUDIO_FREQUENCY_CURVE,
    )

    expected_values = {
        0.0: 11.0,
        0.2: 40.0,
        0.5: 282.0,
        0.8: 1715.0,
        1.0: 6000.0,
    }
    for normalized, expected in expected_values.items():
        knob.set_normalized_value(normalized)
        assert knob.get_value() == pytest.approx(expected)


def test_knob_custom_curve_maps_frequency_values_to_positions():
    """Test custom knob curves map values back to normalized positions."""
    knob = Knob(
        label="Freq",
        min_value=11.0,
        max_value=6000.0,
        default_value=282.0,
        curve_points=AUDIO_FREQUENCY_CURVE,
    )

    expected_positions = {
        11.0: 0.0,
        40.0: 0.2,
        282.0: 0.5,
        1715.0: 0.8,
        6000.0: 1.0,
    }
    for value, expected in expected_positions.items():
        knob.set_value(value)
        assert knob.get_normalized_value() == pytest.approx(expected)


def test_knob_double_click_with_min_default():
    """Test double-click reset when default is min_value."""
    knob = Knob(label="Test", min_value=0.0, max_value=100.0, default_value=0.0)

    # Change to max
    knob.set_value(100.0)
    assert knob.get_value() == 100.0

    # Double-click should reset to min (0.0)
    center = knob.rect().center()
    QTest.mouseDClick(
        knob, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center
    )

    assert knob.get_value() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
