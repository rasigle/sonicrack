"""Unit tests for knob callback functionality."""
from unittest.mock import Mock

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from src.gui.widgets.knob_widget import Knob


@pytest.fixture
def app():
    """Create QApplication instance for testing."""
    return QApplication([])


def test_knob_callback_on_set_value(app):
    """Test that callback is called when set_value is used."""
    callback = Mock()
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0,
        callback=callback
    )

    # Set a new value
    knob.set_value(75.0)

    # Verify callback was called with the new value
    callback.assert_called_once_with(75.0)


def test_knob_callback_multiple_calls(app):
    """Test that callback is called for each value change."""
    callback = Mock()
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=0.0,
        callback=callback
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


def test_knob_no_callback_on_same_value(app):
    """Test that callback is not called when setting the same value."""
    callback = Mock()
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0,
        callback=callback
    )

    # Set the same value
    knob.set_value(50.0)

    # Callback should not be called (value didn't change)
    callback.assert_not_called()


def test_knob_without_callback(app):
    """Test that knob works without a callback."""
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0
    )

    # Should not raise any errors
    knob.set_value(75.0)
    assert knob.get_value() == 75.0


def test_knob_callback_with_clamping(app):
    """Test that callback receives clamped values."""
    callback = Mock()
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0,
        callback=callback
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


def test_knob_callback_with_db_range(app):
    """Test callback with dB range (like master gain)."""
    received_values = []

    def on_gain_changed(value):
        received_values.append(value)

    knob = Knob(
        label="Gain",
        min_value=-80.0,
        max_value=12.0,
        default_value=0.0,
        callback=on_gain_changed
    )

    # Test various gain values
    knob.set_value(-12.0)
    knob.set_value(0.0)
    knob.set_value(6.0)

    assert received_values == [-12.0, 0.0, 6.0]


def test_knob_double_click_reset(app):
    """Test that double-clicking resets the knob to default value."""
    callback = Mock()
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0,
        callback=callback
    )

    # Change value away from default
    knob.set_value(75.0)
    assert knob.get_value() == 75.0
    callback.assert_called_with(75.0)
    callback.reset_mock()

    # Simulate double-click at center of widget
    center = knob.rect().center()
    QTest.mouseDClick(knob, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center)

    # Should reset to default
    assert knob.get_value() == 50.0
    callback.assert_called_with(50.0)


def test_knob_double_click_with_custom_default(app):
    """Test double-click reset with non-midpoint default."""
    callback = Mock()
    knob = Knob(
        label="Gain",
        min_value=-80.0,
        max_value=12.0,
        default_value=0.0,  # Custom default (not midpoint)
        callback=callback
    )

    # Change to different value
    knob.set_value(6.0)
    assert knob.get_value() == 6.0
    callback.reset_mock()

    # Double-click should reset to 0.0, not midpoint (-34.0)
    center = knob.rect().center()
    QTest.mouseDClick(
        knob,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        center
    )

    assert knob.get_value() == 0.0
    callback.assert_called_with(0.0)


def test_knob_double_click_at_default(app):
    """Test double-clicking when already at default value."""
    callback = Mock()
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=50.0,
        callback=callback
    )

    # Already at default, double-click should not call callback
    center = knob.rect().center()
    QTest.mouseDClick(
        knob,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        center
    )

    # Callback should not be called since value didn't change
    callback.assert_not_called()
    assert knob.get_value() == 50.0


def test_knob_default_value_storage(app):
    """Test that default_value is properly stored."""
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=75.0
    )

    # Check default value is stored
    assert knob.default_value == 75.0
    assert knob.get_value() == 75.0

    # Change value
    knob.set_value(25.0)
    assert knob.get_value() == 25.0

    # Default should still be stored
    assert knob.default_value == 75.0


def test_knob_default_value_none(app):
    """Test that default_value defaults to min_value when None."""
    knob = Knob(
        label="Test",
        min_value=10.0,
        max_value=100.0,
        default_value=None
    )

    # Should default to min_value
    assert knob.default_value == 10.0
    assert knob.get_value() == 10.0


def test_knob_double_click_with_min_default(app):
    """Test double-click reset when default is min_value."""
    knob = Knob(
        label="Test",
        min_value=0.0,
        max_value=100.0,
        default_value=0.0
    )

    # Change to max
    knob.set_value(100.0)
    assert knob.get_value() == 100.0

    # Double-click should reset to min (0.0)
    center = knob.rect().center()
    QTest.mouseDClick(knob, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center)

    assert knob.get_value() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

