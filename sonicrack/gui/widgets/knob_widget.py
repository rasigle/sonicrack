"""Rotary knob widget for parameter control."""

import math
from collections.abc import Callable, Sequence

from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QWidget

from sonicrack.gui.widgets.knob_style import KnobStyle, ProceduralKnobStyle


class Knob(QWidget):
    """A rotary knob widget for continuous parameter control.

    Similar to hardware synth knobs with visual feedback.
    """

    value_changed = QtCore.pyqtSignal(float)  # Emits normalized value (0.0 to 1.0)

    def __init__(
        self,
        label: str = "",
        description: str | None = None,
        min_value: float = 0.0,
        max_value: float = 1.0,
        default_value: float | None = None,
        logarithmic: bool = False,
        curve_points: Sequence[tuple[float, float]] | None = None,
        style: KnobStyle | None = None,
        callback: Callable[[float], None] | None = None,
        parent: QWidget | None = None,
    ):
        """Initialize the knob.

        Args:
            label: Text label displayed below the knob
            description: Tooltip description for the knob
            min_value: Minimum value
            max_value: Maximum value
            default_value: Default value (defaults to min_value)
            logarithmic: If True, use logarithmic scaling (useful for frequency)
            curve_points: Optional normalized/value anchors for custom scaling.
            style: Optional visual style controlling painting and size.
            callback: Optional callback function called with the new value when changed
            parent: Parent widget
        """
        super().__init__(parent)

        self.label = label
        self.min_value = min_value
        self.max_value = max_value
        self.logarithmic = logarithmic
        self.curve_points = self._validate_curve_points(curve_points)
        self.callback = callback
        self.default_value = default_value if default_value is not None else min_value
        self._value = self.default_value
        self.knob_style = style or ProceduralKnobStyle.medium()

        self._description = description
        if self._description:
            self.setToolTip(self._description)

        # Visual properties
        geometry = self.knob_style.geometry
        self.knob_size = geometry.knob_size
        self.setMinimumSize(geometry.min_width, geometry.min_height)
        self.setMaximumSize(geometry.max_width, geometry.max_height)

        # Interaction state
        self.dragging = False
        self.last_y = 0
        self.last_x = 0

        # Angle range (This creates a 270-degree arc with the gap at the bottom)
        # Inverted: open zone at bottom
        self.min_angle = 225  # Bottom-right (starting point)
        self.max_angle = -45  # Bottom-left (ending point)

    @staticmethod
    def _validate_curve_points(
        curve_points: Sequence[tuple[float, float]] | None,
    ) -> tuple[tuple[float, float], ...] | None:
        if curve_points is None:
            return None
        points = tuple(
            (float(position), float(value)) for position, value in curve_points
        )
        if len(points) < 2:
            raise ValueError("curve_points must contain at least two anchors")
        if points[0][0] != 0.0 or points[-1][0] != 1.0:
            raise ValueError("curve_points must start at 0.0 and end at 1.0")
        previous_position = -math.inf
        previous_value = -math.inf
        for position, value in points:
            if not 0.0 <= position <= 1.0:
                raise ValueError("curve point positions must be between 0.0 and 1.0")
            if position <= previous_position:
                raise ValueError("curve point positions must be strictly increasing")
            if value <= previous_value:
                raise ValueError("curve point values must be strictly increasing")
            previous_position = position
            previous_value = value
        return points

    @staticmethod
    def _interpolate(start: float, end: float, fraction: float) -> float:
        return start + fraction * (end - start)

    @classmethod
    def _interpolate_value(cls, start: float, end: float, fraction: float) -> float:
        if start > 0.0 and end > 0.0:
            log_start = math.log(start)
            log_end = math.log(end)
            return math.exp(cls._interpolate(log_start, log_end, fraction))
        return cls._interpolate(start, end, fraction)

    def _curve_normalized_to_value(self, norm_value: float) -> float:
        assert self.curve_points is not None
        norm_value = max(0.0, min(1.0, norm_value))
        points = self.curve_points
        for index in range(len(points) - 1):
            start_pos, start_value = points[index]
            end_pos, end_value = points[index + 1]
            if norm_value <= end_pos:
                fraction = (norm_value - start_pos) / (end_pos - start_pos)
                return self._interpolate_value(start_value, end_value, fraction)
        return points[-1][1]

    def _curve_value_to_normalized(self, value: float) -> float:
        assert self.curve_points is not None
        points = self.curve_points
        if value <= points[0][1]:
            return points[0][0]
        if value >= points[-1][1]:
            return points[-1][0]

        for index in range(len(points) - 1):
            start_pos, start_value = points[index]
            end_pos, end_value = points[index + 1]
            if value <= end_value:
                if start_value > 0.0 and end_value > 0.0 and value > 0.0:
                    start = math.log(start_value)
                    end = math.log(end_value)
                    fraction = (math.log(value) - start) / (end - start)
                else:
                    fraction = (value - start_value) / (end_value - start_value)
                return self._interpolate(start_pos, end_pos, fraction)
        return points[-1][0]

    def get_value(self) -> float:
        """Get the current value."""
        return self._value

    def set_value(self, value: float):
        """Set the current value."""
        value = max(self.min_value, min(self.max_value, value))
        if value != self._value:
            self._value = value
            self.update()
            self.value_changed.emit(self.get_normalized_value())
            # Call the callback function if provided
            if self.callback is not None:
                self.callback(self._value)

    def _value_to_normalized(self, value: float) -> float:
        """Convert a value to normalized (0.0-1.0) considering logarithmic scaling.

        Args:
            value: The actual value

        Returns:
            Normalized value (0.0-1.0)
        """
        if self.max_value == self.min_value:
            return 0.0

        if self.curve_points is not None:
            return self._curve_value_to_normalized(value)

        if self.logarithmic:
            # Ensure we don't take log of zero or negative numbers
            if self.min_value <= 0:
                # Shift values to be positive for log calculation
                min_log = 0.0
                max_log = math.log10(self.max_value - self.min_value + 1)
                val_log = math.log10(value - self.min_value + 1)
            else:
                min_log = math.log10(self.min_value)
                max_log = math.log10(self.max_value)
                val_log = math.log10(value)

            return (val_log - min_log) / (max_log - min_log)
        else:
            return (value - self.min_value) / (self.max_value - self.min_value)

    def _normalized_to_value(self, norm_value: float) -> float:
        """Convert normalized value (0.0-1.0) to actual value considering logarithmic
        scaling.

        Args:
            norm_value: Normalized value (0.0-1.0)

        Returns:
            Actual value
        """
        if self.curve_points is not None:
            return self._curve_normalized_to_value(norm_value)

        if self.logarithmic:
            # Ensure we don't take log of zero or negative numbers
            if self.min_value <= 0:
                # Shift values to be positive for log calculation
                min_log = 0.0
                max_log = math.log10(self.max_value - self.min_value + 1)
                val_log = min_log + norm_value * (max_log - min_log)
                return (10**val_log) - 1 + self.min_value
            else:
                min_log = math.log10(self.min_value)
                max_log = math.log10(self.max_value)
                val_log = min_log + norm_value * (max_log - min_log)
                return 10**val_log
        else:
            return self.min_value + norm_value * (self.max_value - self.min_value)

    def get_normalized_value(self) -> float:
        """Get the normalized value (0.0 to 1.0)."""
        return self._value_to_normalized(self._value)

    def set_normalized_value(self, norm_value: float):
        """Set the value using a normalized value (0.0 to 1.0)."""
        value = self._normalized_to_value(norm_value)
        self.set_value(value)

    def paintEvent(self, event):
        """Paint the knob."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.knob_style.paint(painter, self)

    def mousePressEvent(self, event):
        """Handle mouse press to start dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_y = event.pos().y()
            self.last_x = event.pos().x()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.MiddleButton:
            # Reset to default on middle click
            default = (self.min_value + self.max_value) / 2
            self.set_value(default)
            event.accept()

    def mouseMoveEvent(self, event):
        """Handle mouse move to update value."""
        if self.dragging:
            # Support both vertical and horizontal movement
            delta_y = self.last_y - event.pos().y()
            delta_x = event.pos().x() - self.last_x

            # Use the larger delta for more responsive control
            delta = delta_y if abs(delta_y) > abs(delta_x) else delta_x

            if self.logarithmic or self.curve_points is not None:
                # Non-linear scales adjust in normalized space for consistent feel.
                norm_value = self.get_normalized_value()
                norm_delta = delta / 200.0  # Normalized delta
                new_norm_value = max(0.0, min(1.0, norm_value + norm_delta))
                new_value = self._normalized_to_value(new_norm_value)
                self.set_value(new_value)
            else:
                # Linear sensitivity
                sensitivity = (self.max_value - self.min_value) / 200.0
                new_value = self._value + delta * sensitivity
                self.set_value(new_value)

            self.last_y = event.pos().y()
            self.last_x = event.pos().x()

            # Update tooltip to show current value
            self.setToolTip(f"{self.label}: {self._value:.3f}")
            event.accept()

    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop dragging."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if self._description:
                self.setToolTip(self._description)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to reset to default value."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_value(self.default_value)
            self.setToolTip(f"{self.label}: {self.default_value:.3f} (default)")
            if self._description:
                QTimer.singleShot(2000, lambda: self.setToolTip(self._description))
            event.accept()

    def wheelEvent(self, event):
        """Handle mouse wheel for fine adjustment."""
        delta = event.angleDelta().y()

        if self.logarithmic or self.curve_points is not None:
            # Non-linear scales adjust in normalized space.
            norm_value = self.get_normalized_value()
            norm_delta = delta / 2000.0  # Finer adjustment
            new_norm_value = max(0.0, min(1.0, norm_value + norm_delta))
            new_value = self._normalized_to_value(new_norm_value)
            self.set_value(new_value)
        else:
            # Linear fine adjustment
            sensitivity = (self.max_value - self.min_value) / 2000.0
            new_value = self._value + delta * sensitivity
            self.set_value(new_value)
        event.accept()

    def enterEvent(self, event):
        """Handle mouse enter to show interactive cursor."""
        if not self.dragging:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave to restore cursor."""
        if not self.dragging:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)
