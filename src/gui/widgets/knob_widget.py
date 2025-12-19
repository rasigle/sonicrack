"""Rotary knob widget for parameter control."""

import math
from typing import Callable

from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import QWidget


class Knob(QWidget):
    """A rotary knob widget for continuous parameter control.

    Similar to hardware synth knobs with visual feedback.
    """

    value_changed = QtCore.pyqtSignal(float)  # Emits normalized value (0.0 to 1.0)

    def __init__(
        self,
        label: str = "",
        min_value: float = 0.0,
        max_value: float = 1.0,
        default_value: float | None = None,
        logarithmic: bool = False,
        callback: Callable[[float], None] | None = None,
        parent: QWidget | None = None,
    ):
        """Initialize the knob.

        Args:
            label: Text label displayed below the knob
            min_value: Minimum value
            max_value: Maximum value
            default_value: Default value (defaults to min_value)
            logarithmic: If True, use logarithmic scaling (useful for frequency)
            callback: Optional callback function called with the new value when changed
            parent: Parent widget
        """
        super().__init__(parent)

        self.label = label
        self.min_value = min_value
        self.max_value = max_value
        self.logarithmic = logarithmic
        self.callback = callback
        self.default_value = default_value if default_value is not None else min_value
        self._value = self.default_value

        # Visual properties
        self.knob_size = 50
        self.setMinimumSize(70, 90)
        self.setMaximumSize(70, 90)

        # Interaction state
        self.dragging = False
        self.last_y = 0
        self.last_x = 0

        # Angle range (270 degrees of rotation)
        # Inverted: open zone at bottom
        self.min_angle = 225  # Bottom-right (starting point)
        self.max_angle = -45  # Bottom-left (ending point)
        # This creates a 270-degree arc with the gap at the bottom

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

        if self.logarithmic:
            # Ensure we don't take log of zero or negative numbers
            if self.min_value <= 0:
                # Shift values to be positive for log calculation
                min_log = 0
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
        if self.logarithmic:
            # Ensure we don't take log of zero or negative numbers
            if self.min_value <= 0:
                # Shift values to be positive for log calculation
                min_log = 0
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

        # Calculate center position
        center_x = self.width() / 2
        center_y = 35
        radius = self.knob_size / 2

        # Draw outer track (background arc) - inverted with gap at bottom
        painter.setPen(QPen(QColor(60, 60, 60), 4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        track_rect = QRectF(
            center_x - radius - 2,
            center_y - radius - 2,
            (radius + 2) * 2,
            (radius + 2) * 2,
        )
        # Draw 270-degree arc with gap at bottom (from -45° to 225°)
        painter.drawArc(track_rect, int(-45 * 16), int(270 * 16))

        # Draw knob body (solid circle)
        painter.setBrush(QColor(90, 90, 90))
        painter.setPen(QPen(QColor(50, 50, 50), 2))
        painter.drawEllipse(QPointF(center_x, center_y), radius, radius)

        # Draw value arc (fills counter-clockwise from bottom-right)
        norm_value = self.get_normalized_value()
        # Calculate angle: starts at 225° (right side), goes to -45° (left side)
        current_angle = self.min_angle - norm_value * (self.min_angle - self.max_angle)

        # Arc from min_angle (225°) to current position
        painter.setPen(QPen(QColor(100, 180, 255), 4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        value_arc_rect = QRectF(
            center_x - radius - 2,
            center_y - radius - 2,
            (radius + 2) * 2,
            (radius + 2) * 2,
        )
        arc_span = -(self.min_angle - current_angle)  # Negative for counter-clockwise
        painter.drawArc(value_arc_rect, int(self.min_angle * 16), int(arc_span * 16))

        # Draw indicator pointer (from center to edge)

        angle_rad = math.radians(current_angle)

        # Pointer starts from center, points outward
        pointer_start_radius = radius * 0.2
        pointer_end_radius = radius * 0.85

        start_x = center_x + math.cos(angle_rad) * pointer_start_radius
        start_y = center_y - math.sin(angle_rad) * pointer_start_radius
        end_x = center_x + math.cos(angle_rad) * pointer_end_radius
        end_y = center_y - math.sin(angle_rad) * pointer_end_radius

        painter.setPen(
            QPen(
                QColor(255, 200, 50), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap
            )
        )
        painter.drawLine(QPointF(start_x, start_y), QPointF(end_x, end_y))

        # Draw center dot for visual clarity
        painter.setBrush(QColor(70, 70, 70))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(center_x, center_y), 4, 4)

        # Draw label
        painter.setPen(QColor(220, 220, 220))
        font = QFont("Arial", 8, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            QRectF(0, 70, self.width(), 20), Qt.AlignmentFlag.AlignCenter, self.label
        )

        # Draw value text (larger and more visible)
        value_text = f"{self._value:.2f}"
        if abs(self._value) >= 100:
            value_text = f"{self._value:.1f}"
        elif abs(self._value) < 0.01:
            value_text = f"{self._value:.3f}"

        font.setPointSize(8)
        font.setWeight(QFont.Weight.Normal)
        painter.setFont(font)
        painter.setPen(QColor(200, 230, 255))
        painter.drawText(
            QRectF(0, 52, self.width(), 16), Qt.AlignmentFlag.AlignCenter, value_text
        )

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

            if self.logarithmic:
                # For logarithmic scale, adjust in normalized space for consistent feel
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
            event.accept()

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to reset to default value."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_value(self.default_value)
            self.setToolTip(f"{self.label}: {self.default_value:.3f} (default)")
            event.accept()

    def wheelEvent(self, event):
        """Handle mouse wheel for fine adjustment."""
        delta = event.angleDelta().y()

        if self.logarithmic:
            # For logarithmic scale, adjust in normalized space
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
