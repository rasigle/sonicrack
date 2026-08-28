"""Compact peak level meter for mixer channels and output monitoring."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class LevelMeter(QWidget):
    """Vertical peak meter with a short hold needle.

    Audio threads should only write :meth:`set_level`. The GUI thread paints
    from the last stored peak so meter drawing never runs on the DSP path.
    """

    def __init__(
        self,
        *,
        width: int = 10,
        height: int = 48,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._level = 0.0
        self._hold = 0.0
        self.setFixedSize(width, height)
        self.setToolTip("Peak level")

    def set_level(self, level: float) -> None:
        """Set instantaneous peak in ``[0, 1]`` (clamped)."""
        value = float(max(0.0, min(1.0, level)))
        self._level = value
        if value > self._hold:
            self._hold = value
        self.update()

    def decay(self, amount: float = 0.08) -> None:
        """Fall the hold needle toward the current level (call from a UI timer)."""
        if self._hold > self._level:
            self._hold = max(self._level, self._hold - amount)
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(1, 1, self.width() - 2, self.height() - 2)
        painter.setPen(QPen(QColor(16, 18, 22), 1))
        painter.setBrush(QColor(22, 26, 30))
        painter.drawRoundedRect(rect, 2, 2)

        fill_h = rect.height() * self._level
        if fill_h > 0.5:
            fill = QRectF(rect.left(), rect.bottom() - fill_h, rect.width(), fill_h)
            gradient = QLinearGradient(fill.bottomLeft(), fill.topLeft())
            gradient.setColorAt(0.0, QColor(70, 190, 110))
            gradient.setColorAt(0.7, QColor(220, 190, 70))
            gradient.setColorAt(1.0, QColor(230, 80, 70))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawRoundedRect(fill, 2, 2)

        hold_y = rect.bottom() - rect.height() * self._hold
        painter.setPen(QPen(QColor(240, 240, 245), 1))
        painter.drawLine(QPointF(rect.left(), hold_y), QPointF(rect.right(), hold_y))
