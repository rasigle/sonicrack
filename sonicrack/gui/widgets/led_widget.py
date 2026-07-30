"""LED indicator widgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget


@dataclass(frozen=True, slots=True)
class LedStyle:
    """Visual configuration for an LED indicator."""

    size: int = 12
    off_color: QColor = field(default_factory=lambda: QColor(42, 47, 52))
    on_color: QColor = field(default_factory=lambda: QColor(94, 196, 255))
    border_color: QColor = field(default_factory=lambda: QColor(16, 18, 22))
    off_image: str | Path | None = None
    on_image: str | Path | None = None


class LedIndicator(QWidget):
    """Small on/off indicator with procedural and image-backed rendering."""

    def __init__(
        self,
        *,
        on: bool = False,
        style: LedStyle | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._on = bool(on)
        self.led_style = style or LedStyle()
        self._off_pixmap = self._load_pixmap(self.led_style.off_image)
        self._on_pixmap = self._load_pixmap(self.led_style.on_image)
        self.setFixedSize(self.led_style.size, self.led_style.size)

    @staticmethod
    def _load_pixmap(path: str | Path | None) -> QPixmap | None:
        if path is None:
            return None
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return None
        return pixmap

    def is_on(self) -> bool:
        return self._on

    def set_on(self, on: bool) -> None:
        on = bool(on)
        if on != self._on:
            self._on = on
            self.update()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pixmap = self._on_pixmap if self._on else self._off_pixmap
        if pixmap is not None:
            painter.drawPixmap(self.rect(), pixmap)
            return

        rect = QRectF(1, 1, self.width() - 2, self.height() - 2)
        painter.setPen(QPen(self.led_style.border_color, 1))
        painter.setBrush(
            self.led_style.on_color if self._on else self.led_style.off_color
        )
        painter.drawEllipse(rect)

        if self._on:
            glow = QColor(self.led_style.on_color)
            glow.setAlpha(70)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(QRectF(0, 0, self.width(), self.height()))
