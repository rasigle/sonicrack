"""Visual styles for rotary knob widgets."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
)


class KnobLike(Protocol):
    """Subset of Knob used by visual styles."""

    label: str
    min_angle: float
    max_angle: float

    def width(self) -> int: ...

    def height(self) -> int: ...

    def get_value(self) -> float: ...

    def get_normalized_value(self) -> float: ...


@dataclass(frozen=True, slots=True)
class KnobGeometry:
    """Layout dimensions for a knob style."""

    knob_size: int = 44
    min_width: int = 82
    min_height: int = 78
    max_width: int = 96
    max_height: int = 82
    center_y: int = 30
    value_text_y: int = 48
    label_y: int = 62
    value_text_height: int = 12
    label_height: int = 14


@dataclass(frozen=True, slots=True)
class KnobPalette:
    """Colors used by the default procedural knob style."""

    track: QColor = field(default_factory=lambda: QColor(72, 78, 86))
    body: QColor = field(default_factory=lambda: QColor(58, 63, 70))
    body_border: QColor = field(default_factory=lambda: QColor(18, 20, 23))
    value_arc: QColor = field(default_factory=lambda: QColor(94, 196, 255))
    pointer: QColor = field(default_factory=lambda: QColor(255, 207, 87))
    center_dot: QColor = field(default_factory=lambda: QColor(22, 24, 28))
    label: QColor = field(default_factory=lambda: QColor(238, 241, 245))
    value_text: QColor = field(default_factory=lambda: QColor(194, 229, 255))


@dataclass(frozen=True, slots=True)
class ProceduralKnobStyle:
    """Qt-painted knob style used by default."""

    geometry: KnobGeometry = field(default_factory=KnobGeometry)
    palette: KnobPalette = field(default_factory=KnobPalette)

    @classmethod
    def small(cls) -> ProceduralKnobStyle:
        return cls(
            geometry=KnobGeometry(
                knob_size=34,
                min_width=64,
                min_height=66,
                max_width=78,
                max_height=72,
                center_y=24,
                value_text_y=39,
                label_y=52,
            )
        )

    @classmethod
    def medium(cls) -> ProceduralKnobStyle:
        return cls()

    @classmethod
    def large(cls) -> ProceduralKnobStyle:
        return cls(
            geometry=KnobGeometry(
                knob_size=58,
                min_width=104,
                min_height=98,
                max_width=124,
                max_height=108,
                center_y=40,
                value_text_y=66,
                label_y=80,
            )
        )

    def paint(self, painter: QPainter, knob: KnobLike) -> None:
        center_x = knob.width() / 2
        center_y = self.geometry.center_y
        radius = self.geometry.knob_size / 2

        self._paint_track(painter, center_x, center_y, radius)
        self._paint_body(painter, center_x, center_y, radius)
        self._paint_value_arc(painter, knob, center_x, center_y, radius)
        self._paint_pointer(painter, knob, center_x, center_y, radius)
        self._paint_center_dot(painter, center_x, center_y)
        self._paint_text(painter, knob)

    def _paint_track(
        self, painter: QPainter, center_x: float, center_y: float, radius: float
    ) -> None:
        painter.setPen(QPen(self.palette.track, 4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        track_rect = QRectF(
            center_x - radius - 2,
            center_y - radius - 2,
            (radius + 2) * 2,
            (radius + 2) * 2,
        )
        painter.drawArc(track_rect, int(-45 * 16), int(270 * 16))

    def _paint_body(
        self, painter: QPainter, center_x: float, center_y: float, radius: float
    ) -> None:
        painter.setBrush(self.palette.body)
        painter.setPen(QPen(self.palette.body_border, 2))
        painter.drawEllipse(QPointF(center_x, center_y), radius, radius)

    def _paint_value_arc(
        self,
        painter: QPainter,
        knob: KnobLike,
        center_x: float,
        center_y: float,
        radius: float,
    ) -> None:
        norm_value = knob.get_normalized_value()
        current_angle = knob.min_angle - norm_value * (knob.min_angle - knob.max_angle)
        value_arc_rect = QRectF(
            center_x - radius - 2,
            center_y - radius - 2,
            (radius + 2) * 2,
            (radius + 2) * 2,
        )
        arc_span = -(knob.min_angle - current_angle)
        painter.setPen(QPen(self.palette.value_arc, 4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(value_arc_rect, int(knob.min_angle * 16), int(arc_span * 16))

    def _paint_pointer(
        self,
        painter: QPainter,
        knob: KnobLike,
        center_x: float,
        center_y: float,
        radius: float,
    ) -> None:
        norm_value = knob.get_normalized_value()
        current_angle = knob.min_angle - norm_value * (knob.min_angle - knob.max_angle)
        angle_rad = math.radians(current_angle)

        pointer_start_radius = radius * 0.2
        pointer_end_radius = radius * 0.85
        start_x = center_x + math.cos(angle_rad) * pointer_start_radius
        start_y = center_y - math.sin(angle_rad) * pointer_start_radius
        end_x = center_x + math.cos(angle_rad) * pointer_end_radius
        end_y = center_y - math.sin(angle_rad) * pointer_end_radius

        painter.setPen(
            QPen(
                self.palette.pointer,
                3,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.drawLine(QPointF(start_x, start_y), QPointF(end_x, end_y))

    def _paint_center_dot(
        self, painter: QPainter, center_x: float, center_y: float
    ) -> None:
        painter.setBrush(self.palette.center_dot)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(center_x, center_y), 4, 4)

    def _paint_text(self, painter: QPainter, knob: KnobLike) -> None:
        painter.setPen(self.palette.label)
        font = QFont("Arial", 8, QFont.Weight.Bold)
        painter.setFont(font)
        label_metrics = QFontMetrics(font)
        label_text = label_metrics.elidedText(
            knob.label,
            Qt.TextElideMode.ElideRight,
            max(10, knob.width() - 6),
        )
        painter.drawText(
            QRectF(
                3,
                self.geometry.label_y,
                knob.width() - 6,
                self.geometry.label_height,
            ),
            Qt.AlignmentFlag.AlignCenter,
            label_text,
        )

        value = knob.get_value()
        value_text = f"{value:.2f}"
        if abs(value) >= 100:
            value_text = f"{value:.1f}"
        elif abs(value) < 0.01:
            value_text = f"{value:.3f}"

        font.setPointSize(7)
        font.setWeight(QFont.Weight.Normal)
        painter.setFont(font)
        value_metrics = QFontMetrics(font)
        value_text = value_metrics.elidedText(
            value_text,
            Qt.TextElideMode.ElideRight,
            max(10, knob.width() - 8),
        )
        painter.setPen(self.palette.value_text)
        painter.drawText(
            QRectF(
                4,
                self.geometry.value_text_y,
                knob.width() - 8,
                self.geometry.value_text_height,
            ),
            Qt.AlignmentFlag.AlignCenter,
            value_text,
        )


@dataclass(frozen=True, slots=True)
class ImageKnobStyle:
    """Image-backed knob style with an optional rotating pointer layer."""

    body_path: str | Path
    pointer_path: str | Path | None = None
    geometry: KnobGeometry = field(default_factory=KnobGeometry)
    label_painter: Callable[[QPainter, KnobLike], None] | None = None

    def paint(self, painter: QPainter, knob: KnobLike) -> None:
        center_x = knob.width() / 2
        center_y = self.geometry.center_y
        size = self.geometry.knob_size
        top_left_x = center_x - size / 2
        top_left_y = center_y - size / 2

        body = QPixmap(str(self.body_path))
        if not body.isNull():
            painter.drawPixmap(
                QRectF(top_left_x, top_left_y, size, size).toRect(),
                body,
            )

        if self.pointer_path is not None:
            pointer = QPixmap(str(self.pointer_path))
            if not pointer.isNull():
                norm_value = knob.get_normalized_value()
                current_angle = knob.min_angle - norm_value * (
                    knob.min_angle - knob.max_angle
                )
                transform = QTransform()
                transform.translate(center_x, center_y)
                transform.rotate(-current_angle)
                transform.translate(-center_x, -center_y)
                painter.setTransform(transform, combine=True)
                painter.drawPixmap(
                    QRectF(top_left_x, top_left_y, size, size).toRect(),
                    pointer,
                )
                painter.resetTransform()

        if self.label_painter is not None:
            self.label_painter(painter, knob)
        else:
            ProceduralKnobStyle(self.geometry)._paint_text(painter, knob)


KnobStyle = ProceduralKnobStyle | ImageKnobStyle
