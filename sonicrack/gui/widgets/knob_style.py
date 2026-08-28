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

from sonicrack.gui.widgets.skin import (
    KNOB_DAVIES,
    KNOB_METAL,
    load_path_pixmap,
    load_skin_pixmap,
    skin_available,
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
    """Image-backed knob style with a rotating body and/or pointer layer.

    Packaged resources are cached. ``image_zero_angle`` is the math angle
    (degrees, 90 = up) of the graphic at identity transform so Davies caps
    painted at 12 o'clock line up with the 270-degree sweep.
    """

    body_path: str | Path | None = None
    pointer_path: str | Path | None = None
    body_resource: tuple[str, ...] | None = None
    pointer_resource: tuple[str, ...] | None = None
    geometry: KnobGeometry = field(default_factory=KnobGeometry)
    rotate_body: bool = False
    image_zero_angle: float = 90.0
    show_value_arc: bool = True
    draw_pointer: bool = True
    label_painter: Callable[[QPainter, KnobLike], None] | None = None

    def _pixmap(
        self,
        resource: tuple[str, ...] | None,
        path: str | Path | None,
    ) -> QPixmap | None:
        if resource is not None:
            return load_skin_pixmap(*resource)
        if path is not None:
            return load_path_pixmap(str(path))
        return None

    def _current_angle(self, knob: KnobLike) -> float:
        norm_value = knob.get_normalized_value()
        return knob.min_angle - norm_value * (knob.min_angle - knob.max_angle)

    def _draw_rotated(
        self,
        painter: QPainter,
        pixmap: QPixmap,
        center_x: float,
        center_y: float,
        size: float,
        current_angle: float,
    ) -> None:
        dest = QRectF(center_x - size / 2, center_y - size / 2, size, size).toRect()
        transform = QTransform()
        transform.translate(center_x, center_y)
        transform.rotate(self.image_zero_angle - current_angle)
        transform.translate(-center_x, -center_y)
        painter.save()
        painter.setTransform(transform, combine=True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(dest, pixmap)
        painter.restore()

    def _paint_readable_pointer(
        self,
        painter: QPainter,
        center_x: float,
        center_y: float,
        radius: float,
        current_angle: float,
    ) -> None:
        """High-contrast pointer that stays readable at small knob sizes."""
        angle_rad = math.radians(current_angle)
        start_r = radius * 0.16
        end_r = radius * 0.90
        start = QPointF(
            center_x + math.cos(angle_rad) * start_r,
            center_y - math.sin(angle_rad) * start_r,
        )
        end = QPointF(
            center_x + math.cos(angle_rad) * end_r,
            center_y - math.sin(angle_rad) * end_r,
        )
        outline = max(4.0, radius * 0.22)
        fill = max(2.0, radius * 0.11)
        painter.setPen(
            QPen(
                QColor(16, 12, 8),
                outline,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.drawLine(start, end)
        painter.setPen(
            QPen(
                QColor(255, 232, 150),
                fill,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.drawLine(start, end)

    def paint(self, painter: QPainter, knob: KnobLike) -> None:
        center_x = knob.width() / 2
        center_y = self.geometry.center_y
        size = float(self.geometry.knob_size)
        radius = size / 2
        dest = QRectF(center_x - size / 2, center_y - size / 2, size, size).toRect()
        current_angle = self._current_angle(knob)
        procedural = ProceduralKnobStyle(self.geometry)

        if self.show_value_arc:
            procedural._paint_track(painter, center_x, center_y, radius)
            procedural._paint_value_arc(painter, knob, center_x, center_y, radius)

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        body = self._pixmap(self.body_resource, self.body_path)
        if body is not None:
            if self.rotate_body:
                self._draw_rotated(
                    painter, body, center_x, center_y, size, current_angle
                )
            else:
                painter.drawPixmap(dest, body)

        pointer = self._pixmap(self.pointer_resource, self.pointer_path)
        if pointer is not None:
            self._draw_rotated(
                painter, pointer, center_x, center_y, size, current_angle
            )
        elif self.draw_pointer:
            self._paint_readable_pointer(
                painter, center_x, center_y, radius, current_angle
            )

        if self.label_painter is not None:
            self.label_painter(painter, knob)
        else:
            procedural._paint_text(painter, knob)


KnobStyle = ProceduralKnobStyle | ImageKnobStyle


def davies_knob_style(geometry: KnobGeometry | None = None) -> KnobStyle:
    """Static Davies cap with a high-contrast pointer and value arc."""
    geom = geometry or KnobGeometry()
    if not skin_available():
        return ProceduralKnobStyle(geometry=geom)
    return ImageKnobStyle(
        body_resource=KNOB_DAVIES,
        geometry=geom,
        rotate_body=False,
        show_value_arc=True,
        draw_pointer=True,
    )


def metal_knob_style(geometry: KnobGeometry | None = None) -> KnobStyle:
    """Machined metal body with a high-contrast pointer and value arc."""
    geom = geometry or KnobGeometry()
    if not skin_available():
        return ProceduralKnobStyle(geometry=geom)
    return ImageKnobStyle(
        body_resource=KNOB_METAL,
        geometry=geom,
        rotate_body=False,
        show_value_arc=True,
        draw_pointer=True,
    )


def small_knob_style() -> KnobStyle:
    return davies_knob_style(ProceduralKnobStyle.small().geometry)


def medium_knob_style() -> KnobStyle:
    return davies_knob_style(ProceduralKnobStyle.medium().geometry)


def large_knob_style() -> KnobStyle:
    return metal_knob_style(ProceduralKnobStyle.large().geometry)
