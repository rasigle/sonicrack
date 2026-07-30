"""Compact ADSR envelope shape preview widget."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class EnvelopeShapeWidget(QWidget):
    """Paint a static ADSR curve from attack/decay/sustain/release values.

    The sustain segment uses a fixed visual hold so short envelopes still read
    clearly. Times are relative to each other on the x-axis.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._attack = 0.01
        self._decay = 0.2
        self._sustain = 0.7
        self._release = 0.3
        self._points: list[tuple[float, float]] = self.shape_points(
            self._attack, self._decay, self._sustain, self._release
        )

        self.setMinimumHeight(48)
        self.setMaximumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip("ADSR envelope shape preview")

    def set_envelope(
        self,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
    ) -> None:
        """Update envelope parameters and repaint when values change."""
        attack = max(0.0, float(attack))
        decay = max(0.0, float(decay))
        sustain = min(1.0, max(0.0, float(sustain)))
        release = max(0.0, float(release))

        if (
            attack == self._attack
            and decay == self._decay
            and sustain == self._sustain
            and release == self._release
        ):
            return

        self._attack = attack
        self._decay = decay
        self._sustain = sustain
        self._release = release
        self._points = self.shape_points(attack, decay, sustain, release)
        self.update()

    def envelope(self) -> tuple[float, float, float, float]:
        """Return the current (attack, decay, sustain, release) values."""
        return self._attack, self._decay, self._sustain, self._release

    @staticmethod
    def shape_points(
        attack: float,
        decay: float,
        sustain: float,
        release: float,
    ) -> list[tuple[float, float]]:
        """Return normalized (x, y) polyline points for the ADSR shape.

        x is in [0, 1] across the full envelope; y is level in [0, 1].
        Values are assumed non-negative / sustain already clamped by callers
        that store state; this method still sanitizes for direct use in tests.
        """
        attack = max(0.0, float(attack))
        decay = max(0.0, float(decay))
        sustain = min(1.0, max(0.0, float(sustain)))
        release = max(0.0, float(release))

        # Fixed visual sustain hold so the plateau is always readable.
        active = attack + decay + release
        sustain_hold = max(0.15, 0.35 * max(active, 0.05))

        total = attack + decay + sustain_hold + release
        if total <= 0.0:
            return [(0.0, 0.0), (1.0, 0.0)]

        def nx(time: float) -> float:
            return time / total

        points: list[tuple[float, float]] = [(0.0, 0.0)]
        t = attack
        points.append((nx(t), 1.0))
        t += decay
        points.append((nx(t), sustain))
        t += sustain_hold
        points.append((nx(t), sustain))
        t += release
        points.append((nx(t), 0.0))
        return points

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = max(1, self.width())
        height = max(1, self.height())
        margin_x = 4.0
        margin_y = 4.0
        plot_w = max(1.0, width - 2.0 * margin_x)
        plot_h = max(1.0, height - 2.0 * margin_y)

        # Background
        painter.fillRect(self.rect(), QColor(22, 26, 30))
        painter.setPen(QPen(QColor(55, 62, 70), 1.0))
        painter.drawRect(0, 0, width - 1, height - 1)

        # Baseline
        baseline_y = margin_y + plot_h
        painter.setPen(QPen(QColor(70, 78, 88), 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(
            int(margin_x),
            int(baseline_y),
            int(margin_x + plot_w),
            int(baseline_y),
        )

        points = self._points
        path = QPainterPath()
        fill = QPainterPath()

        def to_px(nx: float, ny: float) -> tuple[float, float]:
            x = margin_x + nx * plot_w
            y = margin_y + (1.0 - ny) * plot_h
            return x, y

        first_x, first_y = to_px(*points[0])
        path.moveTo(first_x, first_y)
        fill.moveTo(first_x, baseline_y)
        fill.lineTo(first_x, first_y)

        for nx, ny in points[1:]:
            x, y = to_px(nx, ny)
            path.lineTo(x, y)
            fill.lineTo(x, y)

        last_x, _ = to_px(*points[-1])
        fill.lineTo(last_x, baseline_y)
        fill.closeSubpath()

        painter.fillPath(fill, QColor(80, 180, 100, 55))
        painter.setPen(QPen(QColor(120, 220, 140), 2.0))
        painter.drawPath(path)

        # Segment markers (A/D/S/R corners)
        painter.setPen(QPen(QColor(180, 230, 190), 1.0))
        for nx, ny in points:
            x, y = to_px(nx, ny)
            painter.drawEllipse(int(x) - 2, int(y) - 2, 4, 4)
