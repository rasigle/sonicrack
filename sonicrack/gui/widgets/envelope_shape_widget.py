"""Compact envelope shape preview widget with optional live position."""

from __future__ import annotations

from typing import Literal

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

EnvelopeKind = Literal["adsr", "ad"]


class EnvelopeShapeWidget(QWidget):
    """Paint an envelope curve with an optional live position marker.

    ADSR sustain uses a fixed visual hold so short envelopes still read
    clearly. Attack-decay (AD) shapes scale the peak by amount. Times are
    relative to each other on the x-axis. A glowing playhead shows the
    current envelope position when the module is active.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._kind: EnvelopeKind = "adsr"
        self._attack = 0.01
        self._decay = 0.2
        self._sustain = 0.7
        self._release = 0.3
        self._amount = 1.0
        self._points: list[tuple[float, float]] = self.shape_points(
            self._attack, self._decay, self._sustain, self._release
        )
        self._indicator_visible = False
        self._indicator_xy: tuple[float, float] | None = None

        self.setMinimumHeight(48)
        self.setMaximumHeight(64)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip("Envelope shape preview")

    def set_envelope(
        self,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
    ) -> None:
        """Update ADSR parameters and repaint when values change."""
        attack = max(0.0, float(attack))
        decay = max(0.0, float(decay))
        sustain = min(1.0, max(0.0, float(sustain)))
        release = max(0.0, float(release))

        if (
            self._kind == "adsr"
            and attack == self._attack
            and decay == self._decay
            and sustain == self._sustain
            and release == self._release
        ):
            return

        self._kind = "adsr"
        self._attack = attack
        self._decay = decay
        self._sustain = sustain
        self._release = release
        self._amount = 1.0
        self._points = self.shape_points(attack, decay, sustain, release)
        self._remap_indicator()
        self.update()

    def set_ad_envelope(
        self,
        attack: float,
        decay: float,
        amount: float = 1.0,
    ) -> None:
        """Update attack-decay parameters and repaint when values change."""
        attack = max(0.0, float(attack))
        decay = max(0.0, float(decay))
        amount = min(1.0, max(0.0, float(amount)))

        if (
            self._kind == "ad"
            and attack == self._attack
            and decay == self._decay
            and amount == self._amount
        ):
            return

        self._kind = "ad"
        self._attack = attack
        self._decay = decay
        self._sustain = 0.0
        self._release = 0.0
        self._amount = amount
        self._points = self.ad_shape_points(attack, decay, amount)
        self._remap_indicator()
        self.update()

    def envelope(self) -> tuple[float, float, float, float]:
        """Return the current (attack, decay, sustain, release) values."""
        return self._attack, self._decay, self._sustain, self._release

    def ad_envelope(self) -> tuple[float, float, float]:
        """Return the current (attack, decay, amount) values for AD mode."""
        return self._attack, self._decay, self._amount

    def kind(self) -> EnvelopeKind:
        """Return whether the widget is drawing ADSR or AD."""
        return self._kind

    def set_live_state(
        self,
        phase: str,
        progress: float,
        level: float,
        *,
        active: bool | None = None,
    ) -> None:
        """Update the playhead from envelope phase, progress, and level.

        Args:
            phase: Stage name (idle/attack/decay/sustain/release/retrigger_reset).
            progress: 0..1 progress within the current stage.
            level: Current envelope output level in [0, 1].
            active: Force visibility; defaults to hidden only for idle/ended.
        """
        if hasattr(phase, "value"):
            phase_key = str(phase.value).lower().strip()
        else:
            phase_key = str(phase or "idle").lower().strip()

        progress = min(1.0, max(0.0, float(progress)))
        level = min(1.0, max(0.0, float(level)))

        if active is None:
            active = phase_key not in {"idle", "ended", ""}

        if not active:
            if self._indicator_visible or self._indicator_xy is not None:
                self._indicator_visible = False
                self._indicator_xy = None
                self.update()
            return

        xy = self.phase_to_xy(phase_key, progress, level, self._points, self._kind)
        if xy == self._indicator_xy and self._indicator_visible:
            return

        self._indicator_visible = True
        self._indicator_xy = xy
        self.update()

    def clear_live_state(self) -> None:
        """Hide the live position indicator."""
        self.set_live_state("idle", 0.0, 0.0, active=False)

    def indicator(self) -> tuple[float, float] | None:
        """Return the current indicator (x, y) in normalized plot coords, or None."""
        if not self._indicator_visible:
            return None
        return self._indicator_xy

    def _remap_indicator(self) -> None:
        """Drop a stale playhead when the shape changes under it."""
        # Keep last coordinates if still visible; paint uses clamped values.
        if self._indicator_xy is None:
            return
        x, y = self._indicator_xy
        self._indicator_xy = (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))

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

    @staticmethod
    def ad_shape_points(
        attack: float,
        decay: float,
        amount: float = 1.0,
    ) -> list[tuple[float, float]]:
        """Return normalized (x, y) polyline points for an attack-decay shape."""
        attack = max(0.0, float(attack))
        decay = max(0.0, float(decay))
        amount = min(1.0, max(0.0, float(amount)))

        total = attack + decay
        if total <= 0.0:
            return [(0.0, 0.0), (1.0, 0.0)]

        peak_x = attack / total if attack > 0.0 else 0.0
        return [(0.0, 0.0), (peak_x, amount), (1.0, 0.0)]

    @staticmethod
    def phase_to_xy(
        phase: str,
        progress: float,
        level: float,
        points: list[tuple[float, float]],
        kind: EnvelopeKind = "adsr",
    ) -> tuple[float, float]:
        """Map envelope phase/progress/level onto normalized plot coordinates.

        Y always uses the live level so legato/punch levels match the DSP.
        X is derived from the diagram segment for the active stage.
        """
        progress = min(1.0, max(0.0, float(progress)))
        level = min(1.0, max(0.0, float(level)))
        phase = str(phase or "idle").lower().strip()

        if not points:
            return (0.0, level)

        def lerp_x(index_a: int, index_b: int, t: float) -> float:
            i0 = min(max(0, index_a), len(points) - 1)
            i1 = min(max(0, index_b), len(points) - 1)
            x0 = points[i0][0]
            x1 = points[i1][0]
            return x0 + (x1 - x0) * t

        if kind == "ad":
            # AD polyline: start -> peak -> end
            if phase == "attack":
                return (lerp_x(0, 1, progress), level)
            if phase == "decay":
                return (lerp_x(1, 2, progress), level)
            return (0.0, level)

        # ADSR polyline: start -> peak -> sustain_start -> sustain_end -> zero
        if phase in {"retrigger_reset", "idle", "ended"}:
            return (0.0, level)
        if phase == "attack":
            return (lerp_x(0, 1, progress), level)
        if phase == "decay":
            return (lerp_x(1, 2, progress), level)
        if phase == "sustain":
            # Park mid-plateau; real sustain duration is unbounded.
            return (lerp_x(2, 3, 0.5), level)
        if phase == "release":
            return (lerp_x(3, 4, progress), level)
        return (0.0, level)

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

        # Segment markers (corners)
        painter.setPen(QPen(QColor(180, 230, 190), 1.0))
        for nx, ny in points:
            x, y = to_px(nx, ny)
            painter.drawEllipse(int(x) - 2, int(y) - 2, 4, 4)

        # Live position playhead
        if self._indicator_visible and self._indicator_xy is not None:
            ix, iy = self._indicator_xy
            px, py = to_px(ix, iy)

            # Soft outer glow
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 230, 120, 70))
            painter.drawEllipse(int(px) - 8, int(py) - 8, 16, 16)

            painter.setBrush(QColor(255, 210, 80, 160))
            painter.drawEllipse(int(px) - 5, int(py) - 5, 10, 10)

            painter.setBrush(QColor(255, 250, 210))
            painter.setPen(QPen(QColor(40, 36, 20), 1.0))
            painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)
