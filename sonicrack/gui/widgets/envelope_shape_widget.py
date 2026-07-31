"""Compact envelope shape preview widget with optional live position."""

from __future__ import annotations

import math
from typing import Literal

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

EnvelopeKind = Literal["adsr", "ad"]
EnvelopeCurve = Literal["linear", "exponential", "polynomial"]

# Match soniclab.dsp.modulators.envelope_curve curvature.
_EXP_K = 5.0
_EXP_DENOM = 1.0 - math.exp(-_EXP_K)

# Samples per timed segment in the preview polyline (corners always included).
_SEGMENT_STEPS = 12


def _shape_progress(p: float, curve: EnvelopeCurve) -> float:
    """Shape linear progress p in [0, 1]; matches soniclab envelope curves."""
    p = min(1.0, max(0.0, float(p)))
    if curve == "linear":
        return p
    if curve == "exponential":
        return (1.0 - math.exp(-_EXP_K * p)) / _EXP_DENOM
    # polynomial (Hermite smoothstep)
    return p * p * (3.0 - 2.0 * p)


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
        self._curve: EnvelopeCurve = "linear"
        self._points: list[tuple[float, float]] = self.shape_points(
            self._attack, self._decay, self._sustain, self._release, self._curve
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
        curve: EnvelopeCurve | str = "linear",
    ) -> None:
        """Update ADSR parameters and repaint when values change."""
        attack = max(0.0, float(attack))
        decay = max(0.0, float(decay))
        sustain = min(1.0, max(0.0, float(sustain)))
        release = max(0.0, float(release))
        curve_key = self._normalize_curve(curve)

        if (
            self._kind == "adsr"
            and attack == self._attack
            and decay == self._decay
            and sustain == self._sustain
            and release == self._release
            and curve_key == self._curve
        ):
            return

        self._kind = "adsr"
        self._attack = attack
        self._decay = decay
        self._sustain = sustain
        self._release = release
        self._amount = 1.0
        self._curve = curve_key
        self._points = self.shape_points(attack, decay, sustain, release, curve_key)
        self._remap_indicator()
        self.update()

    def set_ad_envelope(
        self,
        attack: float,
        decay: float,
        amount: float = 1.0,
        curve: EnvelopeCurve | str = "linear",
    ) -> None:
        """Update attack-decay parameters and repaint when values change."""
        attack = max(0.0, float(attack))
        decay = max(0.0, float(decay))
        amount = min(1.0, max(0.0, float(amount)))
        curve_key = self._normalize_curve(curve)

        if (
            self._kind == "ad"
            and attack == self._attack
            and decay == self._decay
            and amount == self._amount
            and curve_key == self._curve
        ):
            return

        self._kind = "ad"
        self._attack = attack
        self._decay = decay
        self._sustain = 0.0
        self._release = 0.0
        self._amount = amount
        self._curve = curve_key
        self._points = self.ad_shape_points(attack, decay, amount, curve_key)
        self._remap_indicator()
        self.update()

    def envelope(self) -> tuple[float, float, float, float]:
        """Return the current (attack, decay, sustain, release) values."""
        return self._attack, self._decay, self._sustain, self._release

    def ad_envelope(self) -> tuple[float, float, float]:
        """Return the current (attack, decay, amount) values for AD mode."""
        return self._attack, self._decay, self._amount

    def curve(self) -> EnvelopeCurve:
        """Return the active segment curve for the preview."""
        return self._curve

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
        if self._indicator_xy is None:
            return
        x, y = self._indicator_xy
        self._indicator_xy = (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))

    @staticmethod
    def _normalize_curve(value: object) -> EnvelopeCurve:
        if not isinstance(value, str):
            return "linear"
        lowered = value.lower().strip()
        if lowered == "polynomial":
            return "polynomial"
        if lowered == "exponential":
            return "exponential"
        return "linear"

    @staticmethod
    def _segment_points(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        curve: EnvelopeCurve,
        *,
        include_start: bool,
    ) -> list[tuple[float, float]]:
        """Sample a curved segment from (x0,y0) to (x1,y1)."""
        points: list[tuple[float, float]] = []
        if include_start:
            points.append((x0, y0))
        if abs(x1 - x0) < 1e-12 and abs(y1 - y0) < 1e-12:
            if not include_start:
                points.append((x1, y1))
            return points

        steps = 1 if curve == "linear" else _SEGMENT_STEPS
        for i in range(1, steps + 1):
            t = i / steps
            s = _shape_progress(t, curve)
            points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * s))
        return points

    @staticmethod
    def shape_points(
        attack: float,
        decay: float,
        sustain: float,
        release: float,
        curve: EnvelopeCurve | str = "linear",
    ) -> list[tuple[float, float]]:
        """Return normalized (x, y) polyline points for the ADSR shape.

        x is in [0, 1] across the full envelope; y is level in [0, 1].
        """
        attack = max(0.0, float(attack))
        decay = max(0.0, float(decay))
        sustain = min(1.0, max(0.0, float(sustain)))
        release = max(0.0, float(release))
        curve_key = EnvelopeShapeWidget._normalize_curve(curve)

        active = attack + decay + release
        sustain_hold = max(0.15, 0.35 * max(active, 0.05))

        total = attack + decay + sustain_hold + release
        if total <= 0.0:
            return [(0.0, 0.0), (1.0, 0.0)]

        def nx(time: float) -> float:
            return time / total

        t_a = attack
        t_d = t_a + decay
        t_s = t_d + sustain_hold
        t_r = t_s + release

        points: list[tuple[float, float]] = []
        points.extend(
            EnvelopeShapeWidget._segment_points(
                0.0, 0.0, nx(t_a), 1.0, curve_key, include_start=True
            )
        )
        points.extend(
            EnvelopeShapeWidget._segment_points(
                nx(t_a), 1.0, nx(t_d), sustain, curve_key, include_start=False
            )
        )
        points.append((nx(t_s), sustain))
        points.extend(
            EnvelopeShapeWidget._segment_points(
                nx(t_s), sustain, nx(t_r), 0.0, curve_key, include_start=False
            )
        )
        return points

    @staticmethod
    def ad_shape_points(
        attack: float,
        decay: float,
        amount: float = 1.0,
        curve: EnvelopeCurve | str = "linear",
    ) -> list[tuple[float, float]]:
        """Return normalized (x, y) polyline points for an attack-decay shape."""
        attack = max(0.0, float(attack))
        decay = max(0.0, float(decay))
        amount = min(1.0, max(0.0, float(amount)))
        curve_key = EnvelopeShapeWidget._normalize_curve(curve)

        total = attack + decay
        if total <= 0.0:
            return [(0.0, 0.0), (1.0, 0.0)]

        peak_x = attack / total if attack > 0.0 else 0.0
        points: list[tuple[float, float]] = []
        points.extend(
            EnvelopeShapeWidget._segment_points(
                0.0, 0.0, peak_x, amount, curve_key, include_start=True
            )
        )
        points.extend(
            EnvelopeShapeWidget._segment_points(
                peak_x, amount, 1.0, 0.0, curve_key, include_start=False
            )
        )
        return points

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

        def stage_x_range(kind_key: EnvelopeKind, stage: str) -> tuple[float, float]:
            xs = [p[0] for p in points]
            if kind_key == "ad":
                peak_i = max(range(len(points)), key=lambda i: points[i][1])
                if stage == "attack":
                    return xs[0], xs[peak_i]
                if stage == "decay":
                    return xs[peak_i], xs[-1]
                return xs[0], xs[0]
            peak_i = 0
            for i, (_x, y) in enumerate(points):
                if y >= points[peak_i][1]:
                    peak_i = i
            sustain_y = None
            sus_start = peak_i
            sus_end = peak_i
            for i in range(peak_i, len(points)):
                y = points[i][1]
                if sustain_y is None and i > peak_i:
                    if abs(y - points[min(i + 1, len(points) - 1)][1]) < 1e-6:
                        sustain_y = y
                        sus_start = i
                        sus_end = i
                if sustain_y is not None and abs(y - sustain_y) < 1e-6:
                    sus_end = i
            if stage == "attack":
                return xs[0], xs[peak_i]
            if stage == "decay":
                return xs[peak_i], xs[sus_start] if sus_start > peak_i else xs[peak_i]
            if stage == "sustain":
                return xs[sus_start], xs[sus_end]
            if stage == "release":
                return xs[sus_end], xs[-1]
            return xs[0], xs[0]

        def lerp_x(x0: float, x1: float, t: float) -> float:
            return x0 + (x1 - x0) * t

        if kind == "ad":
            if phase == "attack":
                x0, x1 = stage_x_range("ad", "attack")
                return (lerp_x(x0, x1, progress), level)
            if phase == "decay":
                x0, x1 = stage_x_range("ad", "decay")
                return (lerp_x(x0, x1, progress), level)
            return (0.0, level)

        if phase in {"retrigger_reset", "idle", "ended"}:
            return (0.0, level)
        if phase == "attack":
            x0, x1 = stage_x_range("adsr", "attack")
            return (lerp_x(x0, x1, progress), level)
        if phase == "decay":
            x0, x1 = stage_x_range("adsr", "decay")
            return (lerp_x(x0, x1, progress), level)
        if phase == "sustain":
            x0, x1 = stage_x_range("adsr", "sustain")
            return (lerp_x(x0, x1, 0.5), level)
        if phase == "release":
            x0, x1 = stage_x_range("adsr", "release")
            return (lerp_x(x0, x1, progress), level)
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

        painter.fillRect(self.rect(), QColor(22, 26, 30))
        painter.setPen(QPen(QColor(55, 62, 70), 1.0))
        painter.drawRect(0, 0, width - 1, height - 1)

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

        painter.setPen(QPen(QColor(180, 230, 190), 1.0))
        marker_indices = {0, len(points) - 1}
        if self._kind == "adsr" and len(points) >= 3:
            peak_i = max(range(len(points)), key=lambda i: points[i][1])
            marker_indices.add(peak_i)
            sus_y = self._sustain
            for i, (_x, y) in enumerate(points):
                if abs(y - sus_y) < 1e-6:
                    marker_indices.add(i)
        elif self._kind == "ad":
            peak_i = max(range(len(points)), key=lambda i: points[i][1])
            marker_indices.add(peak_i)

        for i in sorted(marker_indices):
            nx, ny = points[i]
            x, y = to_px(nx, ny)
            painter.drawEllipse(int(x) - 2, int(y) - 2, 4, 4)

        if self._indicator_visible and self._indicator_xy is not None:
            ix, iy = self._indicator_xy
            px, py = to_px(ix, iy)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 230, 120, 70))
            painter.drawEllipse(int(px) - 8, int(py) - 8, 16, 16)

            painter.setBrush(QColor(255, 210, 80, 160))
            painter.drawEllipse(int(px) - 5, int(py) - 5, 10, 10)

            painter.setBrush(QColor(255, 250, 210))
            painter.setPen(QPen(QColor(40, 36, 20), 1.0))
            painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)
