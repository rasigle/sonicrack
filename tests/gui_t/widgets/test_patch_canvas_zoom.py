"""Tests for patch canvas zoom and pan controls."""

from __future__ import annotations

from typing import Any

import pytest
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent

from sonicrack.gui.widgets.patch_canvas import PatchCanvas


def test_zoom_in_out_and_reset(qapp: Any):
    del qapp
    canvas = PatchCanvas()
    assert canvas.zoom_factor == pytest.approx(1.0)

    canvas.zoom_in()
    assert canvas.zoom_factor == pytest.approx(PatchCanvas.ZOOM_STEP)

    canvas.zoom_out()
    assert canvas.zoom_factor == pytest.approx(1.0)

    canvas.zoom_in()
    canvas.zoom_in()
    canvas.reset_zoom()
    assert canvas.zoom_factor == pytest.approx(1.0)


def test_zoom_clamped_to_min_and_max(qapp: Any):
    del qapp
    canvas = PatchCanvas()

    canvas.set_zoom(100.0)
    assert canvas.zoom_factor == pytest.approx(PatchCanvas.MAX_ZOOM)

    canvas.set_zoom(0.01)
    assert canvas.zoom_factor == pytest.approx(PatchCanvas.MIN_ZOOM)


def test_scroll_wheel_zooms_canvas(qapp: Any):
    del qapp
    canvas = PatchCanvas()
    canvas.resize(400, 300)
    canvas.show()

    # Positive wheel delta zooms in (no modifier required).
    event = QWheelEvent(
        QPointF(100, 100),
        QPointF(100, 100),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    canvas.wheelEvent(event)
    assert canvas.zoom_factor > 1.0


def test_middle_mouse_enters_and_exits_pan_mode(qapp: Any):
    del qapp
    canvas = PatchCanvas()
    canvas.resize(400, 300)
    canvas.show()

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(100, 100),
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(press)
    assert canvas._panning is True

    release = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(120, 110),
        Qt.MouseButton.MiddleButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseReleaseEvent(release)
    assert canvas._panning is False
    assert canvas._pan_start is None
