"""Tests for the packaged Eurorack hardware skin."""

from __future__ import annotations

from typing import Any

from PyQt6.QtGui import QColor, QPainter, QPixmap

from sonicrack.gui.widgets.module_widget import ModuleWidget
from sonicrack.gui.widgets.skin import (
    JACK,
    KNOB_DAVIES,
    LED_ON,
    PANEL_ALUMINUM,
    SCREW,
    load_skin_pixmap,
    skin_available,
    tinted_pixmap,
)
from sonicrack.patching.module import ModuleCategory, ModuleMetadata


class _SkinModule(ModuleWidget):
    metadata = ModuleMetadata("Skin Test", ModuleCategory.SOURCE)

    def __init__(self) -> None:
        super().__init__(width=180, height=160, color=QColor(80, 120, 200))


def test_eurorack_assets_load(qapp: Any):
    del qapp
    assert skin_available()
    for resource in (KNOB_DAVIES, PANEL_ALUMINUM, JACK, SCREW, LED_ON):
        pixmap = load_skin_pixmap(*resource)
        assert pixmap is not None
        assert not pixmap.isNull()


def test_tinted_jack_cache(qapp: Any):
    del qapp
    first = tinted_pixmap(JACK, 255, 159, 67, 80)
    second = tinted_pixmap(JACK, 255, 159, 67, 80)
    assert first is not None
    assert first is second


def test_module_faceplate_paints(qapp: Any):
    del qapp
    module = _SkinModule()
    bounds = module.boundingRect()
    pixmap = QPixmap(int(bounds.width()) + 4, int(bounds.height()) + 4)
    pixmap.fill(0)
    painter = QPainter(pixmap)
    module.paint(painter, None, None)
    painter.end()
    assert not pixmap.isNull()
    assert len(module._screw_rects()) >= 3
