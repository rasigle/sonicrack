"""Packaged Eurorack hardware skin (pixmaps + tint helpers)."""

from __future__ import annotations

from functools import lru_cache

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPixmap

from sonicrack.constants import resource_path

SKIN = ("skins", "eurorack")
PANEL_ALUMINUM = (*SKIN, "panel_aluminum.png")
KNOB_DAVIES = (*SKIN, "knob_davies.png")
KNOB_METAL = (*SKIN, "knob_metal.png")
KNOB_METAL_POINTER = (*SKIN, "knob_metal_pointer.png")
JACK = (*SKIN, "jack.png")
SCREW = (*SKIN, "screw.png")
LED_OFF = (*SKIN, "led_off.png")
LED_ON = (*SKIN, "led_on.png")


@lru_cache(maxsize=32)
def load_skin_pixmap(*parts: str) -> QPixmap | None:
    """Load a packaged skin pixmap, or ``None`` if it is missing."""
    try:
        with resource_path(*parts) as path:
            pixmap = QPixmap(str(path))
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return None
    if pixmap.isNull():
        return None
    return pixmap


@lru_cache(maxsize=32)
def load_path_pixmap(path: str) -> QPixmap | None:
    """Load a filesystem pixmap (tests / custom skins)."""
    pixmap = QPixmap(path)
    if pixmap.isNull():
        return None
    return pixmap


def skin_available() -> bool:
    """Return True when the default Eurorack kit is packaged and loadable."""
    return (
        load_skin_pixmap(*KNOB_DAVIES) is not None
        and load_skin_pixmap(*PANEL_ALUMINUM) is not None
    )


@lru_cache(maxsize=96)
def tinted_pixmap(
    resource: tuple[str, ...],
    red: int,
    green: int,
    blue: int,
    alpha: int,
) -> QPixmap | None:
    """Return ``resource`` with a SourceAtop color wash (for jacks / LEDs)."""
    src = load_skin_pixmap(*resource)
    if src is None:
        return None
    out = QPixmap(src.size())
    out.fill(Qt.GlobalColor.transparent)
    painter = QPainter(out)
    painter.drawPixmap(0, 0, src)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceAtop)
    painter.fillRect(out.rect(), QColor(red, green, blue, alpha))
    painter.end()
    return out
