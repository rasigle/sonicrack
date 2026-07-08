"""Image-capable push button widgets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QToolButton, QWidget

from sonicrack.constants import resource_path

ImageResource = str | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ImageButtonStyle:
    """Images and dimensions for an image-backed button."""

    off_path: str | Path | None = None
    on_path: str | Path | None = None
    pressed_path: str | Path | None = None
    off_resource: ImageResource | None = None
    on_resource: ImageResource | None = None
    pressed_resource: ImageResource | None = None
    size: int = 24
    icon_size: int | None = None


class ImagePushButton(QToolButton):
    """A compact push button that can render from state-specific images."""

    def __init__(
        self,
        text: str = "",
        *,
        style: ImageButtonStyle | None = None,
        checkable: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._style = style or ImageButtonStyle()
        self._off_icon = self._load_icon(
            self._style.off_path,
            self._style.off_resource,
        )
        self._on_icon = self._load_icon(
            self._style.on_path,
            self._style.on_resource,
        )
        self._pressed_icon = self._load_icon(
            self._style.pressed_path,
            self._style.pressed_resource,
        )

        self.setText(text)
        self.setCheckable(checkable)
        self.setFixedSize(self._style.size, self._style.size)
        icon_size = self._style.icon_size or self._style.size
        self.setIconSize(QSize(icon_size, icon_size))
        self.toggled.connect(lambda _checked: self._refresh_icon())
        self.pressed.connect(self._show_pressed_icon)
        self.released.connect(self._refresh_icon)
        self._refresh_icon()

    @classmethod
    def _load_icon(
        cls,
        path: str | Path | None,
        resource_name: ImageResource | None,
    ) -> QIcon | None:
        pixmap = cls._load_pixmap(path, resource_name)
        if pixmap is None:
            return None
        return QIcon(pixmap)

    @staticmethod
    def _load_pixmap(
        path: str | Path | None,
        resource_name: ImageResource | None,
    ) -> QPixmap | None:
        if path is None and resource_name is None:
            return None

        if resource_name is not None:
            resource_parts = (
                (resource_name,)
                if isinstance(resource_name, str)
                else tuple(resource_name)
            )
            with resource_path(*resource_parts) as resolved_path:
                pixmap = QPixmap(str(resolved_path))
        else:
            pixmap = QPixmap(str(path))

        if pixmap.isNull():
            return None
        return pixmap

    def _show_pressed_icon(self) -> None:
        if self._pressed_icon is not None:
            self.setIcon(self._pressed_icon)

    def _refresh_icon(self) -> None:
        icon = self._on_icon if self.isChecked() else self._off_icon
        if icon is not None:
            self.setIcon(icon)
        elif self._off_icon is None and self._on_icon is None:
            self.setIcon(QIcon())
