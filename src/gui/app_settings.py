"""Persistent application-level settings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.gui.ui_constants import DEFAULT_SETTINGS_FILE

logger = logging.getLogger(__name__)


class AppSettings:
    """Small JSON-backed store for application preferences."""

    def __init__(self, settings_path: str | Path | None = None):
        self.settings_path = Path(settings_path or DEFAULT_SETTINGS_FILE)
        self.restore_last_patch = True
        self.load()

    def load(self) -> None:
        """Load settings from disk, keeping defaults for missing values."""
        if not self.settings_path.is_file():
            return

        try:
            with open(self.settings_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to load app settings from %s: %s",
                self.settings_path,
                exc,
                exc_info=True,
            )
            return

        if not isinstance(data, dict):
            logger.warning("Ignoring malformed app settings at %s", self.settings_path)
            return

        self.restore_last_patch = _bool_setting(
            data.get("restore_last_patch"),
            default=self.restore_last_patch,
        )

    def save(self) -> bool:
        """Persist settings to disk."""
        data: dict[str, Any] = {
            "restore_last_patch": self.restore_last_patch,
        }

        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to save app settings to %s: %s",
                self.settings_path,
                exc,
                exc_info=True,
            )
            return False


def _bool_setting(value: object, *, default: bool) -> bool:
    """Return a bool only when the stored value is explicitly boolean."""
    if isinstance(value, bool):
        return value
    return default


app_settings = AppSettings()
