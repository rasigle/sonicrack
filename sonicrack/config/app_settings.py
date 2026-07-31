"""Persistent application-level settings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sonicrack.constants import DEFAULT_SETTINGS_FILE

logger = logging.getLogger(__name__)


class AppSettings:
    """Small JSON-backed store for application preferences."""

    def __init__(self, settings_path: str | Path | None = None):
        self.settings_path = Path(settings_path or DEFAULT_SETTINGS_FILE)
        self.restore_last_patch = True
        self.last_file_directory: str | None = None
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
        self.last_file_directory = _optional_directory_setting(
            data.get("last_file_directory")
        )

    def save(self) -> bool:
        """Persist settings to disk."""
        data: dict[str, Any] = {
            "restore_last_patch": self.restore_last_patch,
        }
        if self.last_file_directory:
            data["last_file_directory"] = self.last_file_directory

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

    def file_dialog_start_dir(self, fallback: str | Path | None = None) -> str:
        """Return the directory to open file dialogs in.

        Prefers the last remembered directory when it still exists, then an
        optional fallback path (file or directory), then the user's Documents.
        """
        if self.last_file_directory:
            remembered = Path(self.last_file_directory)
            if remembered.is_dir():
                return str(remembered)

        if fallback is not None:
            candidate = Path(fallback)
            if candidate.is_dir():
                return str(candidate)
            parent = candidate.parent
            if parent.is_dir():
                return str(parent)

        return str(Path.home() / "Documents")

    def remember_file_directory(self, file_path: str | Path) -> None:
        """Remember the parent directory of a chosen file and persist it."""
        try:
            directory = str(Path(file_path).expanduser().resolve().parent)
        except OSError:
            directory = str(Path(file_path).expanduser().parent)

        if not directory or directory == self.last_file_directory:
            return

        self.last_file_directory = directory
        self.save()


def _bool_setting(value: object, *, default: bool) -> bool:
    """Return a bool only when the stored value is explicitly boolean."""
    if isinstance(value, bool):
        return value
    return default


def _optional_directory_setting(value: object) -> str | None:
    """Return a non-empty directory path string, or None."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


app_settings = AppSettings()
