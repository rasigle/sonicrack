"""Tests for persistent application settings."""

import json
from pathlib import Path

from sonicrack.config.app_settings import AppSettings


class _FakeAppSettings:
    def __init__(self, restore_last_patch: bool):
        self.restore_last_patch = restore_last_patch
        self.saved = False

    def save(self) -> bool:
        self.saved = True
        return True


def test_app_settings_defaults_to_restoring_last_patch(tmp_path):
    """Missing settings should preserve the current startup behavior."""
    settings = AppSettings(tmp_path / "settings.json")

    assert settings.restore_last_patch is True
    assert settings.last_file_directory is None


def test_app_settings_persists_restore_last_patch(tmp_path):
    """The restore-last-patch preference should round-trip through JSON."""
    settings_path = tmp_path / "settings.json"

    settings = AppSettings(settings_path)
    settings.restore_last_patch = False
    assert settings.save()

    reloaded = AppSettings(settings_path)

    assert reloaded.restore_last_patch is False
    assert json.loads(settings_path.read_text(encoding="utf-8")) == {
        "restore_last_patch": False
    }


def test_app_settings_persists_last_file_directory(tmp_path):
    """Last open/save directory should round-trip through JSON."""
    settings_path = tmp_path / "settings.json"
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    patch_file = patch_dir / "demo.apr"
    patch_file.write_text("{}", encoding="utf-8")

    settings = AppSettings(settings_path)
    settings.remember_file_directory(patch_file)

    reloaded = AppSettings(settings_path)

    assert reloaded.last_file_directory == str(patch_dir.resolve())
    assert reloaded.file_dialog_start_dir() == str(patch_dir.resolve())
    assert json.loads(settings_path.read_text(encoding="utf-8"))[
        "last_file_directory"
    ] == str(patch_dir.resolve())


def test_file_dialog_start_dir_falls_back_when_remembered_missing(tmp_path):
    """Missing remembered directories should fall back cleanly."""
    settings = AppSettings(tmp_path / "settings.json")
    settings.last_file_directory = str(tmp_path / "gone")
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    fallback_file = fallback_dir / "patch.apr"

    assert settings.file_dialog_start_dir(fallback_file) == str(fallback_dir)
    assert settings.file_dialog_start_dir() == str(Path.home() / "Documents")


def test_settings_dialog_edits_restore_last_patch(monkeypatch, qapp):
    """The application settings dialog should expose the restore preference."""
    from sonicrack.gui.dialogs import audio_settings_dialog
    from sonicrack.gui.dialogs.audio_settings_dialog import AudioSettingsDialog

    fake_settings = _FakeAppSettings(restore_last_patch=True)
    monkeypatch.setattr(audio_settings_dialog, "app_settings", fake_settings)

    dialog = AudioSettingsDialog()

    assert dialog.restore_last_patch_checkbox.isChecked()

    dialog.restore_last_patch_checkbox.setChecked(False)
    dialog._apply_and_close()

    assert fake_settings.restore_last_patch is False
    assert fake_settings.saved is True
