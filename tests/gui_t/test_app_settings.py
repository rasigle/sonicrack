"""Tests for persistent application settings."""

import json

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
