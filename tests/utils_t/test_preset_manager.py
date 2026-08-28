"""Preset listing and path-extension coverage."""

from __future__ import annotations

import json

from sonicrack.constants import PRESET_FILE_EXTENSION
from sonicrack.patching.preset_manager import PresetManager


def _write_preset(path, name: str, category: str = "User") -> None:
    path.write_text(
        json.dumps(
            {
                "metadata": {"name": name, "category": category},
                "modules": [],
                "connections": [],
            }
        ),
        encoding="utf-8",
    )


def test_list_presets_finds_apr_and_legacy_json(tmp_path) -> None:
    manager = PresetManager(tmp_path)
    _write_preset(tmp_path / f"demo{PRESET_FILE_EXTENSION}", "Apr Demo")
    _write_preset(tmp_path / "legacy.json", "Json Demo")

    names = {preset.get("name") for preset in manager.list_presets()}
    assert "Apr Demo" in names
    assert "Json Demo" in names


def test_list_presets_filters_by_category(tmp_path) -> None:
    manager = PresetManager(tmp_path)
    _write_preset(tmp_path / f"bass{PRESET_FILE_EXTENSION}", "Bass", "Bass")
    _write_preset(tmp_path / f"lead{PRESET_FILE_EXTENSION}", "Lead", "Lead")

    names = {preset.get("name") for preset in manager.list_presets("Bass")}
    assert names == {"Bass"}
