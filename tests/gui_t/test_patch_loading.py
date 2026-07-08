"""Regression tests for GUI patch loading."""

from pathlib import Path

from soniclab.audio_io import AudioOutput

from src.gui.main_window import ModularSynthWindow
from src.gui.modules.output.output import OutputModule


def test_apply_preset_recreates_saved_cables(monkeypatch):
    """Loading a patch with connections should recreate its cables."""
    window = ModularSynthWindow(restore_last_patch=False)
    monkeypatch.setattr(window, "_start_output_playback", lambda: None)

    patch_data = {
        "metadata": {"name": "Connected Patch"},
        "modules": [
            {
                "id": 0,
                "type": "Oscillator",
                "position": {"x": 0, "y": 0},
                "parameters": {},
            },
            {
                "id": 1,
                "type": "Output",
                "position": {"x": 300, "y": 0},
                "parameters": {},
            },
        ],
        "connections": [
            {
                "source_module": 0,
                "source_port": "Sine",
                "target_module": 1,
                "target_port": "Left/Mono",
            }
        ],
    }

    window._apply_preset(patch_data)

    connections = window._require_patch_canvas().get_connections()
    assert len(connections) == 1
    assert connections[0][0].port_name == "Sine"
    assert connections[0][1].port_name == "Left/Mono"


def test_apply_preset_restores_module_parameters(monkeypatch):
    """Loading a patch should restore registered module parameters."""
    window = ModularSynthWindow(restore_last_patch=False)
    monkeypatch.setattr(window, "_start_output_playback", lambda: None)

    patch_data = {
        "metadata": {"name": "Parameterized Patch"},
        "modules": [
            {
                "id": 0,
                "type": "Oscillator",
                "custom_name": "Bass Source",
                "position": {"x": 0, "y": 0},
                "parameters": {
                    "frequency": 220.0,
                    "pulsewidth": 0.25,
                },
            },
        ],
        "connections": [],
    }

    window._apply_preset(patch_data)

    module = window._require_patch_canvas().get_modules()[0]
    assert module.get_custom_name() == "Bass Source"
    assert module.get_parameters()["frequency"] == 220.0
    assert module.get_parameters()["pulsewidth"] == 0.25


def test_add_output_keeps_audio_output_backend():
    """Adding Output must not replace its playback backend with AudioEngine."""
    window = ModularSynthWindow(restore_last_patch=False)

    window._add_module("Output")

    output_module = next(
        module
        for module in window._require_patch_canvas().get_modules()
        if isinstance(module, OutputModule)
    )
    assert isinstance(output_module.audio_output, AudioOutput)
    assert output_module.audio_engine is window.audio_engine


class _AcceptedCloseEvent:
    def __init__(self):
        self.accepted = False

    def accept(self):
        self.accepted = True


def test_close_autosaves_and_next_window_restores_patch(tmp_path):
    """Closing the window should restore unsaved modules and cables next launch."""
    autosave_path = tmp_path / "last_session.apr"
    patch_data = {
        "metadata": {"name": "Unsaved Patch"},
        "modules": [
            {
                "id": 0,
                "type": "Oscillator",
                "custom_name": "Unsaved Source",
                "position": {"x": 10, "y": 20},
                "parameters": {"frequency": 330.0},
            },
            {
                "id": 1,
                "type": "Filter",
                "position": {"x": 300, "y": 40},
                "parameters": {"cutoff": 1200.0},
            },
        ],
        "connections": [
            {
                "source_module": 0,
                "source_port": "Sine",
                "target_module": 1,
                "target_port": "In",
            }
        ],
    }

    window = ModularSynthWindow(
        restore_last_patch=False,
        autosave_patch_path=autosave_path,
    )
    window.current_patch_path = str(Path("unsaved_edit.apr"))
    window.patch_modified = True
    window._apply_preset(patch_data)

    close_event = _AcceptedCloseEvent()
    window.closeEvent(close_event)

    restored = ModularSynthWindow(autosave_patch_path=autosave_path)

    modules = restored._require_patch_canvas().get_modules()
    connections = restored._require_patch_canvas().get_connections()

    assert close_event.accepted
    assert len(modules) == 2
    assert len(connections) == 1
    assert {module.metadata.title for module in modules} == {"Oscillator", "Filter"}
    assert connections[0][0].port_name == "Sine"
    assert connections[0][1].port_name == "In"
    assert restored.current_patch_path == str(Path("unsaved_edit.apr"))
    assert restored.patch_modified is True


def test_window_respects_disabled_restore_last_patch_setting(tmp_path, monkeypatch):
    """Startup should skip autosave restore when the persisted setting disables it."""
    from src.gui import main_window

    autosave_path = tmp_path / "last_session.apr"
    window = ModularSynthWindow(
        restore_last_patch=False,
        autosave_patch_path=autosave_path,
    )
    window._apply_preset(
        {
            "metadata": {"name": "Do Not Restore"},
            "modules": [
                {
                    "id": 0,
                    "type": "Oscillator",
                    "position": {"x": 0, "y": 0},
                    "parameters": {},
                }
            ],
            "connections": [],
        }
    )
    window._save_last_patch()

    monkeypatch.setattr(main_window.app_settings, "restore_last_patch", False)

    restored = ModularSynthWindow(autosave_patch_path=autosave_path)

    assert restored._require_patch_canvas().get_modules() == []


def test_context_delete_uses_module_shutdown_and_engine_cleanup(monkeypatch):
    """Context-menu deletion should not leave timers or engine modules alive."""
    window = ModularSynthWindow(restore_last_patch=False)
    monkeypatch.setattr(window, "_start_output_playback", lambda: None)

    window._add_module("Waveform")
    module = window._require_patch_canvas().get_modules()[0]
    timer = module._viz_timer

    assert timer.isActive()
    assert module in window.audio_engine.modules

    module.delete_from_patch()

    assert not timer.isActive()
    assert module not in window._require_patch_canvas().get_modules()
    assert module not in window.audio_engine.modules
