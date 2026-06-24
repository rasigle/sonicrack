"""Regression tests for GUI patch loading."""

from src.audio_io import AudioOutput
from src.gui.main_window import ModularSynthWindow
from src.gui.modules.output.output import OutputModule


def test_apply_preset_recreates_saved_cables(monkeypatch):
    """Loading a patch with connections should recreate its cables."""
    window = ModularSynthWindow()
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


def test_add_output_keeps_audio_output_backend():
    """Adding Output must not replace its playback backend with AudioEngine."""
    window = ModularSynthWindow()

    window._add_module("Output")

    output_module = next(
        module
        for module in window._require_patch_canvas().get_modules()
        if isinstance(module, OutputModule)
    )
    assert isinstance(output_module.audio_output, AudioOutput)
    assert output_module.audio_engine is window.audio_engine
