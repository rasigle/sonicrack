# SonicRack

SonicRack is a PyQt6 modular synthesizer application for building audio patches
visually. The app owns the patching UI, runtime graph, presets, and packaged
assets; DSP, realtime audio, and MIDI primitives come from [`soniclab`](https://pypi.org/project/soniclab/).

## Snapshot

- Package version: `2026.1.0`
- Python: `>=3.11`
- CLI entry point: `sonicrack`
- Module entry point: `python -m sonicrack.modular_synth_app`
- Main package: `sonicrack`
- DSP dependency: `soniclab`
- License: MIT

## Install (PyPI)

```bash
pip install sonicrack
```

That installs the desktop app stack (PyQt6, device audio via `sounddevice`/`numba`,
and the `soniclab` engine). Launch with:

```bash
sonicrack
# or:
python -m sonicrack.modular_synth_app
```

Optional extras:

```bash
pip install "sonicrack[midi]"       # MIDI ports via soniclab[midi]
pip install "sonicrack[examples]"   # analysis notebooks / live-input demos
pip install "sonicrack[full]"       # midi + examples
```

Notes:

- Realtime playback needs working system audio drivers.
- MIDI needs local ports/backends and the `midi` extra.
- `pyaudio` (examples only) may require PortAudio system packages on some OSes.

## Development install

Recommended environment tool: `uv`.

```powershell
uv sync --group dev
# or editable with pip:
python -m pip install -e ".[dev]"
```

Legacy install scripts that used `--extra gui` or `.[gui]` still work; those
extras are empty because GUI/audio deps are part of the base install.

Windows helper:

```powershell
run_modular_synth.bat
```

Useful launch options:

```powershell
sonicrack --no-splash
sonicrack --log-level DEBUG --detailed-log
```

## Current Scope

This repository contains:

- a desktop modular patching application under `sonicrack.gui`
- app settings and audio configuration under `sonicrack.config`
- patch-domain contracts, ports, presets, and module registration under
  `sonicrack.patching`
- graph rendering and runtime dispatch under `sonicrack.runtime`
- built-in patch modules under `sonicrack.gui.modules`
- packaged icons and splash screens under `sonicrack.resources`
- examples, patch files, and GUI/module tests

The old local engine packages are no longer part of this repository. Code that
needs oscillators, filters, effects, sequencers, audio output, or MIDI helpers
imports them from `soniclab`.

## Package Structure

```text
sonicrack/
|-- modular_synth_app.py      # PyQt application entry point
|-- constants.py              # package resources and app constants
|-- config/
|   |-- app_settings.py       # persistent JSON app preferences
|   `-- audio_config.py       # runtime sample-rate and buffer settings
|-- patching/
|   |-- module.py             # module metadata and AudioModule contract
|   |-- port.py               # signal ports and connection rules
|   |-- preset_manager.py     # preset persistence
|   `-- registry.py           # module discovery and registration
|-- runtime/
|   |-- engine.py             # graph renderer and render-plan cache
|   |-- specs.py              # RuntimeModuleSpec and dispatch contracts
|   `-- helpers.py            # shared runtime block-processing helpers
|-- gui/
|   |-- main_window.py        # main Qt window and patch workflow
|   |-- ui_constants.py       # Qt/UI constants
|   |-- dialogs/              # Qt dialogs
|   |-- modules/              # built-in patch modules
|   |-- utils/                # GUI-specific utilities
|   `-- widgets/              # reusable Qt widgets
|-- resources/                # packaged runtime assets
`-- utils/                    # logging, diagnostics, and audio file helpers
```

Use these canonical imports for shared app, patching, and runtime code:

```python
from sonicrack.config.audio_config import audio_config
from sonicrack.patching.module import ModuleCategory, ModuleMetadata
from sonicrack.patching.registry import register_module
from sonicrack.runtime.specs import RuntimeParameters
from sonicrack.runtime.helpers import read_samples, silence
```

## Runtime Model

`sonicrack.runtime.engine.AudioEngine` owns graph rendering. Sink modules such as
Output, Waveform, and Spectrum request buffers from the engine. The engine
compiles a render plan, processes upstream modules once per render cycle, and
caches port values through `RenderContext`.

Patch modules register themselves with `@register_module()` and are discovered
recursively from `sonicrack.gui.modules`. Module runtime behavior is declared
with `RuntimeModuleSpec` and implemented by the module widget that owns the
controls.

## Testing

Run the current test suite with:

```powershell
uv run pytest tests -q
```

Focused runs:

```powershell
uv run pytest tests/gui_t -q
uv run pytest tests/utils_t -q
```

For headless GUI testing:

```powershell
$env:QT_QPA_PLATFORM="offscreen"
uv run pytest tests/gui_t -q
```

## Publishing

Release builds use Hatchling (`uv build`). CI can publish via Trusted Publishing
on GitHub release (PyPI) or `workflow_dispatch` (TestPyPI). See
`.github/workflows/publish.yml`.

## Development Notes

### Adding GUI Modules

Add a module under `sonicrack/gui/modules`, subclass the established widget and
module base classes, define metadata, and decorate the class with
`@register_module()`. Keep user-visible module widgets in `sonicrack.gui`, and
place shared patch/runtime logic in `sonicrack.patching` or `sonicrack.runtime`.

### Working With DSP

Prefer thin adapters around `soniclab` objects instead of adding a new local DSP
engine package. Keep module-specific runtime code close to the module that owns
the controls.

### Documentation Status

The `docs` directory still contains historical engine-focused design notes and
completion reports. Treat older claims about removed local engine packages as
archival unless the current source tree confirms them.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

See [LICENSE](LICENSE).
