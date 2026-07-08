# SonicRack

SonicRack is a PyQt6 modular synthesizer application for building audio patches
visually. The app owns the patching UI, runtime graph, presets, and packaged
assets; DSP, realtime audio, and MIDI primitives come from `soniclab`.

## Snapshot

- Package version: `0.1.0`
- Python: `>=3.11`
- Primary entry point: `python -m sonicrack.modular_synth_app`
- Recommended environment tool: `uv`
- Main package: `sonicrack`
- DSP dependency: `soniclab`
- License: MIT

## Current Scope

This repository contains:

- a desktop modular patching application under `sonicrack.gui`
- app settings and audio configuration under `sonicrack.config`
- patch-domain contracts, ports, presets, and module registration under
  `sonicrack.patching`
- graph rendering and runtime dispatch under `sonicrack.runtime`
- built-in patch modules under `sonicrack.gui.modules`
- packaged icons, splash screens, and screenshots under `sonicrack.resources`
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

## Quick Start

Install GUI dependencies:

```powershell
uv sync --extra gui
```

Editable install alternative:

```powershell
python -m pip install -e ".[gui]"
```

Optional extras:

```powershell
python -m pip install -e ".[audio-io]"
python -m pip install -e ".[midi]"
python -m pip install -e ".[examples]"
python -m pip install -e ".[full]"
```

Launch the GUI:

```powershell
python -m sonicrack.modular_synth_app
```

Windows helper:

```powershell
run_modular_synth.bat
```

Useful launch options:

```powershell
python -m sonicrack.modular_synth_app --no-splash
python -m sonicrack.modular_synth_app --log-level DEBUG --detailed-log
```

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

## License

See `LICENSE`.
