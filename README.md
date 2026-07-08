# AudioPlayground

AudioPlayground is a PyQt6 modular synthesizer application for building audio
patches visually. 

## Snapshot

- Package version: `0.1.0`
- Python: `>=3.11`
- Primary entry point: `python -m src.modular_synth_app`
- Recommended environment tool: `uv`
- Main application package today: `src`
- DSP dependency: `soniclab`
- License: MIT

## Current Scope

This repository currently contains:

- a desktop modular patching application under `src/gui`
- a graph render coordinator in `src/gui/audio_engine.py`
- GUI module contracts, ports, runtime dispatch, and preset loading under
  `src/gui/core`
- built-in patch modules under `src/gui/modules`
- Qt widgets, dialogs, resources, logging, crash diagnostics, and audio helpers
- example patch files and GUI/module examples under `examples`
- GUI and utility tests under `tests`

The removed packages `src/engine`, `src/audio_io`, and `src/midi_io` are no
longer part of this repository. Code that needs oscillators, filters, effects,
sequencers, audio output, or MIDI helpers imports them from `soniclab`.

## What Is Implemented

### Application Shell

`src/modular_synth_app.py` starts the PyQt6 application, configures logging,
shows the splash screen when available, creates the main window, and activates
crash diagnostics.

### Modular GUI

`src/gui/main_window.py` owns the main window, patch canvas, module library,
menu actions, patch save/load flow, autosave restore behavior, and preset
integration.

### Runtime Graph

`src/gui/audio_engine.py` owns graph rendering for the GUI. Sink modules such as
Output, Waveform, and Spectrum request buffers from the engine. The engine
compiles a render plan, processes upstream modules once per render cycle, and
caches port values through `RenderContext`.

### Module System

GUI modules register themselves with `@register_module()` and are discovered
recursively from `src/gui/modules`. Static analysis currently finds 32
registered module classes across these categories:

- effects
- input
- mixer
- modifier
- modulated source
- output
- sequencing
- source
- visualization
- voice

Module runtime behavior is declared with `RuntimeModuleSpec` and implemented by
the module widgets themselves. Low-level DSP work is delegated to `soniclab`
objects where appropriate.

## Quick Start

### Install Dependencies

Recommended:

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

Notes:

- `gui` installs PyQt6 and the audio-output extra.
- `audio-io` installs realtime audio dependencies.
- `midi` installs MIDI dependencies.
- `examples` installs notebook, plotting, and analysis dependencies.
- Realtime audio and MIDI also depend on working local system drivers.

### Launch The GUI

Windows helper:

```powershell
run_modular_synth.bat
```

Direct launch:

```powershell
python -m src.modular_synth_app
```

Useful launch options:

```powershell
python -m src.modular_synth_app --no-splash
python -m src.modular_synth_app --log-level DEBUG --detailed-log
```

## Current Repository Layout

```text
AudioPlayground/
|-- src/
|   |-- modular_synth_app.py      # PyQt application entry point
|   |-- constants.py              # app constants and packaged resource helpers
|   |-- gui/
|   |   |-- app_settings.py       # persistent JSON app preferences
|   |   |-- audio_config.py       # sample-rate and buffer-size settings
|   |   |-- audio_engine.py       # GUI graph renderer and render-plan cache
|   |   |-- main_window.py        # main Qt window and patch workflow
|   |   |-- module_registry.py    # module discovery and registration
|   |   |-- core/                 # module, port, runtime, preset contracts
|   |   |-- dialogs/              # Qt dialogs
|   |   |-- modules/              # built-in modular synth modules
|   |   |-- utils/                # GUI-specific utilities
|   |   `-- widgets/              # reusable Qt widgets
|   |-- resources/                # packaged icons, splash, screenshots
|   `-- utils/                    # logging, diagnostics, audio file helpers
|-- tests/
|   |-- gui_t/
|   `-- utils_t/
|-- examples/
|   |-- module_components/
|   |-- patches/
|   `-- signal_analysis/
|-- docs/
|-- pyproject.toml
`-- run_modular_synth.bat
```

## Proposed Package Structure

The current package name `src` works, but it makes imports and distribution less
clear. A cleaner long-term structure is to use a real package name and separate
application, domain, infrastructure, and assets:

```text
src/
`-- audioplayground/
    |-- __init__.py
    |-- __main__.py                  # optional: python -m audioplayground
    |-- app/
    |   |-- cli.py                   # argument parsing and process startup
    |   |-- qt_app.py                # QApplication, splash, shutdown wiring
    |   `-- logging.py               # app logging setup
    |-- config/
    |   |-- app_settings.py          # persistent user preferences
    |   |-- audio_config.py          # runtime audio configuration
    |   `-- constants.py             # app-level constants
    |-- runtime/
    |   |-- graph.py                 # render graph and topological planning
    |   |-- render_context.py        # per-buffer cache and port reads
    |   |-- engine.py                # GUI render coordinator
    |   `-- specs.py                 # RuntimeModuleSpec and dispatch helpers
    |-- patching/
    |   |-- ports.py                 # Port, PortType, PortSignal
    |   |-- modules.py               # AudioModule and metadata contracts
    |   |-- registry.py              # module discovery and registration
    |   |-- presets.py               # preset persistence
    |   `-- patch_io.py              # patch save/load serialization
    |-- modules/
    |   |-- sources/
    |   |-- processors/
    |   |-- effects/
    |   |-- sequencing/
    |   |-- input/
    |   |-- output/
    |   |-- visualization/
    |   `-- voices/
    |-- ui/
    |   |-- main_window.py
    |   |-- dialogs/
    |   |-- widgets/
    |   `-- styles/
    |-- integrations/
    |   |-- soniclab.py              # thin adapters around soniclab imports
    |   |-- audio_output.py          # realtime output boundary
    |   `-- midi.py                  # MIDI boundary
    |-- resources/
    `-- utilities/
        |-- diagnostics.py
        |-- audio_files.py
        `-- paths.py
```

### Why This Is Cleaner

- `audioplayground` is a real import package; `src` becomes only the packaging
  layout directory.
- `runtime` contains graph execution, not Qt widgets.
- `patching` contains patch-domain concepts such as ports, module metadata,
  registry, preset management, and serialization.
- `modules` contains user-visible patch modules only.
- `ui` contains Qt presentation code only.
- `integrations` isolates third-party boundaries such as `soniclab`, realtime
  audio output, and MIDI.
- `config` separates persistent settings from UI and runtime logic.

### Suggested Migration Order

1. Create `src/audioplayground` and move import-neutral utilities first.
2. Move resources and constants, then update resource loading to use
   `audioplayground.resources`.
3. Move `src/gui/core` into `patching` and `runtime` based on responsibility.
4. Move Qt widgets, dialogs, and `main_window.py` into `ui`.
5. Move `src/gui/modules` into `modules` without changing module behavior.
6. Add compatibility imports from old `src.*` paths for one release if external
   callers depend on them.
7. Rename tests from `gui_t` and `utils_t` to match the new package boundaries.

## Testing

Run the current test suite with:

```powershell
python -m pytest tests -q
```

Focused runs:

```powershell
python -m pytest tests/gui_t -q
python -m pytest tests/utils_t -q
```

For headless GUI testing on systems without a display server:

```powershell
$env:QT_QPA_PLATFORM="offscreen"
python -m pytest tests/gui_t -q
```

## Development Notes

### Adding GUI Modules

Add a module under `src/gui/modules`, subclass the established widget/module
base classes, define metadata, and decorate the class with `@register_module()`.
The registry imports module files recursively during application startup.

### Working With DSP

Do not add a new local engine package unless there is a clear reason to own DSP
code in this repository again. Prefer a thin GUI adapter around `soniclab`
objects, and keep module-specific runtime code close to the module that owns the
controls.

### Documentation Status

The `docs` directory still contains historical engine-focused design notes and
completion reports. Treat older claims about `src/engine`, `src/audio_io`, or
`src/midi_io` as archival unless the current source tree confirms them.

## License

See `LICENSE`.
