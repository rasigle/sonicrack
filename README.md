# SonicRack

[![CI](https://github.com/rasigle/sonicrack/actions/workflows/ci.yml/badge.svg)](https://github.com/rasigle/sonicrack/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/sonicrack)](https://pypi.org/project/sonicrack/)
[![Python versions](https://img.shields.io/pypi/pyversions/sonicrack)](https://pypi.org/project/sonicrack/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SonicRack is a PyQt6 modular synthesizer application for building audio patches
visually. The app owns the patching UI, runtime graph, presets, and packaged
assets; DSP, realtime audio, and MIDI primitives come from [`soniclab`](https://pypi.org/project/soniclab/).

## Snapshot

- Package version: `2026.3.0`
- Python: `>=3.11`
- CLI entry point: `sonicrack`
- Module entry point: `python -m sonicrack.modular_synth_app`
- Main package: `sonicrack`
- DSP dependency: `soniclab>=2026.2.0`
- License: MIT

## Install (PyPI)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv tool install sonicrack
sonicrack
```

That installs the desktop app stack (PyQt6, device audio via `sounddevice`/`numba`,
and the `soniclab` engine). Alternatives:

```bash
# as a project dependency
uv add sonicrack

# classic pip
pip install sonicrack

# one-off run without a permanent install
uvx sonicrack
```

Module entry point (any install that puts the package on `PYTHONPATH`):

```bash
uv run python -m sonicrack.modular_synth_app
```

Optional extras:

```bash
uv tool install "sonicrack[midi]"       # MIDI ports via soniclab[midi]
uv tool install "sonicrack[examples]"   # analysis notebooks / live-input demos
uv tool install "sonicrack[full]"       # midi + examples
```

Notes:

- Realtime playback needs working system audio drivers.
- MIDI needs local ports/backends and the `midi` extra.
- `pyaudio` (examples only) may require PortAudio system packages on some OSes.

## Development install

Clone the repo and sync with `uv` (creates `.venv` and installs the project):

```powershell
uv sync --group dev
```

Optional extras for local work:

```powershell
uv sync --group dev --extra midi
uv sync --group dev --extra full
```

Windows helpers under `scripts/environment/`:

```powershell
.\scripts\environment\uv_sync_dev.bat      # dev + full extras
.\scripts\environment\uv_sync_mandatory.bat
.\scripts\environment\uv_sync_all.bat
run_modular_synth.bat                      # sync/activate + launch
```

Legacy `--extra gui` / `.[gui]` still work; those extras are empty because
GUI/audio deps are part of the base install.

Useful launch options:

```powershell
uv run sonicrack --no-splash
uv run sonicrack --log-level DEBUG --detailed-log
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
|   |-- export.py             # offline WAV bounce
|   |-- specs.py              # RuntimeModuleSpec and dispatch contracts
|   `-- helpers.py            # shared runtime block-processing helpers
|-- gui/
|   |-- main_window.py        # main Qt window and patch workflow
|   |-- dialogs/              # Qt dialogs
|   |-- modules/              # built-in patch modules
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

## Publishing a release

Version is read from `sonicrack/_version.py` (CalVer-style `YYYY.MINOR.MICRO`).
Publishing uses GitHub Actions (Trusted Publisher) on a GitHub **Release**.

### One-time setup (PyPI + GitHub)

1. Create the project on [PyPI](https://pypi.org/) (and optionally
   [TestPyPI](https://test.pypi.org/)) under the owner that will publish.
2. Add a **Trusted Publisher** for this repo:
   - Owner: `rasigle`
   - Repository: `sonicrack`
   - Workflow: `publish.yml`
   - Environment: `pypi` (and `testpypi` for TestPyPI)
3. In GitHub → **Settings → Environments**, create `pypi` and `testpypi`
   (optional protection rules / required reviewers).
4. Ensure the default branch is `main` and CI (`.github/workflows/ci.yml`) is
   green before cutting a release.

### Release steps

1. **Prep**
   - Bump `major` / `minor` / `micro` in `sonicrack/_version.py`.
   - Move the matching section in `CHANGELOG.md` from *Unreleased* to a dated
     heading (e.g. `## [2026.2.0] - 2026-07-31`).
   - Update the Snapshot version in this README if it is listed explicitly.
   - Commit on `main` and push so CI is green (`pytest`, `ruff`, `mypy`).

2. **Local sanity (optional but recommended)**

   ```powershell
   uv run pytest tests -q
   uv run ruff check sonicrack tests
   uv run mypy sonicrack tests
   uv build
   uvx twine check dist/*
   ```

3. **Tag & release** (triggers PyPI publish)

   ```powershell
   git tag 2026.2.0
   git push origin main --tags
   # Then publish a GitHub Release for that tag (UI or gh):
   # gh release create 2026.2.0 --title "2026.2.0" --notes-file CHANGELOG.md
   ```

   The `Publish to PyPI` workflow builds the sdist/wheel and uploads to PyPI
   when the release is **published**. For a dry run, use
   **Actions → Publish to PyPI → Run workflow** with target `testpypi`.

4. **Verify** — [pypi.org/project/sonicrack](https://pypi.org/project/sonicrack/)
   shows the new version; `uv tool install sonicrack` / `pip install -U sonicrack`
   installs it and `sonicrack --help` works.

Do not upload the same version twice to PyPI (versions are immutable).

## Development Notes

### Adding GUI Modules

Add a module under `sonicrack/gui/modules`, subclass the established widget and
module base classes, define metadata, and decorate the class with
`@register_module()`. Keep user-visible module widgets in `sonicrack.gui`, and
place shared patch/runtime logic in `sonicrack.patching` or `sonicrack.runtime`.

### Working With DSP

Prefer thin adapters around `soniclab` objects instead of adding a new local DSP
engine package. Shared buffer/CV/wave/spectrum helpers and oscillator frequency
slewing live in `soniclab`; keep module-specific wiring close to the module that
owns the controls.

### Documentation Status

The `docs` directory still contains historical engine-focused design notes and
completion reports. Treat older claims about removed local engine packages as
archival unless the current source tree confirms them.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

See [LICENSE](LICENSE).
