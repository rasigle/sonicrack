# AudioPlayground

AudioPlayground is a Python audio-synthesis workspace that combines a vectorized DSP engine with a PyQt6 modular-synth GUI.

This repository currently contains:

- a reusable engine API under `src/engine`
- a desktop modular patching application under `src/gui` and `src/modular_synth_app.py`
- examples, notebooks, and experiment folders under `examples/`
- a large set of implementation notes and design docs under `docs/`
- an active test suite under `tests/`

## Snapshot

- **Package version:** `0.1.0`
- **Python:** `>=3.11`
- **Primary GUI entry point:** `python -m src.modular_synth_app`
- **Recommended environment tool:** `uv`
- **Main engine import surface:** `from src.engine import ...`
- **License:** MIT

## What is implemented

### Engine (`src/engine`)

The engine exposes a fairly broad synthesis surface:

- **Oscillators:** sine, square, sawtooth, triangle
- **Modulation:** ADSR and modulated oscillators / frequency modulation helpers
- **Modifiers:** volume, panning, clipping, frequency-related processors
- **Composition:** `Chain`, `WaveAdder`
- **Noise:** white, pink, brownian, blue, grey, velvet, sample-and-hold, Perlin
- **Filters:** Butterworth filter utilities and filter component support
- **Effects:** distortion, delay, reverb
- **I/O:** `AudioOutput` using `sounddevice`
- **Preset/build helpers:** preset builder and preset library modules

Example:

```python
from src.engine import SineOscillator

osc = SineOscillator(frequency=440, amplitude=0.3)
samples = osc.get_samples(44100)
```

### GUI (`src/gui`)

The GUI is a modular patching application built on PyQt6. At the time of verification, the dynamic module registry discovers **21 GUI modules**:

- `ADSR Envelope`
- `Clipper`
- `Clipper (Mod)`
- `Delay`
- `Distortion`
- `Filter`
- `LFO`
- `MIDI Input`
- `Mixer`
- `Noise`
- `Oscillator`
- `Output`
- `Panner`
- `Panner (Mod)`
- `Reverb`
- `Spectrum`
- `VCA`
- `VCO`
- `Volume`
- `Volume (Mod)`
- `Waveform`

The GUI module system is registry-driven via `src/gui/core/module_registry.py` and recursively discovers modules from `src/gui/modules/`.

## Quick start

### 1. Install dependencies

Recommended:

```powershell
uv sync
```

Alternative editable install:

```powershell
python -m pip install -e .
```

Optional extras:

```powershell
python -m pip install -e ".[midi]"
python -m pip install -e ".[examples]"
python -m pip install -e ".[full]"
```

Notes:

- The base install covers the core engine and PyQt6 GUI application.
- MIDI support now lives behind the `midi` extra.
- Notebook and visualization-heavy example dependencies live behind `examples` extras.
- Depending on your machine, real audio or MIDI features may also require working local system drivers.

### 2. Launch the GUI

Windows helper:

```powershell
run_modular_synth.bat
```

Direct module launch:

```powershell
python -m src.modular_synth_app
```

### 3. Try the engine directly

```powershell
python -c "from src.engine import SineOscillator; import numpy as np; osc=SineOscillator(frequency=440, amplitude=0.2); samples=osc.get_samples(16); print(samples.shape, samples.dtype, np.round(samples[:5], 4))"
```

## Testing

The repository currently has a large automated test suite.

### Verified on 2026-05-26

These test runs were executed successfully during this update:

- `python -m pytest tests/engine/io/test_audio_output.py -q` → `39 passed`
- `python -m pytest tests/engine/io -q` → `177 passed`
- `python -m pytest tests/engine -q` → `679 passed, 8 subtests passed`

In addition, test collection reports:

- `python -m pytest tests --collect-only -q` → `942 tests collected`

Suggested commands:

```powershell
python -m pytest tests/engine -q
python -m pytest tests/gui -q
python -m pytest tests/utils -q
python -m pytest tests --collect-only -q
```

## Repository layout

```text
AudioPlayground/
├── src/
│   ├── engine/               # DSP engine and audio components
│   ├── gui/                  # PyQt6 modular synth UI
│   ├── modular_synth_app.py  # GUI entry point
│   ├── constants.py
│   └── utils/
├── tests/
│   ├── engine/
│   ├── gui/
│   └── utils/
├── examples/
│   ├── audio_components/
│   ├── filter/
│   ├── midi/
│   ├── module_components/
│   ├── oscillators/
│   ├── preset_builder/
│   ├── sequencer/
│   └── signal_analysis/
├── docs/                     # design notes, implementation docs, status docs
├── resources/
├── pyproject.toml
└── run_modular_synth.bat
```

## Documentation

The `docs/` folder is extensive, but it is a mix of:

- design documents
- implementation summaries
- troubleshooting notes
- historical status reports
- targeted feature write-ups

Useful starting points:

- `docs/README.md`
- `docs/REPO_ANALYSIS_2026-05-26.md`
- `docs/ARCHITECTURE_VISUAL_GUIDE.md`
- `docs/PROCESS_ARCHITECTURE_ANALYSIS.md`
- `docs/troubleshooting.md`

Some older docs still describe the project as more polished or more final than the current repository state supports, so treat historical status claims carefully.

## Development notes

### Adding GUI modules

GUI modules register themselves with `@register_module()` and are discovered recursively from `src/gui/modules/`.

### Engine imports

The stable top-level engine import surface is under `src.engine`:

```python
from src.engine import (
    SineOscillator,
    SquareOscillator,
    TriangleOscillator,
    SawtoothOscillator,
    ADSREnvelope,
    Chain,
    WaveAdder,
    AudioOutput,
)
```

## What this README intentionally does not claim

To keep this file honest, it does **not** claim that:

- the repo is fully production-ready
- every documented feature path is complete
- every test in the entire workspace has been freshly executed in this update
- the docs tree is fully consolidated or perfectly current

What it does claim is based on the current source tree, verified entry points, and the test runs listed above.

## Contributing

There is not currently a single dedicated top-level contributing guide. Use `docs/README.md`, the existing tests, and nearby module patterns as the best reference for contributing changes.

## License

See `LICENSE`.
