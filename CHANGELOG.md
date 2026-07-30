# Changelog

All notable changes to SonicRack will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow a calendar-style scheme (`YYYY.MINOR.PATCH`), aligned with
the companion `soniclab` engine package.

## [Unreleased]

### Changed

- Moved engine-domain helpers into `soniclab` (requires `soniclab>=2026.1.1`):
  oscillator frequency slewing, silence/CV/gate helpers, MIDI Trig pulse,
  WAV/channel utilities, spectrum FFT core, and shared audio constants.
  SonicRack keeps thin re-exports for existing import paths.

## [2026.1.0] - 2026-07-30

### Added

- PyPI-oriented packaging metadata: project URLs, desktop/Qt classifiers, and
  `Typing :: Typed`.
- Console entry point `sonicrack` for a standard `uv tool install sonicrack`
  (or `uv pip install sonicrack`) launch path.
- GitHub Actions publish workflow (Trusted Publishing to TestPyPI / PyPI).
- **MIDI Trig outputs**: MIDI Input and MIDI Keyboard emit a one-sample
  `Trig` pulse on every note-on (including legato retriggers while Gate stays
  high), alongside existing `1V/Oct` / Gate / Vel ports.
- **RD-8 / Behringer 182 pattern controls**: per-step gate toggle buttons and a
  **Randomize** action that scrambles CV A, CV B, and the gate pattern.
- **Envelope shape previews**: ADSR and Decay Envelope show live curve previews
  with a glowing playhead for the current phase/level.
- Shared module helpers: `SimpleModifierBase`, modulated I/O setup,
  `bind_parameter_knob`, sample-rate listener install/cleanup, and runtime
  utilities (`read_optional_port`, silence cache, CV influence blend, ramps).

### Changed

- Base install now includes the desktop runtime stack (`pyqt6`, `sounddevice`,
  `numba`) so the app starts without optional extras.
- `gui` / `audio-io` extras are kept as empty compatibility aliases.
- `midi` extra delegates to `soniclab[midi]`.
- Wheel/sdist exclude marketing screenshots and icon/splash generator scripts.
- README documents `uv`-first PyPI and development install; package version
  snapshot set to `2026.1.0`.
- Module library categories reorganized:
  - VCO is **Source** (with oscillators, LFO, noise, MIDI).
  - All sequencing modules use **Sequencer** (Clock, Step Sequencer,
    RD-8/Behringer 182, Accent, Slide).
  - Delay, Reverb, Distortion, and Compressor use **Effect** (audio-processor
    pass-through like Modifier/Filter).
  - **Modulated Source** kept for gate/pitch-driven audio generators
    (TB-303 Voice). Contour generators use **Envelope** (ADSR, Decay).
- Audio hot path: render uses `RenderContext` per-cycle state (skips full-graph
  cache invalidation), cached silence buffers, cheaper visualizer tap peeks,
  and LFO only renders connected outputs.
- Simple/modulated modifiers, effects, VCO/oscillator runtime, and VCA share
  the common control-shell and runtime helpers above (less per-module
  boilerplate).

### Fixed

- Coverage report batch script package path (`fatlife` → `sonicrack`).

### Notes

- First public packaging target for the modular rack application.
- DSP remains provided by the separate `soniclab` distribution on PyPI.

## [2026-07-06] - Performance Optimization Sprint

### Added

- **🚀 Phase 1 Critical Optimizations (Complete)**
  - **VCO Bulk Frequency API**: 30-50x faster frequency-modulated oscillators
    - Added `process_frequency_buffer()` to oscillator base class
    - VCO now uses vectorized processing instead of per-sample loops
    - Eliminates 15-25ms bottleneck → 0.3-0.5ms processing time
  - **Numba-Accelerated Effects**: 10-15x faster Delay and Reverb
    - Added JIT-compiled functions for delay processing
    - Added JIT-compiled functions for reverb processing
    - Automatic fallback to Python if Numba unavailable
    - Delay: 5-8ms → 0.3-0.5ms, Reverb: 20-35ms → 1.5-2.5ms
  - **Delay Time Crossfading**: Eliminates clicks on parameter automation
    - 5ms smooth transitions between delay positions
    - No buffer clearing on delay time changes
  - **Port Copy Optimization**: 50-90% reduction in buffer copies
    - Conditional copying based on read count
    - First read returns view, subsequent reads get copies
  - **Result**: Complex patches now process in 2-4ms vs 42-70ms (15-30x improvement)
  - **CPU Headroom**: From -300% over budget to +66-83% headroom

- **📊 Phase 2 Infrastructure (In Progress)**
  - **Buffer Pool System**: Reduce garbage collection pressure
    - `BufferPool` class for single-size buffer management
    - `MultiSizeBufferPool` for multiple buffer sizes
    - Thread-safe acquire/release with context managers
    - Statistics tracking (peak usage, fallback allocations)
    - Expected: 90% reduction in allocations, 5-10% CPU improvement
  - **Profiling Infrastructure**: Real-time performance analysis
    - `ProfilingRenderContext` for per-module timing
    - `ProfilingSession` for multi-cycle analysis
    - Text, JSON, and CSV export formats
    - Bottleneck identification and budget analysis
    - Minimal overhead (<5%) when enabled
  - **Vectorized Mixing**: 2-3x faster channel mixing
    - `vectorized_mix()` for mono mixing
    - `vectorized_mix_stereo()` with constant-power panning
    - RMS level computation and normalization utilities
    - Built-in benchmarking tools

- **🛠️ Benchmarking and Testing Tools**
  - **Patch Benchmark Tool** (`scripts/benchmarks/benchmark_patch.py`)
    - Measure patch processing time with statistical analysis
    - Calculate CPU headroom and buffer budget
    - Support for single patch or batch processing
    - Visual status indicators (excellent/good/marginal/overbudget)
  - **Click Detection Tool** (`scripts/benchmarks/test_no_clicks.py`)
    - Detect audio discontinuities and clicks
    - Analyze click severity and positions
    - Generate test audio with/without clicks
    - Support for .npy and .wav files
  - **Enhanced existing benchmarks**
    - `dsp_buffer_benchmark.py` - Component-level performance
    - `profile_hot_paths.py` - cProfile integration
    - All tools support custom buffer sizes and iteration counts

- **📚 Documentation**
  - `IMPLEMENTATION_SUMMARY.md` - Complete optimization summary
  - `WEEK2_PHASE2_PLAN.md` - Phase 2 architecture and plan
  - `OPTIMIZATION_QUICK_REFERENCE.md` - Quick reference for all tools
  - `AUDIO_CLICKS_ANALYSIS_AND_PROPOSAL.md` - Original analysis
  - `TOP_3_BOTTLENECKS_FIXED.md` - Phase 1 results
  - Comprehensive code documentation and examples

### Changed

- **VCO Module**: Now uses bulk frequency API for 30-50x speedup
- **Effects Processing**: Automatically uses Numba JIT when available
- **Port Caching**: Optimized to reduce unnecessary buffer copies
- **Delay Effect**: Smooth crossfading instead of buffer clearing

### Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Complex.apr patch | 42-70ms | 2-4ms | **15-30x** ✅ |
| VCO processing | 15-25ms | 0.3-0.5ms | **30-50x** ✅ |
| Delay processing | 5-8ms | 0.3-0.5ms | **10-15x** ✅ |
| Reverb processing | 20-35ms | 1.5-2.5ms | **10-15x** ✅ |
| Buffer copies | 6+/render | 0-1/render | **85%+** ✅ |
| CPU headroom | -300% | +66-83% | **Excellent** ✅ |

### Next Steps

- [ ] Integrate buffer pool with RenderContext
- [ ] Add profiling to GUI
- [ ] Update mixer to use vectorized operations
- [ ] Comprehensive integration testing
- [ ] Performance regression testing
- [ ] Stereo effects support (Phase 3)

---

## Backlog

- Undo / Redo system

### Review Findings

**✅ RESOLVED**: Parameter descriptors now enforce a single source of truth for runtime 
validation, smoothing policy, units, and automation semantics. All parameter behavior 
is centralized in `ParameterDescriptor` with:
  - Runtime validation (type, range, choices, clamping)
  - Smoothing policies (none/linear/exponential/logarithmic)
  - Automation modes (none/control-rate/audio-rate)
  - Automatic enforcement via `RuntimeParameter` and `ParameterRegistry`
  - See `docs/architecture/PARAMETER_SYSTEM.md` for details


## Planned Engine Work

- Introduce a first-class engine render graph with typed mono/stereo/CV/audio
  ports, topological scheduling, cycle diagnostics, graph validation, and a
  stable process-block API independent of GUI modules.
- Standardize the block-processing contract around shape, dtype, sample rate,
  reset semantics, channel count, zero-length buffers, and state carry across
  buffer boundaries.
- Replace class-name stereo special cases with explicit channel-layout metadata
  and shared mono/stereo adapter utilities.
- Add a parameter automation system with sample-accurate ramps, curves, tempo
  sync, unit conversion, smoothing policy, and thread-safe parameter updates.
- Add latency accounting, lookahead support, dry/wet latency compensation, and
  plugin-host style delay reporting for future processors.
- Add oversampling and anti-aliasing policy for nonlinear processors such as
  distortion, filters with drive, and hard clipping.
- Expand DSP modules toward full synth coverage: wavetable/sample playback,
  granular playback, oscillator sync, PWM/LFO shape sets, chorus/flanger/phaser,
  EQ, limiter, gate/expander, envelope follower, waveshapers, convolution
  reverb, and utility meters.
- Add polyphonic voice allocation, voice stealing, note expression, MIDI/MPE
  mapping, glide/portamento policy, and per-voice modulation routing.
- Build a modulation matrix that can connect CV/audio-rate modulators to any
  automatable parameter with scaling, polarity, offset, and clipping.
- Add offline rendering/export, deterministic benchmarking, golden-audio
  regression tests, denormal protection, NaN/Inf guards, and performance budgets
  for realtime-safe components.
- Harden preset serialization with schema versions, migration tests, component
  IDs, nested graph persistence, and non-callable declarative modulation specs.
- Audit wheel contents before release to confirm generated cache artifacts and
  other local-only files are excluded.

  
## Unreleased

### Added

- **ADSR shape display**: live envelope curve preview on the ADSR module with
  a glowing playhead for the current envelope position
- **Decay Envelope shape display**: matching attack-decay curve preview and
  live playhead
- **Envelope module category**: ADSR and Decay Envelope group under
  `ModuleCategory.ENVELOPE` in the module library (TB-303 Voice remains
  Modulated Source)
- **Effect CV inputs**: Delay (`CV_Time`/`CV_Feedback`/`CV_Mix`), Reverb
  (`CV_Room`/`CV_Damping`/`CV_Mix`), and Compressor (`CV_Thresh`/`CV_Ratio`/`CV_Mix`)
- **Canvas zoom/pan**: scroll-wheel zoom, middle-mouse drag to pan, View menu
  Zoom In/Out/Reset (`Ctrl+=`/`Ctrl+-`/`Ctrl+0`)
- **Step sequencer randomize**: button to randomize step values
- **Signal-colored ports/cables**: jacks and cables colored by signal kind
  (audio/CV/trigger)

### Changed

- **Effect CV path**: control-rate modulation shared via `_cv_modulation`
  helpers; Distortion no longer uses a per-sample Python loop
- **Envelope gate scan**: ADSR/Decay use vectorized Schmitt edge detection
- **Patch canvas**: `BoundingRectViewportUpdate` instead of full-viewport
  repaints during pan/zoom/cable drag
- **VCO level control**: removed gain knob and Gain CV; amplitude fixed at
  0 dB — use a VCA for level control
- **Step sequencer accent/slide**: simplified to toggles only
- **Acid Filter defaults**: knob defaults aligned with DSP and TB-303 voice
  (cutoff, resonance, env/accent, drive, output gain)

- **Enhanced Parameter System**: Complete single-source-of-truth parameter management
  - `ParameterDescriptor` now includes smoothing policies (none/linear/exponential/logarithmic)
  - `ParameterDescriptor` now includes automation modes (none/control-rate/audio-rate)
  - New `RuntimeParameter` class provides automatic validation and smoothing
  - New `ParameterRegistry` for component-level parameter management
  - Common parameter descriptors updated with appropriate smoothing policies
  - Validation methods enforce descriptor constraints at runtime
  - Sample-accurate parameter automation support
  - Integration with `ramping.py` for efficient vectorized smoothing
  - Comprehensive documentation in `docs/architecture/PARAMETER_SYSTEM.md`
  - Full test coverage in `tests/engine_t/test_parameter_system.py`

- **Volume Component Migration**: Now uses `RuntimeParameter` system
  - `amplitude` property: LINEAR smoothing in amplitude space (unchanged behavior)
  - `gain_db` property: LOGARITHMIC smoothing in dB space (enhanced perceptual linearity)
  - Eliminated manual smoothing code, uses centralized parameter system
  - Backward-compatible properties maintained for existing code
  - More efficient vectorized buffer generation
  - Smoothing behavior now enforced by descriptors

### Fixed

- **Module panel heights**: panels grow/shrink so controls fit without clipping
- **Connect/disconnect lifecycle**: notify neighbors on cable removal (including
  module delete and bulk Delete), reject duplicate/cyclic cables, clear ports
  only when fully disconnected, and roll back visual cables on model connect failure
- **Connect lifecycle hardening**: clear engine output-module registry on
  delete/clear, fully disconnect ports on canvas clear, improve port hit-testing
  and cable geometry after moves, resync modulated modules after connect/disconnect
- Fixed missing `self.` prefix in RuntimeParameter initialization
- Core DSP package is separated under `src.engine`, with optional audio-device
  and MIDI concerns kept outside the base engine import surface.
- Component model includes descriptors, parameter metadata, fluent-builder names,
  global registration, config serialization helpers, and sample-mode contracts.
- Generator support includes sine, square, sawtooth, triangle, PolyBLEP
  oscillator utilities, minBLEP helpers, VCV-style square/saw behavior, gain-dB
  handling, amplitude smoothing, and oscillator/vectorized continuity coverage.
- Noise support includes white, pink, brownian, blue, grey, velvet,
  sample-and-hold, and Perlin-style generators.
- Modulation support includes ADSR, decay and gate-triggered envelopes,
  amplitude/frequency/linear-FM/phase lanes, retrigger modes, CV scaling, and
  pitch-CV conversion helpers.
- Modifier/effect support includes volume, panning, clipping, modulated volume,
  modulated panning, modulated clipping, distortion, compressor, delay, reverb,
  Butterworth filters, resonant RBJ biquad filters, and an acid/303-style
  resonant low-pass filter.
- Composition and voice support includes `Chain`, `WaveAdder`, preset builder
  and library helpers, sample-accurate step clocking, TB-303-style sequencing,
  accent/slide processing, Behringer 182-style sequencing, and a composable
  `TB303Voice`.
- Realtime-oriented test coverage now exercises render contracts, buffer
  continuity, oscillator edge strategies, smoothing, filters, effects,
  sequencers, presets, and GUI runtime integration paths.
- Removed generated cache artifacts from `src/engine`, including the engine-local
  `.mypy_cache` and nested `__pycache__` directories, and ignored `.mypy_cache/`
  to prevent the caches from returning to source/package paths.
- Added a uniform `process_block()` API to engine modifiers, filters, and
  effects, and updated `Chain` to prefer that block-processing contract when
  applying modifiers to mono and stereo buffers.
- Moved modulated oscillator phase rendering and state commits behind
  oscillator-owned `render_modulated_waveform()` and
  `commit_modulated_phase_state()` hooks, removing direct private phase-field
  coupling from modulation internals.
- Fixed VCO waveform changes so dependent mode parameters are normalized and
  cached together; changing waveform no longer leaves the runtime with an
  invalid stale mode that can mute output until the mode selector is changed.
- Restored saved module parameters during patch loading by using the module
  `set_parameters()` API.
- Allowed Output playback to start from a right-only patch connection.
- Made GUI render graph runtime failures visible by logging and propagating
  `RuntimeError` and `AttributeError` from module processing.
- Fixed legacy `Port.read()` scalar/array mixing when a scalar connection is read
  before an array connection.
- Fixed Output playback status reporting so the main window no longer reports a
  start when the Output module intentionally remains stopped.

### Maintainability

- Made `src.gui` package imports lazy so narrow submodule imports do not
  immediately import the PyQt6 main window.
- Cleaned up this changelog so backlog items are separated from completed
  unreleased changes.


## V2026.1.0

### Added

- Engine-owned GUI render graph for live playback, monitor visualizers, and
  future sink modules such as wave writers.
- Passive visualizer audio taps: waveform and spectrum modules now read rendered
  tap history instead of driving upstream DSP.
- Standalone visualizer monitoring path: an oscillator connected only to a
  visualizer is rendered by the audio engine monitor timer without requiring an
  Output module.
- Runtime module specs with explicit `runtime_kind`, declared input/output
  ports, and parameter snapshots for compiled graph nodes.
- Module-owned runtime processors: GUI modules now declare their own runtime
  specs and DSP adapters, while `AudioEngine` keeps graph traversal and render
  ownership.
- Architecture documentation in affected runtime modules describing render
  ownership, passive sinks, and the shared graph path.
- Rack-style module headers with category labels, accent rails, and a shared
  power/bypass control.
- Improved knob spacing, module sizing, and control contrast for denser rack
  layouts.
- Improved input/output port label placement and contrast.
- Module active state persistence in patch parameters.
- Runtime graph bypass behavior for inactive modules: sources mute, modifiers
  pass their input through.
- Added module subpackages and recursive search in module registry.
- Added ADSR retrigger modes: `Punch` performs a short click-safe reset before
  attack for percussive repeated gates, while `Legato` retriggers from the
  current envelope level for smoother overlaps.
- Added user documentation for the on-screen MIDI Keyboard module, including
  `Freq`, `Gate`, and `Vel` patching patterns for VCO, ADSR, VCA, visualizers,
  and parallel outputs.
- BLEP anti-aliasing for oscillators
- Polish API and documentation and code (comprehensive update complete)
- Documentation hub with clear navigation (docs/README.md)
- Complete package documentation (Engine, GUI, Builder)
- Quick Start Guide (5-minute tutorial)
- Architecture overview with system design
- Comprehensive troubleshooting guide
- Complete documentation index
- Learning paths for users, developers, musicians
- API reference structure
- VCA module implementation and runtime graph fix
- Voltage controlled amplifier module (VCA)
- Clarified module interface around `RuntimeModuleSpec` and `process_runtime()`
- Fixed mixer: independent oscillator outputs per port
- Added mix_mode parameter to WaveAdder (sum/average modes)
- Volume knobs for mixer with hot-swapping (click-free)
- Per-channel gain controls in mixer (4 channels)
- Mixer properly isolates channels with Volume components
- Comprehensive developer documentation for module creation
- Reverb, delay and distortion effects
- Automatic cv range adaption
- Manual button to trigger adsr envelope
- PW support for square waves
- Oscillator enhancements
- No notes on midi playback
- Fix Lock and unlock knobs in modulated widgets
- Modulated clipper
- Noise with db
- Panner and Volume module not working
- Store name of current patch, to allow direct Save or Save As

## [1.0.0] - 2025-11-04

### Initial Release

First release of AudioPlayground!

### Added

#### Audio Engine
- **Oscillators**: Sine, Square, Sawtooth, Triangle (fully vectorized)
- **Noise Generators**: White, Pink, Brown, Blue, Grey, Velvet, Sample & Hold
- **Modulators**: ADSR Envelope, Perlin Noise
- **Modifiers**: Volume (simple & modulated), Panner (simple & modulated), Clipper
- **Composers**: WaveAdder (mixer), Chain, ModulatedOscillator (FM/AM/PM synthesis)
- **Filters**: Butterworth IIR filter implementation
- **Performance**: 100-1000x realtime, 5-47M samples/sec throughput

#### GUI System
- **11 Built-in Modules**: Oscillator, LFO, ADSR, Noise, Mixer, Gain, Volume (Mod), Panner, Panner (Mod), Clipper, Output
- **Visual Patching**: Drag-and-drop module placement, cable routing
- **Real-time Audio**: Live playback with <10ms latency
- **Visualizations**: Stereo waveform display, FFT spectrum analyzer
- **Auto-compilation**: Automatic patch updates on changes
- **Preset System**: Save/load complete patches as JSON

#### Builder API
- **Fluent Interface**: Chainable method calls for patch creation
- **Type Safety**: Full type hints throughout
- **Component Registry**: Automatic method generation for all components
- **Preset Management**: JSON serialization/deserialization

#### Infrastructure
- **Dynamic Component Registry**: Auto-discovery and registration
- **Plugin System**: Drop-in custom modules
- **Comprehensive Testing**: 434 tests with 100% pass rate
- **Full Type Coverage**: Type hints for all public APIs
- **Extensive Documentation**: 10+ documentation files

### Performance
- **Vectorized Processing**: NumPy optimization throughout
- **Real-time Capable**: 40-80 simultaneous voices
- **Low CPU**: 5-15% per voice
- **Memory Efficient**: Minimal allocations, pre-allocated buffers

### Documentation
- Getting Started guide
- Module reference
- API documentation
- Architecture overview
- Contributing guidelines

### Quality
- 434 tests passing (100%)
- Full type hints
- Google-style docstrings
- PEP 8 compliant
- Zero technical debt

---

## [Unreleased]

### Added
- Split core DSP from IO concerns with standalone `src.engine`, `src.audio_io`, and
  `src.midi_io` package boundaries.
- Added realtime audio callback contracts and callback diagnostics for status/error
  tracking without logging from the sounddevice callback.
- Added DSP and audio callback benchmark scripts, including allocation tracking and
  hot-path profiling helpers.
- Added realtime continuity coverage for filters, effects, oscillators, modulated
  chains, nested composer graphs, and stateful chains at `44100`, `48000`, and
  `96000` Hz.
- Added shared engine validation helpers for sample-rate and render-length
  contracts.
- Added shared minBLEP oscillator helpers for band-limited edge correction.
- Added VCV-style sawtooth oscillator mode with iterator and vectorized rendering
  support.
- Added optional non-linear knob curves, used by audio-rate oscillator frequency
  controls for finer low-frequency resolution.

### Changed
- Moved audio-device playback and MIDI functionality out of the engine package so
  the base wheel remains GUI- and device-backend independent.
- Reworked audio output callbacks to reuse buffers, fill `outdata` directly, and
  avoid steady-state callback allocations.
- Optimized VCV-style square generation and the `Reverb` hot path based on local
  benchmark/profile results.
- Standardized finite positive sample-rate validation and non-negative integer
  render-length validation across core engine entry points.
- Changed `NoiseGenerator` smoothing to derive its 10 ms window from the configured
  sample rate.
- Narrowed non-callback exception handling in scoped engine/audio_io/midi_io code
  to explicit IO/runtime/parse exception types.
- Changed VCV-style square oscillator edge handling so up- and down-phase
  traversal are symmetric.
- Changed oscillator and VCO frequency knobs to span roughly `11 Hz` to `6000 Hz`
  with a pitch-oriented response curve.
- Smoothed GUI oscillator and VCO frequency changes with per-sample pitch-space
  slewing for more natural sweeps.

### Fixed
- Added a portable preset graph payload alongside legacy flat component presets,
  including explicit node IDs, audio edges, output topology, and declarative
  modulation edges so saved presets do not rely on serializing Python callables.
- Fixed stereo block handling for stateful effects and smoothed volume paths:
  delay, reverb, compressor, volume, and modulated volume now preserve `(n, 2)`
  buffers without channel-state bleed.
- Fixed state carry across buffer boundaries for `ButterworthFilter`.
- Fixed stereo `Chain` plus `ButterworthFilter` rendering so channel state is
  preserved independently.
- Fixed analog sine/sawtooth/triangle vectorized coloration so continuous sample
  index state is preserved across buffers.
- Fixed `ADSREnvelope(sample_rate=...)` so explicit sample rates survive base
  initialization and drive phase sample counts correctly.
- Fixed MIDI synth render paths to propagate voice-generation errors instead of
  silently returning or mixing silence.
- Kept broad exception handling only at explicit callback isolation boundaries:
  sounddevice callback diagnostics and MIDI user callback dispatch.
- Fixed VCO and oscillator frequency-slew state so partial glides continue across
  audio buffers instead of snapping oscillator internals to the target frequency.
- Fixed split-buffer oscillator frequency sweeps to match continuous rendering.

### Tests
- Added regression tests for runtime contracts, cross-rate continuity, callback
  buffer reuse/allocation behavior, and MIDI synth render-error propagation.
- Added regression tests for VCV square symmetry, VCV sawtooth rendering,
  non-linear knob mapping, and GUI oscillator/VCO frequency-slew continuity.
- Current scoped verification: engine + audio IO + MIDI IO tests pass with
  `782 passed, 2 skipped, 8 subtests passed`.

### Planned for v1.1
- MIDI input support
- Filter modules (low-pass, high-pass, band-pass)
- Additional effects (reverb, delay, chorus)
- UI improvements (zoom, undo/redo)
- More factory presets

### Planned for v1.2
- Polyphony system
- Wavetable synthesis
- Pattern sequencer
- Automation recording

### Planned for v2.0
- VST/AU plugin export
- Multi-track sequencer
- Advanced modulation matrix
- Sample playback engine
- Plugin marketplace

---

## Release Notes

### v1.0.0 Highlights

**AudioPlayground 1.0** represents the culmination of extensive development work, delivering:

1. **World-Class Performance**: Fully vectorized audio engine achieving 100-1000x realtime performance
2. **Professional UI**: Polished modular synthesizer interface with 11 modules
3. **Production Ready**: Comprehensive testing, documentation, and clean architecture
4. **Extensible**: Plugin-ready system for custom modules
5. **Developer Friendly**: Full type hints, extensive docs, multiple APIs

**Perfect for**:
- Electronic music production
- Sound design and experimentation
- Audio algorithm prototyping
- Teaching/learning synthesis
- Game audio development

**Get Started**: See [Quick Start Guide](docs/user-guide/getting-started/quick-start-gui.md)

---

## Version History

| Version | Date | Status | Highlights |
|---------|------|--------|------------|
| 1.0.0 | 2025-11-04 | ✅ Released | Initial production release |

---

[1.0.0]: https://github.com/yourusername/AudioPlayground/releases/tag/v1.0.0
[Unreleased]: https://github.com/yourusername/AudioPlayground/compare/v1.0.0...HEAD

