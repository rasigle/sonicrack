"""High-performance, GUI-independent audio DSP engine.

AudioPlayground Engine is a modular audio synthesis framework designed for real-time
synthesis, sound design, and music production. The engine provides vectorized
oscillators, flexible signal routing, and a unified API across all components.

Components:
    Oscillators: Generate basic waveforms (Sine, Square, Sawtooth, Triangle)
    Modulators: Time-varying control signals (ADSR envelopes)
    Modulated Oscillators: Combine oscillators with modulators for expressive synthesis
    Composers: Route and combine signals (Chain for serial, WaveAdder for parallel)
    Modifiers: Process audio signals (Panning, Volume, Clipping, Frequency)
    Filters: Shape frequency content (Butterworth IIR filters)
    Noise Generators: Various noise types for synthesis and sound design

Performance:
    - Vectorized mode: 50-85x faster than iterator mode
    - Auto mode: Automatically select the best method based on buffer size
    - Real-time ready: Capable of 512-2048 sample buffers at 44.1kHz

Example - Basic Synthesis:
    >>> from src.engine import SineOscillator
    >>> osc = SineOscillator(frequency=440, amplitude=0.8)
    >>> samples = osc.get_samples(44100)  # 1 second at 44.1kHz

Example - Modulated Synthesis:
    >>> from src.engine import SineOscillator, ADSREnvelope, ModulatedOscillator
    >>>
    >>> # Create oscillator and envelope
    >>> osc = SineOscillator(frequency=440, amplitude=1.0)
    >>> env = ADSREnvelope(
    ...     attack_duration=0.1,
    ...     decay_duration=0.2,
    ...     sustain_level=0.7,
    ...     release_duration=0.3
    ... )
    >>>
    >>> # Combine them
    >>> mod_osc = ModulatedOscillator(
    ...     osc,
    ...     env,
    ...     amp_mod=lambda base_amp, env_val: base_amp * env_val
    ... )
    >>>
    >>> # Generate audio with envelope
    >>> samples = mod_osc.get_samples(44100)
    >>> mod_osc.trigger_release()

Example - Signal Chain:
    >>> from src.engine import SineOscillator, Chain, Volume, Panner
    >>>
    >>> osc = SineOscillator(440)
    >>> chain = Chain(
    ...     osc,
    ...     Volume(0.5),      # Reduce amplitude
    ...     Panner(0.7)       # Pan to right
    ... )
    >>> stereo_samples = chain.get_samples(1000)

Example - Mixing Oscillators:
    >>> from src.engine import SineOscillator, WaveAdder
    >>>
    >>> osc1 = SineOscillator(440)   # A4
    >>> osc2 = SineOscillator(880)   # A5
    >>> osc3 = SineOscillator(1320)  # E6
    >>>
    >>> chord = WaveAdder(osc1, osc2, osc3)
    >>> mixed = chord.get_samples(44100)

API Modes:
    All audio generators support three sample generation modes:

    - get_samples_iterator(n): Python loops, flexible, slower
    - get_samples_vectorized(n): NumPy arrays, 50-85x faster
    - get_samples(n, mode='auto'): Automatically selects best method

    Auto mode uses vectorized for n >= 512, iterator for n < 512.

Note:
    For production use, prefer get_samples_vectorized() or auto mode.
    The iterator mode is provided for prototyping and fine-grained control.

"""

from src.engine.audio_component import (
    AudioComponent,
    COMMON_PARAMETER_DESCRIPTORS,
    ComponentCategory,
    ComponentDescriptor,
    ParameterDescriptor,
    make_parameter_descriptors,
)
from src.engine.audio_component_registry import (
    AudioComponentRegistry,
    audio_registry,
    register_component,
)
from src.engine.composer import (
    Chain,
    Composer,
    WaveAdder,
)
from src.engine.cv_utils import (
    CVScaler,
    bipolar_to_unipolar,
    scale_cv,
    unipolar_to_bipolar,
)
from src.engine.effects import (
    Delay,
    Distortion,
    Reverb,
)
from src.engine.filter import ButterworthFilter, apply_filter, butter
from src.engine.modifier import (
    Clipper,
    Frequency,
    Modifier,
    ModulatedClipper,
    ModulatedPanner,
    ModulatedVolume,
    Panner,
    Volume,
)
from src.engine.modulator import (
    ADSREnvelope,
    Modulator,
    getadsr,
)
from src.engine.noise import (
    NoiseGenerator,
    blue_noise,
    brownian_noise,
    grey_noise,
    perlin_noise,
    pink_noise,
    sample_hold_noise,
    velvet_noise,
    white_noise,
)
from src.engine.oscillator import (
    Oscillator,
    SawtoothOscillator,
    SineOscillator,
    SquareOscillator,
    TriangleOscillator,
)
from src.engine.oscillator_modulated import ModulatedFrequency, ModulatedOscillator
from src.engine.presets.preset_builder import PresetBuilder
from src.engine.presets.preset_library import PresetLibrary
from src.engine.synthesis import synth

__all__ = [
    "AudioComponent",
    "COMMON_PARAMETER_DESCRIPTORS",
    "ComponentCategory",
    "ComponentDescriptor",
    "ParameterDescriptor",
    "make_parameter_descriptors",
    "AudioComponentRegistry",
    "register_component",
    "audio_registry",
    "Oscillator",
    "SineOscillator",
    "SquareOscillator",
    "SawtoothOscillator",
    "TriangleOscillator",
    "synth",
    "Modulator",
    "ADSREnvelope",
    "getadsr",
    "ModulatedOscillator",
    "Composer",
    "Chain",
    "WaveAdder",
    "Modifier",
    "Panner",
    "ModulatedPanner",
    "Volume",
    "ModulatedVolume",
    "Frequency",
    "ModulatedFrequency",
    "Clipper",
    "ModulatedClipper",
    "CVScaler",
    "bipolar_to_unipolar",
    "unipolar_to_bipolar",
    "scale_cv",
    "butter",
    "apply_filter",
    "ButterworthFilter",
    "white_noise",
    "pink_noise",
    "brownian_noise",
    "blue_noise",
    "perlin_noise",
    "velvet_noise",
    "grey_noise",
    "sample_hold_noise",
    "NoiseGenerator",
    "Distortion",
    "Delay",
    "Reverb",
    "PresetBuilder",
    "PresetLibrary",
]
