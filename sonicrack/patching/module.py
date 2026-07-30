"""Core audio-module contracts used by GUI graph compilation and runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from soniclab.core.component import AudioComponent

from sonicrack.patching.port import Port, PortSignal, PortType, normalize_port_signal


class ModuleCategory(StrEnum):
    """Categorizes modules by their role in the signal chain.

    Attributes:
        SOURCE: Free-running or continuously generating modules that do not
            need a primary audio input (e.g., Oscillators, VCO, LFO, Noise,
            MIDI input).
        MODULATED_SOURCE: Gate/pitch-driven generators that need control
            inputs to produce their main output (e.g., monophonic voices such
            as TB-303).
        ENVELOPE: Contour generators driven by gate/trigger (e.g., ADSR,
            attack-decay). Produce control CV rather than audio.
        SEQUENCER: Timing, pattern, and sequencing helpers (clocks, step
            sequencers, accent/slide utilities).
        MODIFIER: Modules that modify an audio signal, typically with one
            audio input and optional modulation inputs (e.g., Volume, Pan,
            Clipper, VCA).
        FILTER: Frequency-domain filter modules that process audio
            (e.g., Butterworth, Resonant, Acid). Behaves like a modifier for
            routing and inactive pass-through.
        EFFECT: Time- and dynamics-based effects that process audio
            (e.g., Delay, Reverb, Distortion, Compressor). Behaves like a
            modifier for routing and inactive pass-through.
        MIXER: Modules that combine multiple audio inputs into a single
            output.
        VISUALIZATION: Modules that display audio signals visually without
            modifying them (e.g., Waveform, Spectrum Analyzer). Pass-through.
        OUTPUT: Terminal modules that act as the final node in the signal
            chain (e.g., speakers, audio output).
    """

    # Oscillators, VCO, LFO, noise, MIDI — no primary audio input
    SOURCE = "Source"

    # Gate/pitch-driven audio generators (monophonic voices)
    MODULATED_SOURCE = "Modulated Source"

    # Contour generators (ADSR, AD, …) — control CV output
    ENVELOPE = "Envelope"

    # Clocks, step sequencers, accent/slide sequencing helpers
    SEQUENCER = "Sequencer"

    # Volume, Pan, Clipper, VCA — single audio input + optional modulation
    MODIFIER = "Modifier"

    # Frequency filters — audio in/out (utility, resonant, acid, …)
    FILTER = "Filter"

    # Delay, reverb, distortion, compressor — audio FX processing
    EFFECT = "Effect"

    # Combines multiple audio inputs
    MIXER = "Mixer"

    # Visualization tools — pass-through with display
    VISUALIZATION = "Visualization"

    # Terminal node
    OUTPUT = "Output"

    def is_audio_processor(self) -> bool:
        """True for categories that transform a primary audio input.

        Used for inactive pass-through and similar modifier-like routing.
        """
        return self in (
            ModuleCategory.MODIFIER,
            ModuleCategory.FILTER,
            ModuleCategory.EFFECT,
        )


@dataclass(frozen=True)
class ModuleMetadata:
    """Metadata for audio modules.

    Attributes:
        title (str): Human-readable title of the module.
        description (str): Short description of the module's functionality.
        author (str): Author or creator of the module.
        version (str): Version string of the module.
    """

    title: str
    category: ModuleCategory
    description: str = ""
    author: str = ""
    version: str = "1.0.0"


class AudioModule(ABC):
    """Base interface for all audio modules in the ui patch system.

    Any module that can be placed on the canvas and compiled into an audio
    component should implement this interface.

    An AudioModule represents the configuration and parameters of a module
    as set by the user in the GUI. The actual audio processing logic is handled
    by the AudioComponent created by the `create_engine_component()` method.

    Attributes:
        custom_name (str): Optional custom name for the module instance.
        metadata (ModuleMetadata): Static metadata about the module type.
    """

    metadata: ModuleMetadata
    input_ports: list[Any]
    output_ports: list[Any]

    def __init__(self):
        self.custom_name: str = ""

        self.inputs: dict[str, Port] = {}
        self.outputs: dict[str, Port] = {}

    @abstractmethod
    def create_engine_component(
        self,
        input_components: list[AudioComponent] | None = None,
        modulation_components: dict[str, AudioComponent] | None = None,
    ) -> AudioComponent | None:
        """Create the corresponding audio engine component for this module.

        This is called by the patch compiler to instantiate the actual
        audio processing component based on the module's current parameters.

        **IMPORTANT**: This method is called when:
        - The module has a single output, OR
        - The module has multiple outputs but get_output_component() is NOT implemented

        For modules with multiple INDEPENDENT outputs (e.g., Oscillator with Sine,
        Triangle, Square, Sawtooth), implement get_output_component() instead.
        This method will then only be called if get_output_component() is not defined.

        Args:
            input_components: List of compiled audio components from input
                connections.
                - SOURCE modules: None (generate audio from scratch)
                - MODIFIER modules: Exactly one component (transform audio)
                - MIXER modules: Multiple components (combine audio)
            modulation_components: Dictionary mapping modulation port names to
                their compiled components (e.g., {"freq_mod": lfo_component})

        Returns:
            The audio engine component (Oscillator, Envelope, Chain, etc.)

        Examples:
            # Simple oscillator (single output):
            return SineOscillator(self.freq_knob.get_value())

            # Volume modifier (transforms input):
            return Volume(amplitude=self.gain_knob.get_value())

            # Mixer (combines inputs):
            processed = [Chain(inp, Volume(gain)) for inp, gain in ...]
            return WaveAdder(*processed, mix_mode="sum")
        """

    def get_output_component(self, port_name: str) -> AudioComponent | None:
        """Get the component for a SPECIFIC output port.

        **OPTIONAL METHOD**: Only implement this if your module has multiple
        INDEPENDENT outputs that should produce different signals.

        If this method is implemented, it takes precedence over
        create_engine_component() when building connections from this module's outputs.

        **When to implement this:**
        - Module has multiple output ports (Sine, Triangle, Square, Sawtooth)
        - Each output should produce a DIFFERENT signal (not mixed together)
        - Example: Oscillator module where each waveform is independent

        **When NOT to implement this:**
        - Module has single output → use create_engine_component() only
        - Multiple outputs should produce the SAME signal →
          use create_engine_component()
        - Example: LFO that broadcasts the same signal to multiple destinations

        Args:
            port_name: Name of the output port (e.g., "Sine", "Triangle")

        Returns:
            The audio component for that specific output port, or None if
            the port is not connected or invalid.

        Example (Oscillator with independent outputs):
            def get_output_component(self, port_name: str):
                freq = self.freq_knob.get_value()
                if port_name == "Sine":
                    return SineOscillator(freq)
                elif port_name == "Triangle":
                    return TriangleOscillator(freq)
                elif port_name == "Square":
                    return SquareOscillator(freq)
                return None

        Note:
            If this method is NOT implemented, the patch compiler will fall back
            to calling create_engine_component() and use that for ALL outputs.
        """
        # Default: not implemented - use create_engine_component() instead
        raise NotImplementedError()

    def get_required_inputs(self) -> list[str]:
        """Return list of required input port names.

        These are the main audio input ports that must be connected for
        the module to function (e.g., "In" for a volume modifier).

        Returns:
            List of required input port names (empty for SOURCE modules)
        """
        return []

    def get_modulation_inputs(self) -> list[str]:
        """Return list of optional modulation input port names.

        These are control inputs that can modulate parameters
        (e.g., "Mod" for modulated volume).

        Returns:
            List of modulation port names
        """
        return []

    def set_custom_name(self, name: str):
        """Set a custom name for this module instance.

        Args:
            name: Custom name to display
        """
        self.custom_name = name

    def get_custom_name(self) -> str:
        """Get the custom name for this module.

        Returns:
            Custom name, or empty string if not set
        """
        return self.custom_name

    def _find_port_by_name(self, port_name: str):
        """Find a UI port widget by name, checking inputs then outputs."""
        for port in self.input_ports:
            if port.port_name == port_name:
                return port
        for port in self.output_ports:
            if port.port_name == port_name:
                return port
        return None

    def add_input(
        self,
        name: str,
        component=None,
        signal: str | PortSignal | None = None,
    ):
        """Add an input port to this module.

        Args:
            name: Name of the input port
            component: Optional engine component associated with this port
            signal: Specifies type of Signal.

        Returns:
            The created Port instance
        """
        port = Port(
            PortType.INPUT,
            name,
            parent_module=self,
            component=component,
            signal=signal or infer_port_signal(name, "input"),
        )
        self.inputs[name] = port
        return port

    def add_output(
        self,
        name: str,
        component=None,
        signal: str | PortSignal | None = None,
    ) -> Port:
        """Add an output port to this module.

        Args:
            name: Name of the output port
            component: Optional engine component associated with this port
                      (e.g., SineOscillator for a "Sine" output)
            signal: Specifies type of Signal.

        Returns:
            The created Port instance
        """
        port = Port(
            PortType.OUTPUT,
            name,
            parent_module=self,
            component=component,
            signal=signal or infer_port_signal(name, "output"),
        )
        self.outputs[name] = port
        return port


def infer_port_signal(name: str, direction: str) -> PortSignal:
    """Infer a conservative signal kind from established port names."""
    normalized = name.strip().lower().replace("_", " ")

    if normalized in {"1v/oct", "v/oct", "pitch"}:
        return PortSignal.PITCH_CV

    if normalized in {"freq", "freq in", "freq out", "hz", "frequency hz", "freq hz"}:
        return PortSignal.FREQUENCY_HZ

    if normalized in {"gate", "hold"}:
        return PortSignal.GATE

    if normalized in {"trig", "trigger", "clock", "reset", "end"}:
        return PortSignal.TRIGGER

    if normalized in {
        "accent",
        "slide",
        "vel",
        "cv",
        "cv in",
        "cv a",
        "cv b",
        "cutoff cv",
        "env cv",
        "amp cv",
        "accent cv",
        "cv drive",
        "cv mix",
        "cv_drive",
        "cv_mix",
        "mod",
        "fm",
        "gain",
    }:
        return PortSignal.CONTROL_CV

    if normalized in {
        "in",
        "in 1",
        "in 2",
        "in 3",
        "in 4",
        "out",
        "left/mono",
        "right",
        "l/mono",
        "r",
    }:
        return PortSignal.AUDIO

    if direction == "output" and normalized in {
        "sine",
        "triangle",
        "sawtooth",
        "square",
        "noise",
    }:
        return PortSignal.AUDIO

    return normalize_port_signal(None)
