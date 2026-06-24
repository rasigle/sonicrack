"""Generic interface for audio modules in the patch system.

This interface allows the patch compiler to work with any module type
without needing specific knowledge about each module's implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from src.engine.audio_component import AudioComponent
from src.gui.core.port import Port


class ModuleCategory(StrEnum):
    """Categorizes modules by their role in the signal chain.

    Attributes:
        SOURCE: Represents modules that generate audio signals without requiring
            any input (e.g., Oscillators, LFOs, Envelopes).
        MODULATED_SOURCE: Represents modules that generate audio signals but
            require control voltage (CV) inputs for modulation
            (e.g., Voltage-Controlled Oscillators).
        MODIFIER: Represents modules that modify an audio signal,
            typically with one audio input and optional modulation inputs
            (e.g., Volume, Pan, Clipper).
        MIXER: Represents modules that combine multiple audio inputs into
            a single output.
        VISUALIZATION: Represents modules that display audio signals visually
            without modifying them (e.g., Waveform, Spectrum Analyzer).
            These act as pass-through modules.
        OUTPUT: Represents terminal modules that act as the final node
            in the signal chain (e.g., speakers, audio output).
    """

    # Oscillators, LFOs, Envelopes - no audio input required
    SOURCE = "Source"

    # Oscillators with CV inputs (VCO, etc.) - takes CV, outputs audio
    MODULATED_SOURCE = "Modulated Source"

    # Volume, Pan, Clipper - single audio input + optional modulation
    MODIFIER = "Modifier"

    # Combines multiple audio inputs
    MIXER = "Mixer"

    # Visualization tools - pass-through with display
    VISUALIZATION = "Visualization"

    # Terminal node
    OUTPUT = "Output"


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

    def __init__(self):
        self.custom_name: str = ""

        self.inputs: dict[str, Port] = {}
        self.outputs: dict[str, Port] = {}

    @abstractmethod
    def create_engine_component(
        self,
        input_components: list[AudioComponent] | None = None,
        modulation_components: dict[str, AudioComponent] | None = None,
    ) -> AudioComponent:
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
        """Helper to find a port by name."""
        for port in getattr(self, "input_ports", []):
            if port.port_name == port_name:
                return port
        return None

    def add_input(self, name: str, component=None):
        """Add an input port to this module.

        Args:
            name: Name of the input port
            component: Optional engine component associated with this port

        Returns:
            The created Port instance
        """
        port = Port("input", name, parent_module=self, component=component)
        self.inputs[name] = port
        return port

    def add_output(self, name: str, component=None) -> Port:
        """Add an output port to this module.

        Args:
            name: Name of the output port
            component: Optional engine component associated with this port
                      (e.g., SineOscillator for a "Sine" output)

        Returns:
            The created Port instance
        """
        port = Port("output", name, parent_module=self, component=component)
        self.outputs[name] = port
        return port

    @abstractmethod
    def process(self, num_samples: int = 1):
        """Process method placeholder.

        This method can be overridden by subclasses to implement
        any necessary processing logic specific to the module.
        """
