"""Generic interface for audio modules in the patch system.

This interface allows the patch compiler to work with any module type
without needing specific knowledge about each module's implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from src.engine import AudioComponent


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

    # Terminal node
    OUTPUT = "Output"


@dataclass
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
    """

    metadata: ModuleMetadata

    def __init__(self):
        self.custom_name: str = ""

    @abstractmethod
    def create_engine_component(
        self,
        input_components: list[AudioComponent] | None = None,
        modulation_components: dict[str, AudioComponent] | None = None,
    ) -> AudioComponent:
        """Create the corresponding audio engine component for this module.

        This is called by the patch compiler to instantiate the actual
        audio processing component based on the module's current parameters.

        Args:
            input_components: List of compiled audio components from input
                connections.
                For SOURCE modules, this is None.
                For MODIFIER modules, this contains exactly one component.
                For MIXER modules, this contains multiple components.
            modulation_components: Dictionary mapping modulation port names to
                their compiled components (e.g., {"mod": lfo_component})

        Returns:
            The audio engine component (Oscillator, Envelope, Chain, etc.)
        """
        pass

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

    def validate_connections(self, connections: list[tuple]) -> list[str]:
        """Validate module connections.

        Override this to add custom validation logic for your module.

        Args:
            connections: List of all (source_port, dest_port) connections

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Note: We don't validate required inputs here because the patch compiler
        # gracefully skips modules with missing inputs during compilation.
        # This allows for progressive patch building where modules can be connected
        # to outputs before their inputs are connected.

        # Custom validation can be added by overriding this method in subclasses

        # Custom validation could be that a module is connected with an
        # incompatible one.

        return errors

    def _find_port_by_name(self, port_name: str):
        """Helper to find a port by name."""
        for port in getattr(self, "input_ports", []):
            if port.port_name == port_name:
                return port
        return None

    @staticmethod
    def _is_port_connected(port, connections: list[tuple]) -> bool:
        """Helper to check if a port is connected."""
        for start_port, end_port in connections:
            if end_port == port:
                return True
        return False
