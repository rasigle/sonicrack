"""Generic interface for audio modules in the patch system.

This interface allows the patch compiler to work with any module type
without needing specific knowledge about each module's implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ModuleCategory(StrEnum):
    """Categorizes modules by their role in the signal chain."""

    SOURCE = "Source"  # Oscillators, LFOs, Envelopes - no audio input required
    MODIFIER = (
        "Modifier"  # Volume, Pan, Clipper - single audio input + optional modulation
    )
    MIXER = "Mixer"  # Combines multiple audio inputs
    OUTPUT = "Output"  # Terminal node


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


class AudioModuleInterface(ABC):
    """Base interface for all audio modules in the patch system.

    Any module that can be placed on the canvas and compiled into an audio
    component should implement this interface.

    Attributes:
        custom_name (str): Optional custom name for the module instance.
    """

    metadata: ModuleMetadata

    def __init__(self):
        self.custom_name: str = ""

    @abstractmethod
    def create_component(
        self,
        input_components: list[Any] | None = None,
        modulation_components: dict[str, Any] | None = None,
    ) -> Any:
        """Create the audio engine component for this module.

        This is called by the patch compiler to instantiate the actual
        audio processing component based on the module's current parameters.

        Args:
            input_components: List of compiled audio components from input connections.
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
        self.update()  # Trigger repaint

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

        # Check required inputs are connected
        for port_name in self.get_required_inputs():
            port = self._find_port_by_name(port_name)
            if port and not self._is_port_connected(port, connections):
                module_name = getattr(self, "module_title", "Unknown Module")
                errors.append(
                    f"{module_name}: Required input '{port_name}' is not connected"
                )

        return errors

    def _find_port_by_name(self, port_name: str):
        """Helper to find a port by name."""
        for port in getattr(self, "input_ports", []):
            if port.port_name == port_name:
                return port
        return None

    def _is_port_connected(self, port, connections: list[tuple]) -> bool:
        """Helper to check if a port is connected."""
        for start_port, end_port in connections:
            if end_port == port:
                return True
        return False
