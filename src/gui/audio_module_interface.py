"""Generic interface for audio modules in the patch system.

This interface allows the patch compiler to work with any module type
without needing specific knowledge about each module's implementation.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional


class ModuleCategory(Enum):
    """Categorizes modules by their role in the signal chain."""

    SOURCE = "source"  # Oscillators, LFOs, Envelopes - no audio input required
    MODIFIER = (
        "modifier"  # Volume, Pan, Clipper - single audio input + optional modulation
    )
    MIXER = "mixer"  # Combines multiple audio inputs
    OUTPUT = "output"  # Terminal node


class AudioModuleInterface(ABC):
    """Base interface for all audio modules in the patch system.

    Any module that can be placed on the canvas and compiled into an audio
    component should implement this interface.
    """

    @abstractmethod
    def get_module_category(self) -> ModuleCategory:
        """Return the category of this module.

        Returns:
            The ModuleType enum indicating the module's role
        """
        pass

    @abstractmethod
    def create_component(
        self,
        input_components: Optional[List[Any]] = None,
        modulation_components: Optional[Dict[str, Any]] = None,
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

    def get_required_inputs(self) -> List[str]:
        """Return list of required input port names.

        These are the main audio input ports that must be connected for
        the module to function (e.g., "In" for a volume modifier).

        Returns:
            List of required input port names (empty for SOURCE modules)
        """
        return []

    def get_modulation_inputs(self) -> List[str]:
        """Return list of optional modulation input port names.

        These are control inputs that can modulate parameters
        (e.g., "Mod" for modulated volume).

        Returns:
            List of modulation port names
        """
        return []

    def validate_connections(self, connections: List[tuple]) -> List[str]:
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

    def _is_port_connected(self, port, connections: List[tuple]) -> bool:
        """Helper to check if a port is connected."""
        for start_port, end_port in connections:
            if end_port == port:
                return True
        return False
