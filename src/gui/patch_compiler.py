"""Patch compiler - converts visual patch to audio components."""

import logging
from typing import List, Tuple, Any, Optional

from src.engine.composer import Chain
from src.engine.modifier import ModulatedVolume, ModulatedPanner
from .modules.pan_simple import SimplePannerModule
from .modules.clipper_simple import ClipperModule
from .modules.mixer import MixerModule
from .modules.output import OutputModule
from .modules.pan_mod import PannerModule
from .modules.volume_mod import VolumeModule
from .modules.volume_simple import SimpleVolumeModule
from .modules.lfo import LFOModule
from .modules.envelope_adsr import ADSRModule
from .modules.oscillator import OscillatorModule
from .patch_canvas import Port

logger = logging.getLogger(__name__)


class PatchCompiler:
    """Compiles visual patches into executable audio component graphs.

    Takes the modules and connections from the patch canvas and builds
    a working audio signal chain.
    """

    def __init__(self):
        """Initialize the patch compiler."""
        self.modules: List[Any] = []
        self.connections: List[Tuple[Port, Port]] = []
        self.compiled_patch: Optional[Any] = None

    def set_patch(self, modules: List[Any], connections: List[Tuple[Port, Port]]):
        """Set the patch to compile.

        Args:
            modules: List of module widgets
            connections: List of (output_port, input_port) tuples
        """
        self.modules = modules
        self.connections = connections

    def compile(self) -> Optional[Any]:
        """Compile the patch into an audio component.

        Returns:
            The compiled audio component or None if compilation fails
        """
        try:
            # Find output module
            output_module = None
            for module in self.modules:
                if isinstance(module, OutputModule):
                    output_module = module
                    break

            if not output_module:
                logger.error("No output module found in patch")
                return None

            # Build the signal chain backwards from output
            component = self._build_chain_from_module(output_module)

            if component is None:
                logger.error("Failed to build signal chain")
                return None

            logger.info(f"Patch compiled successfully: {type(component).__name__}")
            self.compiled_patch = component
            return component

        except Exception as e:
            logger.error(f"Patch compilation failed: {e}", exc_info=True)
            return None

    def _build_chain_from_module(self, module: Any) -> Optional[Any]:
        """Build the audio chain from a module by following connections backwards.

        Args:
            module: The module to build from

        Returns:
            The audio component or None
        """
        # For output module, get the input connection
        if isinstance(module, OutputModule):
            input_conn = self._find_connection_to_port(module.in_port)
            if input_conn:
                source_module = input_conn.parent_module
                return self._build_chain_from_module(source_module)
            else:
                logger.warning("Output module has no input connection")
                return None

        # For oscillators, just create the component
        if isinstance(module, OscillatorModule):
            module.create_component()
            return module.component

        # For ADSR envelope, create the component
        if isinstance(module, ADSRModule):
            module.create_component()
            return module.component

        # For LFO, create the component (similar to oscillator)
        if isinstance(module, LFOModule):
            module.create_component()
            return module.component

        # For mixer, collect all input connections
        if isinstance(module, MixerModule):
            # Find all connections to mixer input ports
            input_components = []
            for input_port in module.input_ports:
                input_conn = self._find_connection_to_port(input_port)
                if input_conn:
                    input_module = input_conn.parent_module
                    input_component = self._build_chain_from_module(input_module)
                    if input_component:
                        input_components.append(input_component)

            if not input_components:
                logger.warning("Mixer has no input connections")
                return None

            # Create WaveAdder with all inputs
            module.input_components = input_components
            return module.create_component()

        # For modifiers (volume, panner), need input connection
        if isinstance(module, (VolumeModule, PannerModule)):
            # Get input connection
            input_conn = self._find_connection_to_port(module.in_port)
            if not input_conn:
                logger.warning(f"{type(module).__name__} has no input connection")
                return None

            # Build input component
            input_module = input_conn.parent_module
            input_component = self._build_chain_from_module(input_module)

            if input_component is None:
                return None

            # Check for modulation connection
            mod_conn = self._find_connection_to_port(module.mod_port)
            mod_component = None
            if mod_conn:
                mod_module = mod_conn.parent_module
                mod_component = self._build_chain_from_module(mod_module)

            # Create the modifier component
            if isinstance(module, VolumeModule):
                if mod_component:
                    component = ModulatedVolume(mod_component)
                else:
                    component = module.create_component()
            elif isinstance(module, PannerModule):
                if mod_component:
                    component = ModulatedPanner(mod_component)
                else:
                    component = module.create_component()
            else:
                component = module.create_component()

            # Create chain
            return Chain(input_component, component)

        # For simple modifiers (Gain, Pan, Clipper), need input connection
        if isinstance(module, (SimpleVolumeModule, SimplePannerModule, ClipperModule)):
            # Get input connection
            input_conn = self._find_connection_to_port(module.in_port)
            if not input_conn:
                logger.warning(f"{type(module).__name__} has no input connection")
                return None

            # Build input component
            input_module = input_conn.parent_module
            input_component = self._build_chain_from_module(input_module)

            if input_component is None:
                return None

            # Create the modifier component
            component = module.create_component()

            # Create chain
            return Chain(input_component, component)

        # Unknown module type
        logger.warning(f"Unknown module type: {type(module).__name__}")
        return None

    def _find_connection_to_port(self, port: Port) -> Optional[Port]:
        """Find the source port connected to the given input port.

        Args:
            port: The input port to find connection for

        Returns:
            The source (output) port or None
        """
        for start_port, end_port in self.connections:
            if end_port == port:
                return start_port
        return None

    def get_compilation_errors(self) -> List[str]:
        """Get a list of compilation errors/warnings.

        Returns:
            List of error messages
        """
        errors = []

        # Check for output module
        has_output = any(isinstance(m, OutputModule) for m in self.modules)
        if not has_output:
            errors.append("No output module in patch")

        # Check for disconnected modules
        connected_modules = set()
        for start_port, end_port in self.connections:
            connected_modules.add(start_port.parent_module)
            connected_modules.add(end_port.parent_module)

        for module in self.modules:
            if module not in connected_modules and not isinstance(module, OutputModule):
                errors.append(f"Module '{module.module_title}' is not connected")

        # Check for invalid connections
        for start_port, end_port in self.connections:
            if start_port.port_type != "output":
                errors.append(f"Invalid connection: source port is not an output")
            if end_port.port_type != "input":
                errors.append(f"Invalid connection: destination port is not an input")

        return errors
