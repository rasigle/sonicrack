"""Patch compiler - converts visual patch to audio components."""

import logging
from typing import Any

from src.engine.composer import Chain
from src.gui.audio_module_interface import ModuleCategory, AudioModuleInterface
from src.gui.patch_canvas import Port

logger = logging.getLogger(__name__)


class PatchCompiler:
    """Compiles visual patches into executable audio component graphs.

    Takes the modules and connections from the patch canvas and builds
    a working audio signal chain using the generic AudioModuleInterface.

    This compiler is extensible - it works with any module that implements
    the AudioModuleInterface without needing module-specific code.
    """

    def __init__(self):
        """Initialize the patch compiler."""
        self.modules: list[AudioModuleInterface] = []
        self.connections: list[tuple[Port, Port]] = []
        self.compiled_patch: Any | None = None
        self._build_cache: dict[AudioModuleInterface, Any] = {}

    def set_patch(
        self, modules: list[AudioModuleInterface], connections: list[tuple[Port, Port]]
    ):
        """Set the patch to compile.

        Args:
            modules: List of module widgets implementing AudioModuleInterface
            connections: List of (output_port, input_port) tuples
        """
        self.modules = modules
        self.connections = connections
        self._build_cache = {}

    def compile(self) -> Any | None:
        """Compile the patch into an audio component.

        Returns:
            The compiled audio component or None if compilation fails
        """
        try:
            # Find output module
            output_module = None
            for module in self.modules:
                if module.get_module_category() == ModuleCategory.OUTPUT:
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

    def _build_chain_from_module(self, module: AudioModuleInterface) -> Any | None:
        """Build the audio chain from a module by following connections backwards.

        This method uses the generic AudioModuleInterface to work with any module type.

        Args:
            module: The module to build from

        Returns:
            The audio component or None
        """
        # Check cache to avoid rebuilding the same module multiple times
        if module in self._build_cache:
            return self._build_cache[module]

        module_category = module.get_module_category()

        # Handle SOURCE modules (oscillators, envelopes, LFOs)
        if module_category == ModuleCategory.SOURCE:
            component = module.create_component(
                input_components=None, modulation_components=None
            )
            self._build_cache[module] = component
            return component

        # Handle OUTPUT module
        if module_category == ModuleCategory.OUTPUT:
            # Get the input connection
            input_port = self._find_port_by_name(module, "In")
            if not input_port:
                logger.warning("Output module has no In port")
                return None

            input_conn = self._find_connection_to_port(input_port)
            if not input_conn:
                logger.warning("Output module has no input connection")
                return None

            # Build the input component
            source_module = input_conn.parent_module
            component = self._build_chain_from_module(source_module)
            self._build_cache[module] = component
            return component

        # Handle MIXER modules (combine multiple inputs)
        if module_category == ModuleCategory.MIXER:
            # Collect all input components
            input_components = []
            for input_port in getattr(module, "input_ports", []):
                input_conn = self._find_connection_to_port(input_port)
                if input_conn:
                    input_module = input_conn.parent_module
                    input_component = self._build_chain_from_module(input_module)
                    if input_component:
                        input_components.append(input_component)

            if not input_components:
                logger.warning(
                    f"Mixer module '{module.module_title}' has no input connections"
                )
                return None

            # Create the mixer component
            component = module.create_component(
                input_components=input_components, modulation_components=None
            )
            self._build_cache[module] = component
            return component

        # Handle MODIFIER modules (volume, panner, clipper, etc.)
        if module_category == ModuleCategory.MODIFIER:
            # Get required input(s)
            required_inputs = module.get_required_inputs()
            if not required_inputs:
                logger.warning(
                    f"Modifier module '{module.module_title}' has no required inputs defined"
                )
                return None

            # Get the main input component
            main_input_name = required_inputs[0]
            main_input_port = self._find_port_by_name(module, main_input_name)
            if not main_input_port:
                logger.warning(
                    f"Modifier module '{module.module_title}' missing port '{main_input_name}'"
                )
                return None

            input_conn = self._find_connection_to_port(main_input_port)
            if not input_conn:
                logger.warning(
                    f"Modifier module '{module.module_title}' has no input connection"
                )
                return None

            # Build the input component
            input_module = input_conn.parent_module
            input_component = self._build_chain_from_module(input_module)
            if not input_component:
                return None

            # Collect modulation components
            modulation_components = {}
            for mod_port_name in module.get_modulation_inputs():
                mod_port = self._find_port_by_name(module, mod_port_name)
                if mod_port:
                    mod_conn = self._find_connection_to_port(mod_port)
                    if mod_conn:
                        mod_module = mod_conn.parent_module
                        mod_component = self._build_chain_from_module(mod_module)
                        if mod_component:
                            modulation_components[mod_port_name] = mod_component

            # Create the modifier component
            modifier_component = module.create_component(
                input_components=[input_component],
                modulation_components=(
                    modulation_components if modulation_components else None
                ),
            )

            # Chain input with modifier
            if modifier_component:
                component = Chain(input_component, modifier_component)
                self._build_cache[module] = component
                return component

        # Unknown module type
        logger.warning(f"Unknown module type: {module_category}")
        return None

    def _find_port_by_name(
        self, module: AudioModuleInterface, port_name: str
    ) -> Port | None:
        """Find a port by name in a module.

        Args:
            module: The module to search
            port_name: The name of the port to find

        Returns:
            The port or None if not found
        """
        for port in getattr(module, "input_ports", []):
            if port.port_name == port_name:
                return port
        for port in getattr(module, "output_ports", []):
            if port.port_name == port_name:
                return port
        return None

    def _find_connection_to_port(self, port: Port) -> Port | None:
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

    def get_compilation_errors(self) -> list[str]:
        """Get a list of compilation errors/warnings.

        Returns:
            List of error messages
        """
        errors = []

        # Check for output module
        has_output = any(
            m.get_module_category() == ModuleCategory.OUTPUT for m in self.modules
        )
        if not has_output:
            errors.append("No output module in patch")

        # Validate each module's connections
        for module in self.modules:
            module_errors = module.validate_connections(self.connections)
            errors.extend(module_errors)

        # Check for disconnected modules
        connected_modules = set()
        for start_port, end_port in self.connections:
            connected_modules.add(start_port.parent_module)
            connected_modules.add(end_port.parent_module)

        for module in self.modules:
            if (
                module not in connected_modules
                and module.get_module_category() != ModuleCategory.OUTPUT
            ):
                errors.append(
                    f"Module '{getattr(module, 'module_title', 'Unknown')}' is not connected"
                )

        # Check for invalid connections
        for start_port, end_port in self.connections:
            if start_port.port_type != "output":
                errors.append(f"Invalid connection: source port is not an output")
            if end_port.port_type != "input":
                errors.append(f"Invalid connection: destination port is not an input")

        return errors
