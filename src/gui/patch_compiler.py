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
                if module.metadata.category == ModuleCategory.OUTPUT:
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

            logger.debug(f"Patch compiled successfully: {type(component).__name__}")
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

        module_category = module.metadata.category
        name = module.metadata.title

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
                    f"Mixer module '{name}' has no input connections."
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
                    f"Modifier module '{name}' has no required inputs defined."
                )
                return None

            # Get the main input component
            main_input_name = required_inputs[0]
            main_input_port = self._find_port_by_name(module, main_input_name)
            if not main_input_port:
                logger.warning(
                    f"Modifier module '{name}' missing port '{main_input_name}'."
                )
                return None

            input_conn = self._find_connection_to_port(main_input_port)
            if not input_conn:
                logger.warning(
                    f"Modifier module '{name}' has no input connection"
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

    def _detect_cycles(self) -> list[str]:
        """Detect cycles (infinite loops) in the patch connection graph.

        Returns:
            List of error messages describing any cycles found
        """
        errors = []

        # Build adjacency list for the connection graph
        graph = {}  # module -> list of modules it connects to
        for start_port, end_port in self.connections:
            source_module = start_port.parent_module
            dest_module = end_port.parent_module

            if source_module not in graph:
                graph[source_module] = []
            graph[source_module].append(dest_module)

        # Track visited modules and recursion stack for cycle detection
        visited = set()
        rec_stack = set()

        def dfs(module, path):
            """Depth-first search to detect cycles."""
            visited.add(module)
            rec_stack.add(module)
            path.append(module)

            # Check all neighbors
            for neighbor in graph.get(module, []):
                if neighbor not in visited:
                    # Continue DFS
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle! The neighbor is already in our recursion stack
                    # Build the cycle description from where neighbor first appears in path
                    try:
                        cycle_start_idx = path.index(neighbor)
                        cycle_modules = path[cycle_start_idx:] + [neighbor]
                        cycle_names = [m.metadata.title for m in cycle_modules]
                        errors.append(
                            f"Infinite loop detected: {' → '.join(cycle_names)}\n"
                            f"This creates a feedback loop that cannot be compiled."
                        )
                    except ValueError:
                        # Neighbor not in path (shouldn't happen, but be safe)
                        errors.append(
                            f"Infinite loop detected involving module '{neighbor.metadata.title}'\n"
                            f"This creates a feedback loop that cannot be compiled."
                        )
                    return True

            path.pop()
            rec_stack.remove(module)
            return False

        # Check each module as a potential cycle starting point
        for module in graph.keys():
            if module not in visited:
                dfs(module, [])

        return errors

    def _get_signal_path_modules(self, output_module: AudioModuleInterface) -> set[AudioModuleInterface]:
        """Get all modules that are part of the signal path to the output.

        Args:
            output_module: The output module to trace back from

        Returns:
            Set of modules in the signal path
        """
        signal_path = set()

        def trace_back(module):
            """Recursively trace back from a module to find all inputs."""
            if module in signal_path:
                return  # Already visited

            signal_path.add(module)

            # Find all input connections to this module
            for input_port in getattr(module, "input_ports", []):
                input_conn = self._find_connection_to_port(input_port)
                if input_conn:
                    source_module = input_conn.parent_module
                    trace_back(source_module)

        trace_back(output_module)
        return signal_path

    def get_compilation_errors(self) -> list[str]:
        """Get a list of compilation errors/warnings.

        Returns:
            List of error messages
        """
        errors = []

        # Check for cycles (infinite loops) first
        errors.extend(self._detect_cycles())

        # Check for output module
        has_output = any(
            m.metadata.category == ModuleCategory.OUTPUT for m in self.modules
        )
        if not has_output:
            errors.append("No output module in patch.")

        # Validate connections only for modules that are part of the signal path
        # Disconnected modules are allowed and will be ignored during compilation

        # Find all modules that are part of the signal path to output
        output_module = None
        for m in self.modules:
            if m.metadata.category == ModuleCategory.OUTPUT:
                output_module = m
                break

        if output_module:
            # Build set of modules in the signal path
            # Only validate modules that are in the signal path
            signal_path_modules = self._get_signal_path_modules(output_module)
        else:
            # No output module, so validate all modules (for completeness)
            signal_path_modules = self.modules

        for module in signal_path_modules:
            errors.extend(module.validate_connections(self.connections))

        # Note: Disconnected modules are intentionally NOT reported as errors
        # They remain on the canvas but are ignored during compilation

        # Check for invalid connections
        for start_port, end_port in self.connections:
            if start_port.port_type != "output":
                errors.append("Invalid connection: source port is not an output")
            if end_port.port_type != "input":
                errors.append("Invalid connection: destination port is not an input")

            # Check for self-connections
            if start_port.parent_module == end_port.parent_module:
                module_name = getattr(start_port.parent_module, 'metadata', None)
                if module_name:
                    module_name = module_name.title
                else:
                    module_name = "Unknown"
                errors.append(
                    f"Invalid self-connection in module '{module_name}': "
                    f"output cannot connect to own input"
                )

        return errors

    def build_patch_tree(self) -> dict[str, Any]:
        """Build a hierarchical tree representation of the patch.

        Returns:
            Dictionary representing the patch tree structure.
            If there's an output module, returns its tree.
            If no output, returns a multi-root structure with all disconnected modules.
        """
        if not self.modules:
            return {"error": "No modules in patch"}

        # Find output module as root
        output_module = None
        for module in self.modules:
            if module.metadata.category == ModuleCategory.OUTPUT:
                output_module = module
                break

        # If we have an output module, build tree from it
        if output_module:
            visited = set()
            return self._build_tree_node(output_module, visited)

        # No output module - build a multi-root tree showing all modules
        # First, find all modules that have no incoming connections (roots)
        connected_as_input = set()
        for start_port, end_port in self.connections:
            connected_as_input.add(end_port.parent_module)

        root_modules = [m for m in self.modules if m not in connected_as_input]

        # If no clear roots, just show all modules
        if not root_modules:
            root_modules = self.modules

        # Build a virtual root containing all roots
        visited = set()
        return {
            "name": "Patch (no output)",
            "type": "ROOT",
            "id": 0,
            "inputs": [
                {"port_name": "Module", "node": self._build_tree_node(module, visited)}
                for module in root_modules
            ],
            "modulations": [],
        }

    def _build_tree_node(
        self, module: AudioModuleInterface, visited: set
    ) -> dict[str, Any]:
        """Build a tree node for a module recursively.

        Args:
            module: The module to build node for
            visited: Set of already visited modules to prevent cycles

        Returns:
            Dictionary representing the tree node
        """
        module_id = id(module)

        node = {
            "name": module.metadata.title,
            "type": module.metadata.category.value,
            "id": module_id,
            "inputs": [],
            "modulations": [],
        }

        # Prevent infinite recursion
        if module_id in visited:
            node["cycle"] = True
            return node

        visited.add(module_id)

        # Get modulation input names
        mod_inputs = set(module.get_modulation_inputs())

        # Get input connections
        for input_port in getattr(module, "input_ports", []):
            input_conn = self._find_connection_to_port(input_port)
            if input_conn:
                source_module = input_conn.parent_module
                port_info = {
                    "port_name": input_port.port_name,
                    "node": self._build_tree_node(source_module, visited.copy()),
                }

                # Classify as modulation or main input
                if input_port.port_name in mod_inputs:
                    node["modulations"].append(port_info)
                else:
                    node["inputs"].append(port_info)

        return node
