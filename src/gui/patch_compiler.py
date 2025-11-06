"""Patch compiler - converts visual patch to audio components."""

import logging
from typing import Any, Callable

from src.engine import CVScaler
from src.engine.audio_component import AudioComponent
from src.engine.composer import Chain, WaveAdder
from src.gui.audio_module_interface import ModuleCategory, AudioModule
from src.gui.patch_canvas import Port

logger = logging.getLogger(__name__)


class PatchCompiler:
    """Compiles visual patches into executable audio component graphs.

    Takes the modules and connections from the patch canvas and builds
    a working audio signal chain using the generic AudioModule interface.

    This compiler is extensible - it works with any module that implements
    the AudioModule interface.
    """

    def __init__(self):
        """Initialize the patch compiler."""
        self.modules: list[AudioModule] = []
        self.connections: list[tuple[Port, Port]] = []

        self.compiled_patch: AudioComponent | None = None
        self._build_cache: dict[AudioModule, Any] = {}

        # Hot-swapping support: track module → component mapping
        self._module_to_component: dict[AudioModule, AudioComponent] = {}

    def set_patch(
        self, modules: list[AudioModule], connections: list[tuple[Port, Port]]
    ):
        """Set the patch to compile.

        Args:
            modules: List of module widgets implementing AudioModuleInterface
            connections: List of (output_port, input_port) tuples
        """
        self.modules = modules
        self.connections = connections
        self._build_cache = {}
        self._module_to_component = {}

        # Log all connections for debugging
        logger.info(f"Patch set with {len(connections)} connections:")
        for start_port, end_port in connections:
            logger.info(
                f"  {start_port.parent_module.metadata.title}.{start_port.port_name} → {end_port.parent_module.metadata.title}.{end_port.port_name}"
            )

    def compile(self) -> AudioComponent | None:
        """Compile the patch into an audio component.

        Returns:
            The compiled audio component or None if compilation fails
        """
        self.compiled_patch = None
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

    def update_parameter(
        self, module: AudioModule, param_name: str, value: Any
    ) -> bool:
        """Hot-swap a parameter value without recompiling (eliminates clicks).

        This method updates parameters directly in the compiled audio components,
        avoiding the need to recreate the entire patch. This prevents phase
        discontinuities and eliminates clicking when adjusting parameters.

        Args:
            module: The module widget whose parameter changed
            param_name: Name of the parameter (e.g., "frequency", "gain_db")
            value: New parameter value

        Returns:
            True if parameter was updated successfully, False otherwise

        Example:
            >>> # User rotates frequency knob
            >>> compiler.update_parameter(osc_module, "frequency", 880)  # noqa
            >>> # Frequency changes instantly without click!
        """
        source_module_name = module.metadata.title
        if module not in self._module_to_component:
            logger.warning(
                f"Cannot hot-swap parameter: module {source_module_name} "
                f"not found in compiled patch."
            )
            return False

        component = self._module_to_component[module]
        if not hasattr(component, param_name):
            logger.warning(
                f"Engine component {type(component).__name__} does not have "
                f"configuration parameter '{param_name}'."
            )
            return False

        # Try to set parameter directly on engine component
        try:
            setattr(component, param_name, value)
            return True
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(
                f"Failed to hot-swap {param_name} in {source_module_name}: {e}",
                exc_info=True,
            )
            return False

    def _build_chain_from_module(self, module: AudioModule) -> AudioComponent | None:
        """Build the audio chain from a module by following connections backwards.

        Dispatches to category-specific builders for clean, focused logic.

        Args:
            module: The module to build from

        Returns:
            The audio component or None
        """
        # Check cache to avoid rebuilding the same module multiple times
        if module in self._build_cache:
            return self._build_cache[module]

        # Dispatch to category-specific builder
        category: ModuleCategory = module.metadata.category
        builders: dict[ModuleCategory, Callable[[Any], AudioComponent | None]] = {
            ModuleCategory.SOURCE: self._build_source_module,
            ModuleCategory.MODULATED_SOURCE: self._build_modulated_source_module,
            ModuleCategory.OUTPUT: self._build_output_module,
            ModuleCategory.MIXER: self._build_mixer_module,
            ModuleCategory.MODIFIER: self._build_modifier_module,
        }

        builder = builders.get(category)
        if not builder:
            raise ValueError(f"Unknown module category: {category}")

        component = builder(module)

        # Cache component (but NOT for modifiers - they cache internally to avoid Chain wrapper issues)
        if component and category != ModuleCategory.MODIFIER:
            self._cache_component(module, component)
        elif component and category == ModuleCategory.MODIFIER:
            # For modifiers, only cache in build_cache (not module_to_component - that's handled in _build_modifier_module)
            self._build_cache[module] = component

        return component

    # === Helper Methods ===

    def _cache_component(self, module: AudioModule, component: AudioComponent):
        """Cache component and track for hot-swapping.

        Args:
            module: The module widget
            component: The compiled audio component
        """
        self._build_cache[module] = component
        self._module_to_component[module] = component

    def _build_component_from_port(self, connection: Port) -> Any | None:
        """Build component from a connection's source module.

        Args:
            connection: The source (output) port

        Returns:
            Built component or None
        """
        source_module = connection.parent_module
        source_port_name = connection.port_name

        # Check if source module has multiple outputs
        if hasattr(source_module, "get_output_component"):
            return source_module.get_output_component(source_port_name)
        else:
            return self._build_chain_from_module(source_module)

    def _collect_input_components_from_port(
        self, port: Port, track_skipped: bool = False
    ) -> tuple[list, list[str]]:
        """Collect all components connected to a port.

        Args:
            port: The input port to collect from
            track_skipped: Whether to track skipped module names

        Returns:
            Tuple of (components list, skipped module names list)
        """
        all_connections = self._find_all_connections_to_port(port)
        components = []
        skipped = []

        for conn in all_connections:
            component = self._build_component_from_port(conn)
            if component:
                components.append(component)
            elif track_skipped:
                skipped.append(conn.parent_module.metadata.title)

        return components, skipped

    def _insert_cv_scaler_if_needed(
        self,
        mod_component: Any,
        source_module: AudioModule,
        target_module: AudioModule,
        port_name: str,
    ) -> Any:
        """Insert CV scaler if source and target ranges don't match.

        Automatically detects CV range mismatches and inserts a CVScaler
        to convert the signal to the expected range.

        Args:
            mod_component: The modulation component to potentially wrap
            source_module: The source module providing CV
            target_module: The target module receiving CV
            port_name: Name of the modulation port

        Returns:
            Either the original component (if ranges match) or a CVScaler
            wrapping the component (if ranges don't match)

        Example:
            LFO [-1, 1] → Volume expects [0, 1]
            Returns: CVScaler(lfo_component, (-1, 1), (0, 1))
        """

        source_range = None
        target_range = None

        # Get target's expected CV range
        if hasattr(target_module, "get_cv_range"):
            target_range = target_module.get_cv_range(port_name)
            logger.debug(
                f"  Target {target_module.metadata.title}.{port_name} expects CV range: {target_range}"
            )

        # Get source's output range
        if hasattr(source_module, "get_cv_output_range"):
            source_range = source_module.get_cv_output_range()
            logger.debug(
                f"  Source {source_module.metadata.title} outputs CV range: {source_range}"
            )

        # Check if we need to insert a scaler
        if source_range and target_range and source_range != target_range:
            logger.info(
                f"CV range mismatch detected: "
                f"{source_module.metadata.title} outputs {source_range}, "
                f"but {target_module.metadata.title}.{port_name} expects {target_range}"
            )
            logger.info(f"Auto-inserting CVScaler: {source_range} → {target_range}")

            # Create scaler with clamping enabled
            scaler = CVScaler(
                mod_component,
                input_range=source_range,
                output_range=target_range,
                clamp=True,
            )

            logger.debug(
                f"  CVScaler created (id={id(scaler)}): "
                f"scale={scaler._scale:.3f}, offset={scaler._offset:.3f}"
            )

            return scaler

        # No scaling needed
        if source_range and target_range:
            logger.debug(f"  CV ranges match {source_range}, no scaling needed")

        return mod_component

    def _collect_modulation_components(self, module: AudioModule) -> dict:
        """Collect modulation components as dictionary.

        Automatically inserts CV scalers when the source output range doesn't
        match the target input range.

        Args:
            module: The module to collect modulations for

        Returns:
            Dictionary mapping port_name -> component (possibly wrapped in CVScaler)
        """
        modulation_components = {}

        for mod_port_name in module.get_modulation_inputs():
            mod_port = self._find_port_by_name(module, mod_port_name)
            logger.debug(f"Checking modulation port '{mod_port_name}'")

            if not mod_port:
                continue

            mod_conn = self._find_connection_to_port(mod_port)
            if not mod_conn:
                logger.debug(f"  No connection found for port '{mod_port_name}'")
                continue

            source_module = mod_conn.parent_module
            logger.debug(
                f"  Found connection from: {source_module.metadata.title if source_module else 'Unknown'}"
            )

            # Build the modulation component
            mod_component = self._build_component_from_port(mod_conn)
            if not mod_component:
                continue

            # Insert CV scaler if ranges don't match
            mod_component = self._insert_cv_scaler_if_needed(
                mod_component, source_module, module, mod_port_name
            )

            modulation_components[mod_port_name] = mod_component
            logger.debug(
                f"  Collected modulation '{mod_port_name}': "
                f"{type(mod_component).__name__} (id={id(mod_component)})"
            )

        return modulation_components

    # === Category-Specific Builders ===
    @staticmethod
    def _build_source_module(module: AudioModule) -> AudioComponent | None:
        """Build SOURCE module (oscillators, LFOs, envelopes without CV inputs).

        Args:
            module: The SOURCE module

        Returns:
            The audio component
        """
        return module.create_engine_component(
            input_components=None, modulation_components=None
        )

    def _build_modulated_source_module(
        self, module: AudioModule
    ) -> AudioComponent | None:
        """Build MODULATED_SOURCE module (VCO, ADSR with CV inputs).

        Args:
            module: The MODULATED_SOURCE module

        Returns:
            The audio component
        """
        # Collect CV input components (frequency, gate, etc.) from required inputs
        input_components = []

        for input_name in module.get_required_inputs():
            input_port = self._find_port_by_name(module, input_name)
            if input_port:
                input_conn = self._find_connection_to_port(input_port)
                if input_conn:
                    component = self._build_component_from_port(input_conn)
                    if component:
                        input_components.append(component)

        # Also collect from ALL other input ports (like VCO's Freq port)
        # that aren't required or modulation ports
        required_set = set(module.get_required_inputs())
        modulation_set = set(module.get_modulation_inputs()) if hasattr(module, 'get_modulation_inputs') else set()

        for port in getattr(module, "input_ports", []):
            port_name = port.port_name
            # Skip if already handled as required or modulation
            if port_name in required_set or port_name in modulation_set:
                continue

            # Collect this input
            input_conn = self._find_connection_to_port(port)
            if input_conn:
                component = self._build_component_from_port(input_conn)
                if component:
                    input_components.append(component)

        # Collect modulation components (like VCO's Gain port)
        modulation_components = self._collect_modulation_components(module) if hasattr(module, 'get_modulation_inputs') else None

        # Create component with CV inputs and modulation
        return module.create_engine_component(
            input_components=input_components if input_components else None,
            modulation_components=modulation_components if modulation_components else None,
        )

    def _build_output_module(self, module: AudioModule) -> AudioComponent | None:
        """Build OUTPUT module (terminal node, may mix multiple inputs).

        Args:
            module: The OUTPUT module

        Returns:
            The audio component (possibly WaveAdder if multiple inputs)
        """
        name = module.metadata.title

        # Find the input port
        input_port = self._find_port_by_name(module, "In")
        if not input_port:
            logger.warning(f"{name}: No 'In' port found")
            return None

        # Collect all input components (may be multiple for mixing)
        components, skipped = self._collect_input_components_from_port(
            input_port, track_skipped=True
        )

        # Log skipped modules
        if skipped:
            logger.info(
                f"{name}: Skipped {len(skipped)} module(s) with missing inputs: "
                f"{', '.join(skipped)}"
            )

        if not components:
            logger.warning(f"{name}: No valid input components")
            return None

        # Mix multiple inputs or return single input
        return WaveAdder(*components) if len(components) > 1 else components[0]

    def _build_mixer_module(self, module: AudioModule) -> AudioComponent | None:
        """Build MIXER module (combines multiple inputs).

        Args:
            module: The MIXER module

        Returns:
            The mixer component
        """
        name = module.metadata.title

        # Collect all input components
        input_components = []
        for input_port in getattr(module, "input_ports", []):
            input_conn = self._find_connection_to_port(input_port)
            if input_conn:
                component = self._build_component_from_port(input_conn)
                if component:
                    input_components.append(component)

        if not input_components:
            logger.debug(f"{name}: No input connections - skipping")
            return None

        # Create the mixer component
        return module.create_engine_component(
            input_components=input_components, modulation_components=None
        )

    def _build_modifier_module(self, module: AudioModule) -> AudioComponent | None:
        """Build MODIFIER module (volume, panner, effects).

        Args:
            module: The MODIFIER module

        Returns:
            Chain of input -> modifier, or None if no input
        """
        name = module.metadata.title

        # Get required inputs
        required_inputs = module.get_required_inputs()
        if not required_inputs:
            logger.debug(f"{name}: No required inputs defined - skipping")
            return None

        # Get the main input port
        main_input_name = required_inputs[0]
        main_input_port = self._find_port_by_name(module, main_input_name)
        if not main_input_port:
            logger.debug(f"{name}: Missing port '{main_input_name}' - skipping")
            return None

        # Collect input components (may be multiple, will be mixed)
        components, _ = self._collect_input_components_from_port(main_input_port)

        if not components:
            logger.debug(f"{name}: No input connection - skipping")
            return None

        # Mix multiple inputs or use single input
        input_comp = WaveAdder(*components) if len(components) > 1 else components[0]

        # Collect modulation components
        modulation_components = self._collect_modulation_components(module)

        # Create the modifier component
        modifier_component = module.create_engine_component(
            input_components=[input_comp],
            modulation_components=(
                modulation_components if modulation_components else None
            ),
        )

        if not modifier_component:
            return None

        # Chain input with modifier
        component = Chain(input_comp, modifier_component)

        # Track modifier for hot-swapping (not the Chain wrapper)
        logger.info(
            f"Storing {name} modifier for hot-swapping: "
            f"component id={id(modifier_component)}, type={type(modifier_component).__name__}"
        )
        self._module_to_component[module] = modifier_component

        return component

    @staticmethod
    def _find_port_by_name(module: AudioModule, port_name: str) -> Port | None:
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

    def _find_all_connections_to_port(self, port: Port) -> list[Port]:
        """Find ALL source ports connected to the given input port.

        This is important when multiple outputs connect to a single input -
        all signals should be mixed together.

        Args:
            port: The input port to find connections for

        Returns:
            List of source (output) ports connected to this input
        """
        connections = []
        for start_port, end_port in self.connections:
            if end_port == port:
                connections.append(start_port)
        return connections

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
                    # Build the cycle description from where neighbor first appears
                    # in path
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
                            f"Infinite loop detected involving module "
                            f"'{neighbor.metadata.title}'\n"
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

    def _get_signal_path_modules(self, output_module: AudioModule) -> set[AudioModule]:
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

    def get_prevalidation_errors(self) -> list[str]:
        """Check the consistency of the patch (cycles, no output, ...) and returns a
        list of errors/warnings.

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
                module_name = getattr(start_port.parent_module, "metadata", None)
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

    def _build_tree_node(self, module: AudioModule, visited: set) -> dict[str, Any]:
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
