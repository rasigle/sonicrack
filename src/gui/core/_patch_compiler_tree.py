"""Tree rendering helpers for the patch compiler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.gui.core.module import AudioModule, ModuleCategory
from src.gui.widgets.port_widget import PortWidget


class PatchCompilerTreeMixin:
    """Build hierarchical tree representations of patches."""

    modules: list[AudioModule]
    connections: list[tuple[PortWidget, PortWidget]]

    if TYPE_CHECKING:

        def _find_connection_to_port(self, port: PortWidget) -> PortWidget | None: ...

    def _get_signal_path_modules(self, output_module: AudioModule) -> set[AudioModule]:
        """Collect all modules that contribute to the output signal path."""
        signal_path: set[AudioModule] = set()

        def trace_back(module):
            if module in signal_path:
                return

            signal_path.add(module)

            for input_port in getattr(module, "input_ports", []):
                input_conn = self._find_connection_to_port(input_port)
                if input_conn:
                    trace_back(input_conn.parent_module)

        trace_back(output_module)
        return signal_path

    def build_patch_tree(self) -> dict[str, Any]:
        """Build a hierarchical tree representation of the current patch."""
        modules = self.modules
        connections = self.connections

        if not modules:
            return {"error": "No modules in patch"}

        output_module: AudioModule | None = None
        for module in modules:
            if module.metadata.category == ModuleCategory.OUTPUT:
                output_module = module
                break

        if output_module is not None:
            output_visited: set[int] = set()
            return self._build_tree_node(output_module, output_visited)

        connected_as_input = set()
        for _start_port, end_port in connections:
            connected_as_input.add(end_port.parent_module)

        root_modules = [
            module for module in modules if module not in connected_as_input
        ]
        if not root_modules:
            root_modules = modules

        visited: set[int] = set()
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
        self, module: AudioModule, visited: set[int]
    ) -> dict[str, Any]:
        """Build a recursive tree node for a module."""
        module_id = id(module)
        node: dict[str, Any] = {
            "name": module.metadata.title,
            "type": module.metadata.category.value,
            "id": module_id,
            "inputs": [],
            "modulations": [],
        }

        if module_id in visited:
            node["cycle"] = True
            return node

        visited.add(module_id)
        mod_inputs = set(module.get_modulation_inputs())

        for input_port in getattr(module, "input_ports", []):
            input_conn = self._find_connection_to_port(input_port)
            if input_conn:
                source_module = input_conn.parent_module
                port_info = {
                    "port_name": input_port.port_name,
                    "node": self._build_tree_node(source_module, visited.copy()),
                }

                if input_port.port_name in mod_inputs:
                    node["modulations"].append(port_info)
                else:
                    node["inputs"].append(port_info)

        return node

    def get_tree_structure(self) -> dict[str, Any]:
        """Compatibility alias for build_patch_tree()."""
        return self.build_patch_tree()
