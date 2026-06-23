"""Validation helpers for the patch compiler."""

from __future__ import annotations

from typing import Any

from src.gui.core.module import AudioModule, ModuleCategory
from src.gui.widgets.port_widget import PortWidget


class PatchCompilerValidationMixin:
    """Cycle detection and patch prevalidation."""

    modules: list[AudioModule]
    connections: list[tuple[PortWidget, PortWidget]]

    def _detect_cycles(self) -> list[str]:
        """Detect cycles in the patch graph."""
        errors: list[str] = []
        graph: dict[Any, list[Any]] = {}
        connections = self.connections

        for start_port, end_port in connections:
            source_module = start_port.parent_module
            dest_module = end_port.parent_module
            graph.setdefault(source_module, []).append(dest_module)

        visited = set()
        rec_stack = set()

        def dfs(module, path):
            visited.add(module)
            rec_stack.add(module)
            path.append(module)

            for neighbor in graph.get(module, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    try:
                        cycle_start_idx = path.index(neighbor)
                        cycle_modules = path[cycle_start_idx:] + [neighbor]
                        cycle_names = [m.metadata.title for m in cycle_modules]
                        errors.append(
                            f"Infinite loop detected: {' → '.join(cycle_names)}\n"
                            f"This creates a feedback loop that cannot be compiled."
                        )
                    except ValueError:
                        errors.append(
                            f"Infinite loop detected involving module "
                            f"'{neighbor.metadata.title}'\n"
                            f"This creates a feedback loop that cannot be compiled."
                        )
                    return True

            path.pop()
            rec_stack.remove(module)
            return False

        for module in graph:
            if module not in visited:
                dfs(module, [])

        return errors

    def get_prevalidation_errors(self) -> list[str]:
        """Validate the current patch before compilation."""
        errors = []
        errors.extend(self._detect_cycles())

        modules = self.modules
        connections = self.connections
        has_output = any(
            module.metadata.category == ModuleCategory.OUTPUT for module in modules
        )
        if not has_output:
            errors.append("No output module in patch.")

        for start_port, end_port in connections:
            if start_port.port_type != "output":
                errors.append("Invalid connection: source port is not an output")
            if end_port.port_type != "input":
                errors.append("Invalid connection: destination port is not an input")

            if start_port.parent_module == end_port.parent_module:
                module_name = getattr(start_port.parent_module, "metadata", None)
                module_name = module_name.title if module_name else "Unknown"
                errors.append(
                    f"Invalid self-connection in module '{module_name}': "
                    f"output cannot connect to own input"
                )

        return errors
