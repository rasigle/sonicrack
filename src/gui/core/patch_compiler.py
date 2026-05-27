"""Patch compiler - converts visual patches to audio components."""

from __future__ import annotations

import logging
from typing import Any

from src.engine.audio_component import AudioComponent
from src.gui.core._patch_compiler_builders import PatchCompilerBuilderMixin
from src.gui.core._patch_compiler_graph import PatchCompilerGraphMixin
from src.gui.core._patch_compiler_tree import PatchCompilerTreeMixin
from src.gui.core._patch_compiler_validation import PatchCompilerValidationMixin
from src.gui.core.module import AudioModule, ModuleCategory
from src.gui.widgets.port_widget import PortWidget

logger = logging.getLogger(__name__)


class PatchCompiler(
    PatchCompilerBuilderMixin,
    PatchCompilerTreeMixin,
    PatchCompilerValidationMixin,
    PatchCompilerGraphMixin,
):
    """Compile visual patches into executable audio component graphs.

    The public API stays here, while graph traversal, category-specific builders,
    validation, and tree rendering live in focused helper mixins.
    """

    def __init__(self):
        """Initialize the patch compiler."""
        self.modules: list[AudioModule] = []
        self.connections: list[tuple[PortWidget, PortWidget]] = []
        self.compiled_patch: AudioComponent | None = None
        self._build_cache: dict[AudioModule, Any] = {}
        self._module_to_component: dict[AudioModule, AudioComponent] = {}

    def set_patch(
        self,
        modules: list[AudioModule],
        connections: list[tuple[PortWidget, PortWidget]],
    ):
        """Set the patch to compile."""
        self.modules = modules
        self.connections = connections
        self._build_cache = {}
        self._module_to_component = {}

        logger.info(f"Patch set with {len(connections)} connections:")
        for start_port, end_port in connections:
            logger.info(
                f"  {start_port.parent_module.metadata.title}.{start_port.port_name} "
                f"-> {end_port.parent_module.metadata.title}.{end_port.port_name}"
            )

    def compile(self) -> AudioComponent | None:
        """Compile the patch into an audio component."""
        self.compiled_patch = None
        try:
            output_module = next(
                (
                    module
                    for module in self.modules
                    if module.metadata.category == ModuleCategory.OUTPUT
                ),
                None,
            )

            if not output_module:
                logger.error("No output module found in patch")
                return None

            component = self._build_chain_from_module(output_module)
            if component is None:
                logger.error("Failed to build signal chain")
                return None

            logger.debug(f"Patch compiled successfully: {type(component).__name__}")
            self.compiled_patch = component
            return component

        except Exception as exc:
            logger.error(f"Patch compilation failed: {exc}", exc_info=True)
            return None

    def update_parameter(
        self, module: AudioModule, param_name: str, value: Any
    ) -> bool:
        """Hot-swap a parameter value without recompiling."""
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

        try:
            setattr(component, param_name, value)
            return True
        except (AttributeError, TypeError, ValueError) as exc:
            logger.error(
                f"Failed to hot-swap {param_name} in {source_module_name}: {exc}",
                exc_info=True,
            )
            return False
