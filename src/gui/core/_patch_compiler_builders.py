"""Category-specific build logic for the patch compiler."""

from __future__ import annotations

import logging
from typing import Any, Callable

from src.engine.audio_component import AudioComponent
from src.engine.composer import Chain, WaveAdder
from src.gui.core.module import AudioModule, ModuleCategory

logger = logging.getLogger(__name__)


class PatchCompilerBuilderMixin:
    """Build compiled components for modules by category."""

    def _build_chain_from_module(self, module: AudioModule) -> AudioComponent | None:
        """Build an audio chain by dispatching to a category-specific builder."""
        build_cache: dict[AudioModule, Any] = self._build_cache
        if module in build_cache:
            return build_cache[module]

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

        if component and category != ModuleCategory.MODIFIER:
            self._cache_component(module, component)
        elif component and category == ModuleCategory.MODIFIER:
            build_cache[module] = component

        return component

    @staticmethod
    def _build_source_module(module: AudioModule) -> AudioComponent | None:
        """Build a SOURCE module."""
        return module.create_engine_component(
            input_components=None, modulation_components=None
        )

    def _build_modulated_source_module(
        self, module: AudioModule
    ) -> AudioComponent | None:
        """Build a MODULATED_SOURCE module."""
        input_components = []

        for input_name in module.get_required_inputs():
            input_port = self._find_port_by_name(module, input_name)
            if input_port:
                input_conn = self._find_connection_to_port(input_port)
                if input_conn:
                    component = self._build_component_from_port(input_conn)
                    if component:
                        input_components.append(component)

        required_set = set(module.get_required_inputs())
        modulation_set = set(module.get_modulation_inputs())

        for port in getattr(module, "input_ports", []):
            port_name = port.port_name
            if port_name in required_set or port_name in modulation_set:
                continue

            input_conn = self._find_connection_to_port(port)
            if input_conn:
                component = self._build_component_from_port(input_conn)
                if component:
                    input_components.append(component)

        modulation_components = self._collect_modulation_components(module)

        return module.create_engine_component(
            input_components=input_components if input_components else None,
            modulation_components=(
                modulation_components if modulation_components else None
            ),
        )

    def _build_output_module(self, module: AudioModule) -> AudioComponent | None:
        """Build an OUTPUT module."""
        name = module.metadata.title
        input_port = self._find_port_by_name(module, "In")
        if not input_port:
            logger.warning(f"{name}: No 'In' port found")
            return None

        components, skipped = self._collect_input_components_from_port(
            input_port, track_skipped=True
        )

        if skipped:
            logger.info(
                f"{name}: Skipped {len(skipped)} module(s) with missing inputs: "
                f"{', '.join(skipped)}"
            )

        if not components:
            logger.warning(f"{name}: No valid input components")
            return None

        return WaveAdder(*components) if len(components) > 1 else components[0]

    def _build_mixer_module(self, module: AudioModule) -> AudioComponent | None:
        """Build a MIXER module."""
        name = module.metadata.title
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

        return module.create_engine_component(
            input_components=input_components, modulation_components=None
        )

    def _build_modifier_module(self, module: AudioModule) -> AudioComponent | None:
        """Build a MODIFIER module."""
        name = module.metadata.title
        required_inputs = module.get_required_inputs()
        if not required_inputs:
            logger.debug(f"{name}: No required inputs defined - skipping")
            return None

        main_input_name = required_inputs[0]
        main_input_port = self._find_port_by_name(module, main_input_name)
        if not main_input_port:
            logger.debug(f"{name}: Missing port '{main_input_name}' - skipping")
            return None

        components, _ = self._collect_input_components_from_port(main_input_port)
        if not components:
            logger.debug(f"{name}: No input connection - skipping")
            return None

        input_comp = WaveAdder(*components) if len(components) > 1 else components[0]
        modulation_components = self._collect_modulation_components(module)

        logger.info(
            f"{name}: Calling create_engine_component with "
            f"input_components=[{input_comp}], "
            f"modulation_components={modulation_components}"
        )
        modifier_component = module.create_engine_component(
            input_components=[input_comp],
            modulation_components=(
                modulation_components if modulation_components else None
            ),
        )

        logger.info(
            f"{name}: create_engine_component returned: {modifier_component}, type="
            f"{type(modifier_component).__name__ if modifier_component else 'None'}"
        )

        if not modifier_component:
            logger.warning(
                f"{name}: create_engine_component returned None - skipping module"
            )
            return None

        component = Chain(input_comp, modifier_component)
        module_to_component: dict[AudioModule, Any] = self._module_to_component
        logger.info(
            f"Storing {name} modifier for hot-swapping: "
            f"component id={id(modifier_component)}, "
            f"type={type(modifier_component).__name__}"
        )
        module_to_component[module] = modifier_component
        return component
