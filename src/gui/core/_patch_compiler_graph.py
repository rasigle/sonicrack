"""Graph and connection helpers for the patch compiler."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from src.engine import CVScaler
from src.gui.core.module import AudioModule, ModuleCategory
from src.gui.widgets.port_widget import PortWidget

logger = logging.getLogger(__name__)


class PatchCompilerGraphMixin:
    """Helpers for traversing module connections and collecting components."""

    connections: list[tuple[PortWidget, PortWidget]]
    _build_cache: dict[AudioModule, Any]
    _module_to_component: dict[AudioModule, Any]

    if TYPE_CHECKING:

        def _build_chain_from_module(self, module: AudioModule) -> Any | None: ...

    def _cache_component(self, module: AudioModule, component: Any) -> None:
        """Cache a compiled component and track it for hot-swapping."""
        build_cache: dict[AudioModule, Any] = self._build_cache
        module_to_component: dict[AudioModule, Any] = self._module_to_component
        build_cache[module] = component
        module_to_component[module] = component

    def _build_component_from_port(self, connection: PortWidget) -> Any | None:
        """Build a component from a connection's source port."""
        source_module = connection.parent_module
        source_port_name = connection.port_name

        logger.info(
            f"_build_component_from_port: source_module={source_module.metadata.title}"
            f", port_name='{source_port_name}'"
        )

        if (
            not getattr(source_module, "is_active", True)
            and source_module.metadata.category != ModuleCategory.MODIFIER
        ):
            logger.info(
                f"_build_component_from_port: {source_module.metadata.title} "
                "is inactive, skipping"
            )
            return None

        if hasattr(source_module, "get_output_component"):
            logger.info(
                f"_build_component_from_port: {source_module.metadata.title} has "
                f"get_output_component, calling it"
            )
            component = source_module.get_output_component(source_port_name)
            logger.info(
                f"_build_component_from_port: get_output_component returned {component}"
            )

            if component is not None:
                return component

            logger.info(
                "_build_component_from_port: get_output_component returned None, "
                "falling back to _build_chain_from_module"
            )
            return self._build_chain_from_module(source_module)

        logger.info(
            f"_build_component_from_port: {source_module.metadata.title} does not "
            f"have get_output_component, calling _build_chain_from_module"
        )
        return self._build_chain_from_module(source_module)

    def _collect_input_components_from_port(
        self, port: PortWidget, track_skipped: bool = False
    ) -> tuple[list[Any], list[str]]:
        """Collect all components connected to an input port."""
        all_connections = self._find_all_connections_to_port(port)
        components: list[Any] = []
        skipped: list[str] = []

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
        """Insert a CV scaler when source and target CV ranges differ."""
        source_range = None
        target_range = None

        if hasattr(target_module, "get_cv_range"):
            target_range = target_module.get_cv_range(port_name)
            logger.debug(
                f"  Target {target_module.metadata.title}.{port_name} expects CV "
                f"range: {target_range}"
            )

        if hasattr(source_module, "get_cv_output_range"):
            source_range = source_module.get_cv_output_range()
            logger.debug(
                f"  Source {source_module.metadata.title} outputs CV range: "
                f"{source_range}"
            )

        if source_range and target_range and source_range != target_range:
            source_cv_range = cast(tuple[float, float], source_range)
            target_cv_range = cast(tuple[float, float], target_range)
            logger.info(
                f"CV range mismatch detected: "
                f"{source_module.metadata.title} outputs {source_cv_range}, "
                f"but {target_module.metadata.title}.{port_name} expects "
                f"{target_cv_range}"
            )
            logger.info(
                f"Auto-inserting CVScaler: {source_cv_range} → {target_cv_range}"
            )

            scaler = CVScaler(
                mod_component,
                input_range=source_cv_range,
                output_range=target_cv_range,
                clamp=True,
            )

            logger.debug(
                f"  CVScaler created (id={id(scaler)}): "
                f"scale={scaler._scale:.3f}, offset={scaler._offset:.3f}"
            )
            return scaler

        if source_range and target_range:
            logger.debug(f"  CV ranges match {source_range}, no scaling needed")

        return mod_component

    def _collect_modulation_components(self, module: AudioModule) -> dict[str, Any]:
        """Collect modulation components for a module."""
        modulation_components: dict[str, Any] = {}

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
                f"  Found connection from: "
                f"{source_module.metadata.title if source_module else 'Unknown'}"
            )

            mod_component = self._build_component_from_port(mod_conn)
            if not mod_component:
                continue

            mod_component = self._insert_cv_scaler_if_needed(
                mod_component, source_module, module, mod_port_name
            )
            modulation_components[mod_port_name] = mod_component
            logger.debug(
                f"  Collected modulation '{mod_port_name}': "
                f"{type(mod_component).__name__} (id={id(mod_component)})"
            )

        return modulation_components

    @staticmethod
    def _find_port_by_name(module: AudioModule, port_name: str) -> PortWidget | None:
        """Find a named port in a module's inputs or outputs."""
        for port in getattr(module, "input_ports", []):
            if port.port_name == port_name:
                return port
        for port in getattr(module, "output_ports", []):
            if port.port_name == port_name:
                return port
        return None

    def _find_connection_to_port(self, port: PortWidget) -> PortWidget | None:
        """Find the first source port connected to an input port."""
        connections: list[tuple[PortWidget, PortWidget]] = self.connections
        for start_port, end_port in connections:
            if end_port == port:
                return start_port
        return None

    def _find_all_connections_to_port(self, port: PortWidget) -> list[PortWidget]:
        """Find all source ports connected to an input port."""
        compiler_connections: list[tuple[PortWidget, PortWidget]] = self.connections
        connections: list[PortWidget] = []
        for start_port, end_port in compiler_connections:
            if end_port == port:
                connections.append(start_port)
        return connections
