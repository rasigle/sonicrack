"""Typing protocols shared by patch compiler helper mixins."""

from __future__ import annotations

from typing import Any, Protocol

from src.engine.audio_component import AudioComponent
from src.gui.core.module import AudioModule
from src.gui.widgets.port_widget import PortWidget


class PatchCompilerProtocol(Protocol):
    """Structural interface implemented by the main PatchCompiler class."""

    modules: list[AudioModule]
    connections: list[tuple[PortWidget, PortWidget]]
    compiled_patch: AudioComponent | None
    _build_cache: dict[AudioModule, Any]
    _module_to_component: dict[AudioModule, AudioComponent]

    def _cache_component(self, module: AudioModule, component: Any) -> None: ...

    def _build_chain_from_module(
        self, module: AudioModule
    ) -> AudioComponent | None: ...

    def _build_component_from_port(self, connection: PortWidget) -> Any | None: ...

    def _collect_input_components_from_port(
        self, port: PortWidget, track_skipped: bool = False
    ) -> tuple[list[Any], list[str]]: ...

    def _insert_cv_scaler_if_needed(
        self,
        mod_component: Any,
        source_module: AudioModule,
        target_module: AudioModule,
        port_name: str,
    ) -> Any: ...

    def _collect_modulation_components(self, module: AudioModule) -> dict[str, Any]: ...

    def _find_port_by_name(
        self, module: AudioModule, port_name: str
    ) -> PortWidget | None: ...

    def _find_connection_to_port(self, port: PortWidget) -> PortWidget | None: ...

    def _find_all_connections_to_port(self, port: PortWidget) -> list[PortWidget]: ...

    def _build_tree_node(
        self, module: AudioModule, visited: set[int]
    ) -> dict[str, Any]: ...
