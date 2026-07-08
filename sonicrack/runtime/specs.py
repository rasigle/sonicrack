"""Runtime dispatch contracts for GUI patch modules.

The audio engine owns graph traversal and render order. Individual modules own
their runtime spec and DSP adapter through ``ModuleWidget.get_runtime_spec()``
and ``process_runtime()``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from sonicrack.patching.module import ModuleCategory

if TYPE_CHECKING:
    from sonicrack.gui.widgets.port_widget import PortWidget
    from sonicrack.patching.module import ModuleMetadata

logger = logging.getLogger(__name__)

RuntimeParameters = Mapping[str, object]
RuntimeProcessor = Callable[[int, RuntimeParameters], None]


class RuntimeWidget(Protocol):
    """Common widget contract used by the runtime dispatcher."""

    runtime_kind: str
    is_active: bool
    metadata: ModuleMetadata
    output_ports: list[PortWidget]

    def get_runtime_spec(self) -> RuntimeModuleSpec: ...

    def get_required_inputs(self) -> list[str]: ...

    def _find_port_by_name(self, port_name: str) -> PortWidget | None: ...


@dataclass(frozen=True)
class RuntimeModuleSpec:
    """Declarative runtime behavior for a GUI module."""

    kind: str
    processor: RuntimeProcessor | None = field(default=None, compare=False)
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    parameter_names: tuple[str, ...] = ()


def get_runtime_spec(module: RuntimeWidget) -> RuntimeModuleSpec:
    """Return a module's declarative runtime spec."""
    return module.get_runtime_spec()


def process_runtime_module(
    module: RuntimeWidget,
    num_samples: int,
    spec: RuntimeModuleSpec | None = None,
    parameters: RuntimeParameters | None = None,
) -> None:
    """Render one GUI module through the engine-owned runtime path."""
    if not module.is_active:
        _process_inactive(module, num_samples)
        return

    spec = spec or get_runtime_spec(module)
    parameters = parameters or {}

    if spec.processor is not None:
        spec.processor(num_samples, parameters)
        return

    logger.debug(
        "No runtime node for %s with spec kind %r", type(module).__name__, spec.kind
    )


def _process_inactive(module: RuntimeWidget, num_samples: int) -> None:
    if module.metadata.category == ModuleCategory.MODIFIER:
        required_inputs = module.get_required_inputs()
        if required_inputs:
            input_port = module._find_port_by_name(required_inputs[0])
            if input_port is not None:
                value = input_port.port.read(num_samples)
                for output_port in module.output_ports:
                    output_port.write(value)
                return

    silence = np.zeros(num_samples, dtype=np.float32)
    for output_port in module.output_ports:
        output_port.write(silence)
