"""Small shared helpers for module-owned runtime processors.

Engine-domain buffer/CV helpers come from ``soniclab``; port/parameter helpers
that depend on SonicRack's graph model stay here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from soniclab.utils.buffers import as_samples, silence
from soniclab.utils.cv import apply_cv_influence, gate_transition_indices
from soniclab.utils.ramping import ramp_if_changed

if TYPE_CHECKING:
    from sonicrack.patching.port import Port

RuntimeParameters = Mapping[str, object]

# Shared immutable empty mapping for modules with no runtime parameters.
EMPTY_PARAMETERS: RuntimeParameters = MappingProxyType({})


def read_samples(port: Port, num_samples: int) -> np.ndarray:
    return as_samples(port.read(num_samples), num_samples)


def read_optional_port(port: Any | None, num_samples: int) -> np.ndarray | None:
    """Read a port when connected; otherwise return None.

    Safe for missing ports and for objects that expose ``is_connected`` either
    as a property on the port model or via a thin widget wrapper.
    """
    if port is None or not getattr(port, "is_connected", False):
        return None
    return read_samples(port, num_samples)


# Alias used by effect CV helpers / older call sites.
read_optional_cv = read_optional_port


def write_silence_if_disconnected(
    input_port: Any,
    output_port: Any,
    num_samples: int,
) -> bool:
    """Write silence and return True when the input port is not connected."""
    if getattr(input_port, "is_connected", False):
        return False
    output_port.write(silence(num_samples))
    return True


def parameter(
    parameters: RuntimeParameters,
    name: str,
    fallback_getter: Callable[[], object],
) -> object:
    if name in parameters:
        return parameters[name]
    return fallback_getter()


def float_parameter(
    parameters: RuntimeParameters,
    name: str,
    fallback_getter: Callable[[], object],
) -> float:
    value = parameter(parameters, name, fallback_getter)
    return float(cast(float | int | str, value))


def str_parameter(
    parameters: RuntimeParameters,
    name: str,
    fallback_getter: Callable[[], object],
) -> str:
    return str(parameter(parameters, name, fallback_getter))


def write_output(
    output_port: Port, value: float | np.ndarray, num_samples: int
) -> None:
    if value is None:
        output_port.write(silence(num_samples))
    elif isinstance(value, tuple):
        output_port.write(np.asarray(value, dtype=np.float32))
    else:
        output_port.write(value)


__all__ = [
    "EMPTY_PARAMETERS",
    "RuntimeParameters",
    "apply_cv_influence",
    "as_samples",
    "float_parameter",
    "gate_transition_indices",
    "parameter",
    "ramp_if_changed",
    "read_optional_cv",
    "read_optional_port",
    "read_samples",
    "silence",
    "str_parameter",
    "write_output",
    "write_silence_if_disconnected",
]
