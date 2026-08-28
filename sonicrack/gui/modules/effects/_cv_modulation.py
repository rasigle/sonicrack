"""Shared CV modulation helpers for effect modules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from sonicrack.runtime.helpers import float_parameter, read_optional_cv
from sonicrack.runtime.specs import RuntimeParameters

if TYPE_CHECKING:
    from sonicrack.patching.port import Port

# Re-export for existing importers.
__all__ = [
    "ControlRateCvSpec",
    "apply_control_rate_cv",
    "control_rate_offset",
    "modulate_param",
    "read_optional_cv",
]


@dataclass(frozen=True, slots=True)
class ControlRateCvSpec:
    """Describe one control-rate CV-modulated parameter on a DSP component."""

    attr: str
    param_name: str
    fallback: Callable[[], float]
    port: Port | None
    minimum: float
    maximum: float
    scale: float = 1.0


def control_rate_offset(
    cv: np.ndarray | None,
    *,
    scale: float = 1.0,
) -> float:
    """Convert a CV buffer to a single bipolar offset (first sample).

    Stateful effects (delay, reverb, compressor, distortion) apply modulation
    at control rate once per render block rather than retuning every sample.
    Using the first sample avoids a full-buffer reduction each block.
    """
    if cv is None:
        return 0.0
    if cv.size == 0:
        return 0.0
    return float(cv.flat[0]) * float(scale)


def modulate_param(
    base: float,
    cv: np.ndarray | None,
    *,
    minimum: float,
    maximum: float,
    scale: float = 1.0,
) -> float:
    """Apply a bipolar CV offset to a scalar parameter and clamp."""
    value = base + control_rate_offset(cv, scale=scale)
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return float(value)


def apply_control_rate_cv(
    component: Any,
    parameters: RuntimeParameters,
    specs: Sequence[ControlRateCvSpec],
    num_samples: int,
) -> None:
    """Apply a list of control-rate CV parameter specs to a DSP component.

    Hot-path notes:
    - Unconnected CV ports skip the buffer read entirely.
    - ``setattr`` is only issued when the resolved value changed, avoiding
      redundant RuntimeParameter target updates every audio block.
    """
    for spec in specs:
        base = float_parameter(parameters, spec.param_name, spec.fallback)
        port = spec.port
        cv = None
        if port is not None and getattr(port, "is_connected", False):
            cv = read_optional_cv(port, num_samples)
        value = modulate_param(
            base,
            cv,
            minimum=spec.minimum,
            maximum=spec.maximum,
            scale=spec.scale,
        )

        try:
            current = getattr(component, spec.attr)
        except AttributeError:
            current = None
        if current is None or current != value:
            setattr(component, spec.attr, value)
