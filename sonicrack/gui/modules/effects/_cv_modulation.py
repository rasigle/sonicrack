"""Shared CV modulation helpers for effect modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from sonicrack.runtime.helpers import read_samples

if TYPE_CHECKING:
    from sonicrack.patching.port import Port


def read_optional_cv(port: Port | None, num_samples: int) -> np.ndarray | None:
    """Read a CV port when connected, otherwise return None."""
    if port is None or not port.is_connected:
        return None
    return read_samples(port, num_samples)


def control_rate_offset(
    cv: np.ndarray | None,
    *,
    scale: float = 1.0,
) -> float:
    """Convert a CV buffer to a single bipolar offset (mean of the buffer).

    Stateful effects (delay, reverb, compressor) apply modulation at control
    rate once per render block rather than retuning every sample.
    """
    if cv is None:
        return 0.0
    values = np.asarray(cv, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.mean(values)) * float(scale)


def modulate_param(
    base: float,
    cv: np.ndarray | None,
    *,
    minimum: float,
    maximum: float,
    scale: float = 1.0,
) -> float:
    """Apply a bipolar CV offset to a scalar parameter and clamp."""
    return float(
        np.clip(base + control_rate_offset(cv, scale=scale), minimum, maximum)
    )


def any_cv_connected(ports: Sequence[Port | None]) -> bool:
    """Return True if any provided port is connected."""
    return any(port is not None and port.is_connected for port in ports)
