"""Runtime validation helpers for engine-level DSP contracts."""

from __future__ import annotations

import math
import operator
from numbers import Real
from typing import Any


def validate_sample_rate(value: Any, *, name: str = "sample_rate") -> float:
    """Return a finite, positive sample rate as ``float``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")

    sample_rate = float(value)
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")

    return sample_rate


def validate_sample_count(value: Any, *, name: str = "n") -> int:
    """Return a non-negative integer sample count."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")

    try:
        sample_count = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}"
        ) from exc

    if sample_count < 0:
        raise ValueError(f"{name} must be non-negative, got {sample_count}")

    return sample_count
