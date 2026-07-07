"""Runtime validation helpers for engine-level DSP contracts."""

from __future__ import annotations

import logging
import math
import operator
from numbers import Real
from typing import Any

import numpy as np

from src.engine.utils.math import db_to_linear

logger = logging.getLogger(__name__)


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


def validate_numeric_range(
    value: Any,
    minimum: float,
    maximum: float,
    *,
    name: str,
) -> float:
    """Return a finite real number inside an inclusive range."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")

    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, got {value!r}"
        )

    return result


def is_number(value: Any) -> bool:
    """Return True for Python and NumPy scalar numbers."""
    return isinstance(value, (int, float, np.number))


def validate_numeric(value: Any, name: str) -> float:
    """Validate and normalize a numeric scalar to float."""
    if not is_number(value):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def _validate_amplitude_args(amplitude: float | None, gain_db: float | None) -> None:
    """Validate the types/ranges of amplitude and gain_db constructor args."""
    if amplitude is not None:
        if isinstance(amplitude, bool) or not isinstance(
            amplitude, (int, float, np.number)
        ):
            raise TypeError(
                f"amplitude must be a number, got {type(amplitude).__name__}"
            )
        if amplitude < 0.0:
            raise ValueError(f"amplitude must be non-negative, got {amplitude}")

    if gain_db is not None and (
        isinstance(gain_db, bool) or not isinstance(gain_db, (int, float, np.number))
    ):
        raise TypeError(f"gain_db must be a number, got {type(gain_db).__name__}")


def derive_amplitude_from_init(
    given_args: set[str], amplitude: float | None, gain_db: float | None
) -> float:
    """Determine the initial linear amplitude from constructor arguments.

    Precedence (highest to lowest):
      1. Explicitly-passed gain_db (wins even over an explicit amplitude;
         a conflict just gets a warning, not an error).
      2. Explicitly-passed amplitude.
      3. Default (non-explicit) gain_db, if the caller's default is not None.
      4. Fallback of 1.0 (unity amplitude).

    Note that a *default* gain_db outranks a *default* amplitude in step 3 —
    this is intentional and easy to get backwards if this function is
    refactored, since it's the least obvious branch.
    """
    _validate_amplitude_args(amplitude, gain_db)

    gain_db_set = "gain_db" in given_args
    amplitude_set = "amplitude" in given_args

    if gain_db_set and gain_db is not None:
        expected_amp = float(db_to_linear(gain_db))
        if (
            amplitude_set
            and amplitude is not None
            and not np.isclose(amplitude, expected_amp)
        ):
            logger.warning(
                "Both gain_db=%s and amplitude=%s were specified. "
                "Using gain_db, which results in an amplitude of %.3f.",
                gain_db,
                amplitude,
                expected_amp,
            )
        return expected_amp

    if amplitude_set and amplitude is not None:
        return amplitude

    if gain_db is not None:
        return float(db_to_linear(gain_db))

    return 1.0
