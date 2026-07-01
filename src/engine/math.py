"""Math utility functions for audio processing."""

import math

import numpy as np


def db_to_linear(db: float | np.ndarray) -> float | np.ndarray:
    """Convert decibels to linear amplitude.

    Standard audio conversion using the formula: amplitude = 10^(dB/20)

    Args:
        db: Gain in decibels (scalar or array)

    Returns:
        Linear amplitude (same type as input)

    Examples:
        >>> db_to_linear(0)    # 1.0 (unity gain)
        >>> db_to_linear(-6)   # ~0.5 (half amplitude)
        >>> db_to_linear(-20)  # 0.1 (1/10 amplitude)
        >>> db_to_linear(6)    # ~2.0 (double amplitude)
    """
    return 10 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels.

    Standard audio conversion using the formula: dB = 20 * log10(amplitude)

    Args:
        linear: Linear amplitude (must be > 0)

    Returns:
        Gain in decibels (-inf for zero or negative)

    Examples:
        >>> linear_to_db(1.0)   # 0 dB (unity gain)
        >>> linear_to_db(0.5)   # ~-6 dB (half amplitude)
        >>> linear_to_db(0.1)   # -20 dB (1/10 amplitude)
        >>> linear_to_db(0.0)   # -inf (silence)
    """
    if linear <= 0:
        return -math.inf
    return 20 * np.log10(linear)


def squish_val(val, min_val=0, max_val=1):
    """Map a value in [-1, 1] to a range [min_val, max_val].

    Args:
        val (float): Value expected roughly in [-1, 1].
        min_val (float, optional): Minimum of target range. Defaults to 0.
        max_val (float, optional): Maximum of target range. Defaults to 1.

    Returns:
        float: Rescaled value in [min_val, max_val].
    """
    return (((val + 1) / 2) * (max_val - min_val)) + min_val
