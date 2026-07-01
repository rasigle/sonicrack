"""Frequency modifier."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np

from src.engine.core.component import (
    ComponentDescriptor,
    ParameterDescriptor,
)
from src.engine.core.registry import ComponentCategory
from src.engine.dsp.modifiers.base import (
    Modifier,
)

logger = logging.getLogger(__name__)


class Frequency(Modifier):
    """Scales the input values by frequency multiplier.

    Can be used to increase or decrease frequency-related values.

    Args:
        frequency: Frequency multiplier. 1.0=no change. Defaults to 1.0.

    Attributes:
        frequency: Current frequency multiplier.
    """

    descriptor = ComponentDescriptor(
        name="Frequency",
        category=ComponentCategory.MODIFIER,
        description="Frequency scaling modifier",
        fluent_api_name="frequency",
        parameters={
            "frequency": ParameterDescriptor(
                name="frequency",
                default=1.0,
                minimum=0.0,
                description="Frequency multiplier.",
            )
        },
        tags=["modifier", "frequency"],
    )

    def __init__(self, frequency: float = 1.0):
        """Initialize frequency modifier.

        Args:
            frequency: Sets the frequency multiplier for the
                input signal (1 : no change, 0 : no output).

        Raises:
            TypeError: If frequency is not a number.
            ValueError: If frequency is negative.
        """
        super().__init__()

        # Input validation
        if not isinstance(frequency, (int, float, np.number)):
            raise TypeError(
                f"frequency must be a number, got {type(frequency).__name__}"
            )
        if frequency < 0:
            raise ValueError(f"frequency must be non-negative, got {frequency}")

        self.frequency = frequency
        logger.debug(f"Frequency initialized with multiplier: {frequency}")

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Apply frequency scaling to input.

        Args:
            val: Input value (float, tuple, or array).

        Returns:
            Scaled value (same type as input).

        Raises:
            TypeError: If input is not int, float, array, or Iterable.
        """
        # Optimize: check array first
        if isinstance(val, np.ndarray):
            return val * self.frequency

        if isinstance(val, Iterable):
            return tuple(v * self.frequency for v in val)

        if isinstance(val, (int, float, np.number)):
            return val * self.frequency

        raise TypeError("Input value must be an int, float, numpy array, or Iterable.")

    def scale_vectorized(self, samples: np.ndarray) -> np.ndarray:
        """Apply frequency scaling to array of samples (vectorized).

        Args:
            samples: Input array.

        Returns:
            Scaled array (float32).
        """
        return (samples * self.frequency).astype(np.float32)
