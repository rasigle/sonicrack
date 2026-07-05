"""Frequency modifier."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

import numpy as np

from src.engine.core.component import ComponentDescriptor, ParameterDescriptor
from src.engine.core.registry import ComponentCategory, register_component
from src.engine.dsp.modifiers.base import Modifier
from src.engine.utils.validation import validate_numeric

logger = logging.getLogger(__name__)


@register_component()
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

        self.frequency = frequency
        logger.debug("Frequency initialized with multiplier: %s", self.frequency)

    @property
    def frequency(self) -> float:
        """Current frequency multiplier."""
        return self._frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        value = validate_numeric(value, "frequency")
        if value < 0:
            raise ValueError(f"frequency must be non-negative, got {value}")
        self._frequency = float(value)

    def __call__(
        self, val: float | tuple[float, ...] | np.ndarray
    ) -> float | tuple[float, ...] | np.ndarray:
        """Apply frequency scaling to input.

        Args:
            val: Input value (float, numeric iterable, or array).

        Returns:
            Scaled value (same type as input).

        Raises:
            TypeError: If input is not int, float, array, or Iterable.
        """
        # Scalar input
        if isinstance(val, (float, int, np.number)):
            return float(val * self.frequency)

        # Vectorized input
        if isinstance(val, np.ndarray):
            return val * self.frequency

        if isinstance(val, (str, bytes, Mapping)):
            logger.error("Invalid input type for Frequency: %s", type(val))
            raise TypeError(
                "Input value must be an int, float, numpy number, array, "
                f"or numeric Iterable. Got {type(val)}"
            )

        if isinstance(val, Iterable):
            try:
                return tuple(
                    float(validate_numeric(v, "val item")) * self.frequency for v in val
                )
            except (TypeError, ValueError) as exc:
                raise TypeError("All Iterable input values must be numeric.") from exc

        logger.error("Invalid input type for Frequency: %s", type(val))
        raise TypeError(
            f"Input value must be an int, float, numpy number, array, or Iterable. "
            f"Got {type(val)}"
        )

    def scale_vectorized(self, samples: np.ndarray) -> np.ndarray:
        """Apply frequency scaling to array of samples (vectorized).

        Args:
            samples: Input array.

        Returns:
            Scaled array (float32).
        """
        return (samples * self.frequency).astype(np.float32)
