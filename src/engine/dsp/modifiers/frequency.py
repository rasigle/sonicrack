"""Frequency modifier."""

from __future__ import annotations

import logging

import numpy as np

from src.engine.core.component import ComponentDescriptor, ParameterDescriptor
from src.engine.core.registry import ComponentCategory, register_component
from src.engine.dsp.modifiers.base import Modifier
from src.engine.utils.validation import is_number, validate_numeric

logger = logging.getLogger(__name__)

# Precompute common type checks to avoid repeated tuple creation in hot paths
_NP_NUMBER = np.number
_SCALAR_TYPES = (float, int, _NP_NUMBER)


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

    def __call__(self, val: float | tuple | np.ndarray) -> float | tuple | np.ndarray:
        """Apply frequency scaling to input.

        Fast paths for the common cases (scalar, ndarray, tuple).
        Lists and other iterables are converted to ndarray for batch processing.

        Args:
            val: Input value (float, numeric iterable, or array).

        Returns:
            Scaled value (same type as input).

        Raises:
            TypeError: If input is not a numeric type.
        """
        freq = self._frequency  # local lookup avoids repeated attribute access

        # Fast path: scalar (most common in hot loops)
        if isinstance(val, _SCALAR_TYPES):
            return float(val * freq)

        # Fast path: ndarray (second most common)
        if isinstance(val, np.ndarray):
            return (val * freq).astype(np.float32)

        # Fast path: tuple (stereo pairs, control signals)
        if isinstance(val, tuple):
            # Check if tuple contains only scalars (common case)
            if val and isinstance(val[0], _SCALAR_TYPES):
                return tuple(v * freq for v in val)
            # Fallback: tuple of iterables — convert to array
            arr = np.asarray(val)
            return (arr * freq).astype(np.float32)

        # Convert list/other iterables to ndarray for batch processing
        # (avoids slow per-element Python loop + validation)
        try:
            arr = np.asarray(val, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise TypeError(
                f"Input value must be a numeric scalar, tuple, array, "
                f"or numeric Iterable. Got {type(val)}"
            ) from exc

        if arr.dtype.kind not in ("f", "i", "u", "c"):
            raise TypeError(
                f"Input value must contain numeric values. Got {type(val)}"
            )

        return (arr * freq).astype(np.float32)

    def scale_vectorized(self, samples: np.ndarray) -> np.ndarray:
        """Apply frequency scaling to array of samples (vectorized).

        Args:
            samples: Input array.

        Returns:
            Scaled array (float32).
        """
        return (samples * self._frequency).astype(np.float32)

    def process_block(self, samples: np.ndarray) -> np.ndarray:
        """Process an audio/control block through the modifier.

        Optimized path for numpy arrays — avoids type dispatch overhead
        of __call__ and ensures float32 output for the audio pipeline.

        Args:
            samples: Input array (float32 or float64).

        Returns:
            Scaled array (float32).
        """
        return (samples * self._frequency).astype(np.float32)
