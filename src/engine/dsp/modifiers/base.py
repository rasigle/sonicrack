"""Base classes and utilities for signal modifiers."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

import numpy as np

from src.engine.core.component import AudioComponent

logger = logging.getLogger(__name__)


def _validate_modulator(modulator: Any) -> None:
    """Validate that a modulator is usable.

    Args:
        modulator: The modulator to validate.

    Raises:
        TypeError: If modulator is None or not iterable/iterator.
    """
    if modulator is None:
        raise TypeError("modulator cannot be None")
    if not hasattr(modulator, "__iter__") and not hasattr(modulator, "__next__"):
        raise TypeError(
            f"modulator must be iterable or have __next__, got "
            f"{type(modulator).__name__}"
        )


def _get_modulation_values(
    modulator_source: Any, modulator_iter: Any, num_samples: int
) -> np.ndarray:
    """Get modulation values for vectorized processing.

    Tries to get samples from the modulator in the most efficient way:
    1. Call get_samples() on the source (preferred - usually vectorized)
    2. Call get_samples() on the iterator (if available)
    3. Fall back to iterating manually

    Args:
        modulator_source: The original modulator source object.
        modulator_iter: The iterator instance.
        num_samples: Number of modulation values to retrieve.

    Returns:
        Array of modulation values.
    """
    # Prefer calling `get_samples` on the original source if available
    if hasattr(modulator_source, "get_samples"):
        return modulator_source.get_samples(num_samples, mode="vectorized")

    if hasattr(modulator_iter, "get_samples"):
        # Iterator might itself expose get_samples
        return modulator_iter.get_samples(num_samples, mode="vectorized")

    # Fallback to iterator if vectorization not available
    return np.array(
        [next(modulator_iter) for _ in range(num_samples)], dtype=np.float32
    )


def _get_next_modulation_value(
    modulator_source: Any, modulator_iter: Any
) -> tuple[float, Any]:
    """Get the next modulation value for scalar processing.

    Handles iterator exhaustion by recreating the iterator from the source.

    Args:
        modulator_source: The original modulator source object.
        modulator_iter: The current iterator instance.

    Returns:
        Tuple of (next_value, potentially_new_iterator).
    """
    try:
        return next(modulator_iter), modulator_iter
    except StopIteration:
        # Re-create iterator and advance
        new_iter = iter(modulator_source)
        return next(new_iter), new_iter


class Modifier(AudioComponent):
    """Base for components that modify signals (effects, filters)."""

    @abstractmethod
    def __call__(self, val: Any) -> Any:
        """Apply modification to a value or a bunch of values.

        Args:
            val: Input value (mono float or stereo tuple).

        Returns:
            Modified value (same type as input).
        """

    def process_block(self, samples: np.ndarray) -> Any:
        """Process an audio/control block through the modifier.

        ``__call__`` remains the scalar-friendly compatibility API. New engine
        routing code should prefer this method when processing buffers so all
        modifiers, filters, and effects expose the same block entry point.
        """
        return self(samples)
