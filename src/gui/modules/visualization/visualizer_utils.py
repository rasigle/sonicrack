"""Shared utilities for passive visualization modules.

Visualizer widgets never render upstream modules themselves. Audio output and
the monitor timer both render through ``AudioEngine``; visualizers only read
recent tap history from connected ports. This keeps scopes/meters from changing
oscillator, envelope, or effect state while still allowing silent patches such
as ``Oscillator -> Waveform`` to animate without an Output module.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PyQt6.QtCore import QTimer

if TYPE_CHECKING:
    from gui.core import Port

logger = logging.getLogger(__name__)


def stop_visualizer_timer(module: Any, callback: Callable[[], None]) -> None:
    """Stop and detach a module's visualization timer if it exists."""
    timer = getattr(module, "_viz_timer", None)
    if timer is None:
        return

    timer = cast(QTimer, timer)
    if timer.isActive():
        timer.stop()

    with contextlib.suppress(TypeError):
        timer.timeout.disconnect(callback)

    module._viz_timer = None


def get_visualizer_samples(
    input_port: Port, num_samples: int | None = None
) -> np.ndarray | None:
    """Read cached port tap history.

    The audio callback or monitor timer has already rendered the graph and
    written samples to the connected output port. Visualizers copy that history
    without calling ``process()`` or advancing upstream DSP state.

    Args:
        input_port: The input port to read from
        num_samples: Optional number of recent samples to return

    Returns:
        Audio samples as numpy array, or None if unavailable
    """
    if not input_port.is_connected:
        return None

    try:
        for connected_port in input_port.connected_to:
            samples = connected_port.peek_recent(num_samples)

            if samples is None:
                continue

            if isinstance(samples, (int, float)):
                return np.array([samples], dtype=np.float32)

            arr = np.asarray(samples, dtype=np.float32)

            if arr.size > 0:
                return arr

        logger.debug("No valid tap samples found in connected ports")

    except Exception:
        logger.debug("Error reading cached visualizer samples", exc_info=True)

    return None


def validate_samples(samples: Any) -> bool:
    """Check if samples are valid for visualization."""
    if samples is None:
        return False

    if isinstance(samples, (int, float)):
        return np.isfinite(samples)

    if isinstance(samples, np.ndarray):
        return samples.size > 0 and np.any(np.isfinite(samples))

    return False
