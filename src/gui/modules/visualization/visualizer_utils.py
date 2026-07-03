"""Shared utilities for passive visualization modules.

Visualizer widgets never render upstream modules themselves. Audio output and
the monitor timer both render through ``AudioEngine``; visualizers only read
recent tap history from connected ports. This keeps scopes/meters from changing
oscillator, envelope, or effect state while still allowing silent patches such
as ``Oscillator -> Waveform`` to animate without an Output module.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def stop_visualizer_timer(module: Any, callback: Callable[[], None]) -> None:
    """Stop and detach a module's visualization timer if it exists."""
    timer = getattr(module, "_viz_timer", None)
    if timer is None:
        return

    if timer.isActive():
        timer.stop()
    try:
        timer.timeout.disconnect(callback)
    except TypeError:
        pass


def get_visualizer_samples(
    input_port, num_samples: int | None = None
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
            if samples is not None:
                if isinstance(samples, np.ndarray) and samples.size > 0:
                    logger.debug(
                        f"PASSIVE: Got {len(samples)} samples from port "
                        f"{connected_port.port_name}"
                    )
                    return samples
                elif isinstance(samples, (int, float)):
                    # Scalar value - convert to small array for visualization
                    scalar_samples = np.array([samples], dtype=np.float32)
                    logger.debug(
                        f"PASSIVE: Got scalar {samples} "
                        f"from port {connected_port.port_name}"
                    )
                    return scalar_samples
        logger.debug("No valid tap samples found in connected ports")
    except Exception as e:
        logger.debug(f"Error reading cached visualizer samples: {e}")

    return None


def validate_samples(samples) -> bool:
    """Check if samples are valid for visualization.

    Args:
        samples: The samples to validate

    Returns:
        True if samples are valid and non-empty, False otherwise
    """
    if samples is None:
        return False

    if isinstance(samples, (int, float)):
        return samples != 0.0

    if isinstance(samples, np.ndarray):
        return samples.size > 0

    return False
