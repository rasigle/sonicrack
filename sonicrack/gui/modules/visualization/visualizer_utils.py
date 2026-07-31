"""Shared utilities for visualization modules.

Visualizer widgets never render upstream modules themselves for display.
Audio output and the monitor timer both render through ``AudioEngine``;
scopes only read recent tap history from connected ports. That keeps meters
from advancing oscillator/effect state while silent patches such as
``Oscillator -> Waveform`` still animate without an Output module.

When used inline (``Source -> Waveform -> next``), the same modules also act
as pass-through processors: ``process_visualizer_passthrough`` copies the
input buffer to the output unchanged so the signal can be picked again.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PyQt6.QtCore import QTimer

from sonicrack.runtime.helpers import read_samples, write_silence_if_disconnected

if TYPE_CHECKING:
    from sonicrack.patching.port import Port

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


def process_visualizer_passthrough(
    input_port: Port,
    output_port: Port,
    num_samples: int,
) -> None:
    """Route the input signal to the output unchanged.

    Used by Waveform/Spectrum when they sit inline in a patch. Display still
    reads tap history separately; this only provides a pickable thru jack.
    """
    if write_silence_if_disconnected(input_port, output_port, num_samples):
        return

    samples = read_samples(input_port, num_samples)
    # Own the output buffer so later modules cannot mutate the input port value.
    if isinstance(samples, np.ndarray):
        output_port.write(samples.copy())
    else:
        output_port.write(samples)


def validate_samples(samples: Any) -> bool:
    """Check if samples are valid for visualization."""
    if samples is None:
        return False

    if isinstance(samples, (int, float)):
        return bool(np.isfinite(samples))

    if isinstance(samples, np.ndarray):
        return bool(samples.size > 0 and np.any(np.isfinite(samples)))

    return False
