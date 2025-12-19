"""Shared utilities for visualization modules.

This module provides common functionality for waveform and spectrum visualizers,
including hybrid active/passive mode sample acquisition.
"""

import logging
from typing import Optional

import numpy as np

from src.gui.widgets.module_widget import ModuleWidget

logger = logging.getLogger(__name__)


def check_output_module_exists(visualizer_module) -> bool:
    """Check if an output module exists in the scene.

    This is used to determine whether visualizer should be in passive mode
    (output exists, just read cached values) or active mode (no output,
    actively generate samples).

    Args:
        visualizer_module: The visualizer module instance (needs .scene() method)

    Returns:
        True if output module exists, False otherwise
    """
    if not hasattr(visualizer_module, 'scene'):
        return False

    scene = visualizer_module.scene()
    if not scene:
        return False

    # Check all items in scene for output module
    for item in scene.items():
        if isinstance(item, ModuleWidget) and hasattr(item, 'audio_output'):
            return True

    return False


def get_samples_passive_mode(input_port) -> Optional[np.ndarray]:
    """Get samples in PASSIVE mode - read cached port value.

    In passive mode, the audio thread has already written samples to port.value.
    We simply read and copy those samples. This is safe because we're only
    READING, not calling process() or generating new samples.

    This avoids race conditions with the audio thread.

    Args:
        input_port: The input port to read from

    Returns:
        Audio samples as numpy array, or None if unavailable
    """
    try:
        for connected_port in input_port.connected_to:
            if connected_port.value is not None:
                if isinstance(connected_port.value, np.ndarray) and connected_port.value.size > 0:
                    # Make a copy to avoid any threading issues
                    return connected_port.value.copy()
    except Exception as e:
        logger.debug(f"Error reading cached samples in passive mode: {e}")

    return None


def get_samples_active_mode(input_port, num_samples: int = 1024) -> Optional[np.ndarray]:
    """Get samples in ACTIVE mode - actively generate samples.

    In active mode (no output module exists), we directly call process() on
    upstream modules to generate samples. This is used for standalone
    visualization without audio output.

    Args:
        input_port: The input port to read from
        num_samples: Number of samples to request

    Returns:
        Audio samples as numpy array, or None if unavailable
    """
    try:
        samples = None

        # Directly trigger upstream module generation
        # (can't use port.read() because visualizers are marked non-processing)
        for connected_port in input_port.connected_to:
            if connected_port.parent_module:
                # Call process() directly on upstream module
                if hasattr(connected_port.parent_module, 'process'):
                    connected_port.parent_module.process(num_samples)

                # Now read the generated value from the port
                if connected_port.value is not None:
                    if isinstance(connected_port.value, np.ndarray) and connected_port.value.size > 0:
                        samples = connected_port.value
                        break

        # Fallback: try port.read() if direct call didn't work
        if samples is None:
            samples = input_port.read(num_samples)
            if not isinstance(samples, np.ndarray) or samples.size == 0:
                samples = None

        return samples

    except Exception as e:
        logger.debug(f"Error pulling samples in active mode: {e}")
        return None


def get_samples_hybrid(visualizer_module, input_port, num_samples: int = 1024) -> Optional[np.ndarray]:
    """Get samples using hybrid active/passive mode.

    This is the main entry point for visualizers. It automatically detects
    whether to use passive or active mode based on whether an output module
    exists in the scene.

    **Passive Mode (output exists):**
    - Just reads port.value that was already written by audio thread
    - No race conditions, zero audio interference
    - Safe concurrent access

    **Active Mode (standalone):**
    - Directly calls process() on upstream modules
    - Enables visualization without output module
    - Used for monitoring generators directly

    Args:
        visualizer_module: The visualizer module (needs .scene() method)
        input_port: The input port to read from
        num_samples: Number of samples to request (used in active mode)

    Returns:
        Audio samples as numpy array, or None if unavailable
    """
    # Check if input is connected
    if not hasattr(input_port, 'is_connected') or not input_port.is_connected:
        return None

    # Determine mode based on output module presence
    output_exists = check_output_module_exists(visualizer_module)

    if output_exists:
        # PASSIVE MODE: Read cached value (no generation)
        return get_samples_passive_mode(input_port)
    else:
        # ACTIVE MODE: Generate samples (standalone)
        return get_samples_active_mode(input_port, num_samples)


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

