"""Audio engine for real-time synthesis and playback.

NOTE: This engine now uses a pull-based architecture where the audio callback
(in OutputModule) directly drives all processing. No QTimer is needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PyQt6 import QtCore

from src.constants import DEFAULT_SAMPLE_RATE
from src.gui.audio_config import audio_config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AudioEngine(QtCore.QObject):
    """Audio engine coordinates module management and cache invalidation.

    In the new pull-based architecture:
    - Audio callback (OutputModule) directly pulls samples from connected modules
    - Port.read() triggers upstream module generation
    - AudioEngine just manages module lifecycle and cache invalidation
    """

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, buffer_size: int = 2048):
        """Initialize the audio engine."""
        super().__init__()

        self.sample_rate: float = sample_rate
        self.buffer_size: float = buffer_size

        self.modules: list[Any] = []
        self.connections: list[Any] = []

        # No QTimer needed - audio callback drives processing now!
        audio_config.add_sample_rate_listener(self._on_config_changed)
        audio_config.add_buffer_size_listener(self._on_config_changed)

    def add_module(self, mod):
        """Add a module to the engine.

        Args:
            mod: Module to add
        """
        self.modules.append(mod)

    def _on_config_changed(self, value):
        """Handle sample rate or buffer size changes.

        Just update local values - no timer to update anymore.
        """
        _ = value
        self.sample_rate = audio_config.sample_rate
        self.buffer_size = audio_config.buffer_size
        logger.debug(
            f"AudioEngine config updated: SR={self.sample_rate}, BS={self.buffer_size}"
        )

    def invalidate_all_caches(self):
        """Invalidate all module caches.

        Called at the start of each audio processing cycle by OutputModule.
        """
        for module in self.modules:
            if hasattr(module, "invalidate_cache"):
                module.invalidate_cache()
