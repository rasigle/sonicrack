"""Audio engine for real-time synthesis and playback.

NOTE: This engine now uses a pull-based architecture where the audio callback
(in OutputModule) directly drives all processing. No QTimer is needed.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING

from PyQt6 import QtCore

from src.constants import DEFAULT_SAMPLE_RATE
from src.gui.audio_config import audio_config

if TYPE_CHECKING:
    from src.gui.core.module import AudioModule

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

        self.modules = []
        self.connections = []

        # No QTimer needed - audio callback drives processing now!
        audio_config.add_sample_rate_listener(self._on_config_changed)
        audio_config.add_buffer_size_listener(self._on_config_changed)

    def add_module(self, mod):
        """Add a module to the engine.

        Args:
            mod: Module to add
        """
        self.modules.append(mod)

    def process(self, num_samples: int | None = None):
        """Process all modules in topological order.

        NOTE: This is kept for debugging/testing only. In the pull-based architecture,
        the audio callback (OutputModule) drives processing by pulling samples through
        Port.read() which triggers upstream modules.

        This method is no longer used in normal operation.
        """
        if num_samples is None:
            num_samples = audio_config.buffer_size

        logger.warning("AudioEngine.process() called - this should not happen in pull-based architecture")
        ordered = self._build_graph()
        for module in ordered:
            if hasattr(module, 'process'):
                module.process(num_samples)

    def _build_graph(self) -> list[AudioModule]:
        """Build a topologically ordered list of audio components."""
        graph = defaultdict(set)
        indegree = defaultdict(int)

        for m in self.modules:
            indegree[m] = 0

        for src, outp, dst, inp in self.connections:
            graph[src].add(dst)
            indegree[dst] += 1

        # topological sort
        queue = deque([m for m in self.modules if indegree[m] == 0])
        order = []
        while queue:
            m = queue.popleft()
            order.append(m)
            for nbr in graph[m]:
                indegree[nbr] -= 1
                if indegree[nbr] == 0:
                    queue.append(nbr)

        return order

    def _on_config_changed(self, value):
        """Handle sample rate or buffer size changes.

        Just update local values - no timer to update anymore.
        """
        self.sample_rate = audio_config.sample_rate
        self.buffer_size = audio_config.buffer_size
        logger.debug(f"AudioEngine config updated: SR={self.sample_rate}, BS={self.buffer_size}")

    def invalidate_all_caches(self):
        """Invalidate all module caches.

        Called at the start of each audio processing cycle by OutputModule.
        """
        for module in self.modules:
            if hasattr(module, 'invalidate_cache'):
                module.invalidate_cache()

    def start(self):
        """Start audio processing.

        NOTE: This is now a no-op. Audio starts when OutputModule.start_playback() is called,
        which starts the audio callback that drives everything.
        """
        logger.info("AudioEngine.start() - Audio is callback-driven, no timer to start")

    def stop(self):
        """Stop audio processing.

        NOTE: This is now a no-op. Audio stops when OutputModule.stop_playback() is called.
        """
        logger.info("AudioEngine.stop() - Audio is callback-driven, no timer to stop")

    def set_sample_rate(self, value: int):
        """Set sample rate (for backward compatibility)."""
        self.sample_rate = value

    def set_buffer_size(self, value: int):
        """Set buffer size (for backward compatibility)."""
        self.buffer_size = value
